"""Service-level update semantics: repeat, reorder, boundedness, queries."""
import threading

import pytest

from voxel_occupancy.service import OccupancyService
from voxel_occupancy.types import (
    CellState,
    ExpiryPolicy,
    GridConfig,
    Observation,
)


def _occupied(service, cell):
    snap = service.get_cell(cell)
    return snap is not None and snap.state is CellState.OCCUPIED


def test_single_observation_affects_only_target_voxel():
    service = OccupancyService(GridConfig())
    service.submit([Observation((3, 3, 3), 0.95, 1)])
    assert _occupied(service, (3, 3, 3))
    # Neighbours remain unobserved.
    for cell in [(3, 3, 2), (2, 3, 3), (3, 0, 3), (0, 0, 0)]:
        assert service.get_cell(cell) is None


def test_repeated_observation_converges_but_stays_bounded():
    service = OccupancyService(GridConfig())
    for ts in range(1, 50):
        service.submit([Observation((0, 0, 0), 0.99, ts)])
    snap = service.get_cell((0, 0, 0))
    assert snap.state is CellState.OCCUPIED
    assert 0.0 < snap.probability < 1.0
    assert snap.probability == pytest.approx(
        1.0, abs=0.01
    )  # near, not equal to 1


def test_duplicate_and_stale_are_rejected_under_reject_policy():
    service = OccupancyService(GridConfig(expiry_policy=ExpiryPolicy.REJECT))
    first = service.submit([Observation((0, 0, 0), 0.9, 5)])
    assert first.applied == 1
    dup = service.submit([Observation((0, 0, 0), 0.99, 5)])
    old = service.submit([Observation((0, 0, 0), 0.99, 4)])
    assert dup.rejected_expired == 1
    assert old.rejected_expired == 1
    # Belief unchanged after rejected writes.
    assert service.get_cell((0, 0, 0)).update_count == 1


def test_out_of_order_batch_partially_applied():
    service = OccupancyService(GridConfig())
    result = service.submit(
        [
            Observation((0, 0, 0), 0.9, 10),
            Observation((0, 0, 0), 0.01, 3),  # stale reorder -> rejected
            Observation((1, 0, 0), 0.9, 10),
        ]
    )
    assert result.applied == 2
    assert result.rejected_expired == 1


def test_downweight_policy_reproducible():
    def run():
        service = OccupancyService(
            GridConfig(expiry_policy=ExpiryPolicy.DOWNWEIGHT, downweight_decay=0.6)
        )
        service.submit([Observation((0, 0, 0), 0.9, 10)])
        service.submit([Observation((0, 0, 0), 0.1, 8)])
        service.submit([Observation((0, 0, 0), 0.8, 12)])
        snap = service.get_cell((0, 0, 0))
        return (snap.log_odds, snap.update_count)

    assert run() == run()


def test_ignore_time_accumulates_everything():
    service = OccupancyService(
        GridConfig(expiry_policy=ExpiryPolicy.IGNORE_TIME)
    )
    service.submit([Observation((0, 0, 0), 0.9, 10)])
    result = service.submit([Observation((0, 0, 0), 0.9, 1)])
    assert result.applied == 1 and result.rejected_expired == 0
    assert service.get_cell((0, 0, 0)).update_count == 2


def test_region_and_ray_reflect_latest_commit_only():
    service = OccupancyService(GridConfig())
    v1 = service.submit([Observation((4, 0, 0), 0.95, 1)])
    assert service.query_ray((0, 0, 0), (6, 0, 0)).reachable is False
    assert service.snapshot_version() == v1.sequence
    # Clear it with strong free evidence in a later commit.
    for ts in range(2, 12):
        service.submit([Observation((4, 0, 0), 0.02, ts)])
    snap = service.get_cell((4, 0, 0))
    assert snap.state is CellState.FREE
    assert service.query_ray((0, 0, 0), (6, 0, 0)).reachable is True
    region = service.query_region((0, 0, 0), 4)
    assert (4, 0, 0) in region.cells
    assert region.cells[(4, 0, 0)].state is CellState.FREE


def test_query_never_sees_partial_commit_under_contention():
    """A query must always observe a fully committed version."""
    service = OccupancyService(GridConfig())
    stop = threading.Event()
    violations: list[int] = []
    target_cells = [(i, 0, 0) for i in range(20)]

    def writer():
        ts = 0
        while not stop.is_set():
            ts += 1
            batch = [Observation(c, 0.95, ts) for c in target_cells]
            service.submit(batch)

    def reader():
        while not stop.is_set():
            region = service.query_region((0, 0, 0), 20)
            occupied = [
                c for c in target_cells
                if c in region.cells
                and region.cells[c].state is CellState.OCCUPIED
            ]
            # Each commit sets all 20 cells together; a consistent snapshot can
            # only ever show 0 or 20 of them occupied (once any commit landed).
            if occupied and len(occupied) != len(target_cells):
                # Before the first full commit cells might be from prior
                # commits though - all target cells are always in same batch,
                # so any non-empty consistent view must contain all 20.
                violations.append(len(occupied))

    t = threading.Thread(target=writer)
    t.start()
    readers = [threading.Thread(target=reader) for _ in range(3)]
    for r in readers:
        r.start()
    timer = threading.Timer(0.5, stop.set)
    timer.start()
    t.join()
    for r in readers:
        r.join()
    timer.join()
    assert not violations, violations[:5]
