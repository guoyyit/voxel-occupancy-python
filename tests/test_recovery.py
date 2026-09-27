"""Restart recovery and crash-safety of the durable snapshot."""
import os

import pytest

from voxel_occupancy.persistence import CorruptStateError, PersistenceStore
from voxel_occupancy.service import OccupancyService
from voxel_occupancy.types import (
    CellState,
    GridConfig,
    Observation,
    PersistenceConfig,
)


def _service(path, **grid_kw):
    return OccupancyService.start(
        grid_config=GridConfig(**grid_kw),
        persistence_config=PersistenceConfig(path=path),
    )


def test_restart_recovers_last_committed_state(tmp_path):
    db = str(tmp_path / "state.bin")
    service = _service(db)
    service.submit([Observation((1, 2, 3), 0.95, 1)])
    service.submit([Observation((-1, 0, 4), 0.05, 2)])
    expected_version = service.snapshot_version()
    expected = {cell: service.get_cell(cell) for cell in [(1, 2, 3), (-1, 0, 4)]}
    del service

    restarted = _service(db)
    assert restarted.snapshot_version() == expected_version
    for cell, snap_before in expected.items():
        snap_after = restarted.get_cell(cell)
        assert snap_after is not None
        assert snap_after.log_odds == snap_before.log_odds
        assert snap_after.state == snap_before.state
        assert snap_after.update_count == snap_before.update_count
    # Queries work identically after recovery.
    region = restarted.query_region((0, 0, 0), 4)
    assert set(region.cells) == {(1, 2, 3), (-1, 0, 4)}


def test_recovery_is_consistent_with_last_commit_not_in_flight(tmp_path):
    db = str(tmp_path / "state.bin")
    service = _service(db)
    service.submit([Observation((0, 0, 0), 0.9, 1)])
    # Strong free evidence in the second commit flips the cell to FREE.
    for ts in range(2, 15):
        service.submit([Observation((0, 0, 0), 0.02, ts)])
    service.submit([Observation((1, 0, 0), 0.9, 100)])
    v2 = service.snapshot_version()
    del service

    restarted = _service(db)
    assert restarted.snapshot_version() == v2
    # Both changes from the committed history are present together.
    assert restarted.get_cell((1, 0, 0)).state is CellState.OCCUPIED
    assert restarted.get_cell((0, 0, 0)).state is CellState.FREE


def test_atomic_write_leaves_no_tmp_after_clean_commit(tmp_path):
    db = str(tmp_path / "state.bin")
    service = _service(db)
    service.submit([Observation((0, 0, 0), 0.9, 1)])
    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert leftovers == []


def test_crash_during_write_keeps_previous_good_snapshot(tmp_path, monkeypatch):
    db = str(tmp_path / "state.bin")
    service = _service(db)
    service.submit([Observation((0, 0, 0), 0.9, 1)])
    good_bytes = open(db, "rb").read()

    # Simulate a crash after the temp file was fully written but before the
    # atomic rename: a temp file is present, the committed file is untouched.
    store = PersistenceStore(PersistenceConfig(path=db))
    cells = {(9, 9, 9): (7.0, 2, 1)}
    raw = store._encode(2, cells)
    tmp = db + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(raw)
    # Recovery must ignore the orphan temp and load the previous snapshot.
    restarted = _service(db)
    assert restarted.snapshot_version() == 1
    assert restarted.get_cell((0, 0, 0)).state is CellState.OCCUPIED
    assert restarted.get_cell((9, 9, 9)) is None
    assert open(db, "rb").read() == good_bytes


def test_torn_final_file_detected_not_silently_loaded(tmp_path):
    db = str(tmp_path / "state.bin")
    service = _service(db)
    service.submit([Observation((0, 0, 0), 0.9, 1)])
    with open(db, "r+b") as fh:
        fh.seek(-5, os.SEEK_END)
        fh.write(b"\x00\x00\x00\x00\x00")
    with pytest.raises(CorruptStateError):
        _service(db)


def test_persistence_roundtrip_large_coordinates(tmp_path):
    db = str(tmp_path / "state.bin")
    service = _service(db)
    far = (1_000_000, -2_000_000, 42)
    service.submit([Observation(far, 0.93, 1)])
    del service
    restarted = _service(db)
    assert restarted.get_cell(far).state is CellState.OCCUPIED


def test_no_persistence_means_fresh_state(tmp_path):
    service = OccupancyService(GridConfig())
    service.submit([Observation((0, 0, 0), 0.9, 1)])
    assert service.recover() == 0
