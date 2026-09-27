"""并发更新/查询：串行化等价性、概率有界、无丢失更新、幂等竞态。"""

from __future__ import annotations

import threading

import pytest

from voxel_occupancy.config import GridConfig, ServiceConfig
from voxel_occupancy.grid import VoxelGrid
from voxel_occupancy.service import OccupancyService
from voxel_occupancy.types import (
    Observation,
    ObservationBatch,
    ObservationKind,
)


def test_concurrent_increments_no_lost_update_and_bounded():
    """N 个线程对同一批体素累加小增量（不触发夹逼），

    最终值必须恰好等于全部增量之和（可交换，故与串行顺序无关）。
    """
    cfg = GridConfig(
        occupied_log_odds=0.01,
        free_log_odds=-0.01,
        clamp_min=-100.0,
        clamp_max=100.0,
    )
    g = VoxelGrid(cfg)
    n_threads = 8
    per_thread = 200
    coords = [(x, 0, 0) for x in range(10)]

    def worker(tid):
        for i in range(per_thread):
            obs = tuple(
                Observation(c, ObservationKind.OCCUPIED, weight=1.0) for c in coords
            )
            g.commit(ObservationBatch(obs, sequence=tid * per_thread + i + 1))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = n_threads * per_thread * 0.01
    for c in coords:
        info = g.get(c)
        assert info is not None
        assert info.log_odds == pytest.approx(total, abs=1e-9)
        assert cfg.clamp_min <= info.log_odds <= cfg.clamp_max
    assert g.stats().committed_batches == n_threads * per_thread


def test_concurrent_mixed_updates_stay_bounded():
    """高强度混合 OCCUPIED/FREE 并发：概率永不越界。"""
    cfg = GridConfig(occupied_log_odds=0.85, free_log_odds=-0.7,
                     clamp_min=-4.0, clamp_max=4.0)
    g = VoxelGrid(cfg)
    n_threads = 8
    per_thread = 300
    errors = []

    def worker(tid):
        for i in range(per_thread):
            kind = ObservationKind.OCCUPIED if (tid + i) % 2 else ObservationKind.FREE
            coord = (i % 20, tid % 4, 0)
            try:
                g.commit(
                    ObservationBatch((Observation(coord, kind),),
                                     sequence=tid * per_thread + i + 1)
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []

    for state in g.snapshot_cells().values():
        assert cfg.clamp_min <= state.log_odds <= cfg.clamp_max
        p = state.probability
        assert 0.0 < p < 1.0


def test_concurrent_readers_never_see_partial_or_crash():
    cfg = GridConfig(occupied_log_odds=0.2, free_log_odds=-0.2)
    g = VoxelGrid(cfg)
    coords = [(x, y, z) for x in range(8) for y in range(8) for z in range(4)]
    stop = threading.Event()
    problems = []

    def reader():
        while not stop.is_set():
            try:
                with g.read() as view:
                    r = view.trace_line((0, 0, 0), (7, 7, 3))
                    assert len(r.path) > 0
                    view.query_region((3, 3, 2), 6.0)
                g.stats()
            except Exception as exc:  # noqa: BLE001
                problems.append(exc)
                return

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for t in readers:
        t.start()

    def writer(tid):
        for seq in range(100):
            obs = tuple(
                Observation(
                    c,
                    ObservationKind.OCCUPIED if seq % 2 else ObservationKind.FREE,
                )
                for c in coords
            )
            g.commit(ObservationBatch(obs, sequence=tid * 100 + seq + 1))

    writers = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join()
    assert problems == []


def test_concurrent_idempotent_submit_applies_once():
    svc = OccupancyService(
        ServiceConfig(
            grid=GridConfig(occupied_log_odds=0.01),
        )
    )
    barrier = threading.Barrier(8)
    results = []
    lock = threading.Lock()

    def fire():
        barrier.wait()
        b = ObservationBatch(
            (Observation((0, 0, 0), ObservationKind.OCCUPIED),),
            sequence=1,
            idempotency_key="same-key",
        )
        r = svc.submit(b)
        with lock:
            results.append(r)

    threads = [threading.Thread(target=fire) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    committed = [r for r in results if r.committed and not r.dedup]
    dedup = [r for r in results if r.dedup]
    assert len(committed) == 1
    assert len(dedup) == 7
    assert svc.stats().committed_batches == 1
    assert svc.get((0, 0, 0)).log_odds == pytest.approx(0.01)


def test_concurrent_result_equivalent_to_some_serial_order():
    """结果落在合法串行化结果集合内：

    使用两个体素与方向相反的批次，枚举所有串行交错的最终值，
    并发结果必须是其中之一（夹逼使不同串行顺序可能产生不同结果）。
    """
    cfg = GridConfig(occupied_log_odds=3.0, free_log_odds=-3.0,
                     clamp_min=-3.5, clamp_max=3.5)

    def serial_values(order):
        g = VoxelGrid(cfg)
        for seq, kind in enumerate(order, start=1):
            g.commit(
                ObservationBatch(
                    (Observation((0, 0, 0), kind),), sequence=seq
                )
            )
        info = g.get((0, 0, 0))
        # log_odds 恰为 0 时被剪枝为未观测，等价于 0.0
        return info.log_odds if info is not None else 0.0

    # 线程 A 连续 2 次 OCCUPIED，线程 B 连续 2 次 FREE
    possible = set()
    import itertools

    for mask in itertools.product(range(2), repeat=4):
        # mask 中恰好两个 A 两个 B 才是合法交错
        if sum(mask) != 2:
            continue
        kinds = [ObservationKind.OCCUPIED if m == 0 else ObservationKind.FREE
                 for m in mask]
        possible.add(round(serial_values(kinds), 12))

    g = VoxelGrid(cfg)

    def worker(kind):
        for _ in range(2):
            g.commit(ObservationBatch((Observation((0, 0, 0), kind),),))

    t1 = threading.Thread(target=worker, args=(ObservationKind.OCCUPIED,))
    t2 = threading.Thread(target=worker, args=(ObservationKind.FREE,))
    t1.start(); t2.start(); t1.join(); t2.join()
    info = g.get((0, 0, 0))
    actual = info.log_odds if info is not None else 0.0
    assert round(actual, 12) in possible
