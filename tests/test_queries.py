"""区域查询、直线可达查询、Bresenham 与快照隔离。"""

from __future__ import annotations

import threading

from voxel_occupancy.config import GridConfig
from voxel_occupancy.grid import VoxelGrid, bresenham3d
from voxel_occupancy.types import Observation, ObservationBatch, ObservationKind


def occ_batch(coords, seq=1, n=5):
    return ObservationBatch(
        tuple(
            Observation(c, ObservationKind.OCCUPIED)
            for c in coords
            for _ in range(n)
        ),
        sequence=seq,
    )


def test_bresenham_endpoints_and_axis():
    assert bresenham3d((0, 0, 0), (0, 0, 0)) == [(0, 0, 0)]
    assert bresenham3d((0, 0, 0), (5, 0, 0)) == [(i, 0, 0) for i in range(6)]
    diag = bresenham3d((0, 0, 0), (4, 4, 4))
    assert diag[0] == (0, 0, 0) and diag[-1] == (4, 4, 4)
    assert len(diag) == 5
    # 负方向
    line = bresenham3d((3, 3, 3), (0, 0, 0))
    assert line[0] == (3, 3, 3) and line[-1] == (0, 0, 0)


def test_region_sphere_bounds():
    g = VoxelGrid()
    coords = [(x, 0, 0) for x in range(-5, 6)] + [(0, y, 0) for y in range(-5, 6)]
    g.commit(occ_batch(coords, seq=1))
    res = g.query_region((0, 0, 0), 3.0)
    got = {v.coord for v in res}
    expected = {c for c in set(coords) if c[0] ** 2 + c[1] ** 2 <= 9}
    assert got == expected
    assert all(v.occupied for v in g.query_region((0, 0, 0), 5.0, occupied_only=True))


def test_region_free_voxels_excluded_by_occupied_only():
    cfg = GridConfig(free_log_odds=-2.0)
    g = VoxelGrid(cfg)
    g.commit(
        ObservationBatch(
            (Observation((1, 0, 0), ObservationKind.OCCUPIED),) * 5
            + (Observation((2, 0, 0), ObservationKind.FREE),) * 5,
            sequence=1,
        )
    )
    allv = g.query_region((0, 0, 0), 3.0)
    assert {v.coord for v in allv} == {(1, 0, 0), (2, 0, 0)}
    occ = g.query_region((0, 0, 0), 3.0, occupied_only=True)
    assert {v.coord for v in occ} == {(1, 0, 0)}


def test_region_negative_radius_rejected():
    g = VoxelGrid()
    try:
        g.query_region((0, 0, 0), -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_line_trace_blocked_and_reachable():
    g = VoxelGrid()
    g.commit(occ_batch([(3, 0, 0)], seq=1))
    r = g.trace_line((0, 0, 0), (5, 0, 0))
    assert not r.reachable
    assert r.blocked_coord == (3, 0, 0)
    # 起点即使被占用也不作为自身阻挡点
    g.commit(occ_batch([(0, 0, 0)], seq=2))
    r = g.trace_line((0, 0, 0), (2, 0, 0))
    assert r.reachable
    # 平行但不同轴的直线可达
    assert g.trace_line((0, 1, 0), (5, 1, 0)).reachable
    # 终点被占用 -> 不可达
    end = g.trace_line((0, 2, 0), (3, 2, 0))
    g.commit(occ_batch([(3, 2, 0)], seq=3))
    end = g.trace_line((0, 2, 0), (3, 2, 0))
    assert not end.reachable and end.blocked_coord == (3, 2, 0)


def test_queries_never_observe_partial_update():
    """查询线程在并发提交期间看到的必须是某个已提交前缀的完整状态。

    序列：提交 A（20 点）-> 提交 B（20 点）-> 清除 B -> 清除 A，循环。
    合法快照占用集合只有四种；尤其不能出现"某批只可见一部分"的中间态。
    """
    import threading as _threading

    g = VoxelGrid()
    coords_a = [(x, 0, 0) for x in range(1, 21)]      # 避开原点，两批不相交
    coords_b = [(0, 0, z) for z in range(1, 21)]
    set_a, set_b = set(coords_a), set(coords_b)
    legal_states = {frozenset(), frozenset(set_a), frozenset(set_b),
                    frozenset(set_a | set_b)}
    stop = _threading.Event()
    violations = []

    def reader():
        checked = 0
        while not stop.is_set():
            # 整个扫描在同一个一致性读视图上进行：必须对应某个已提交快照
            with g.read() as view:
                occ = frozenset(
                    c for c in (set_a | set_b) if view.is_occupied(c)
                )
            if occ not in legal_states:
                violations.append(occ)
                return
            checked += 1
        assert checked > 0  # 读者确实运行过

    t = _threading.Thread(target=reader)
    t.start()
    for seq in range(1, 101):
        g.commit(occ_batch(coords_a, seq=seq * 4 + 1))
        g.commit(occ_batch(coords_b, seq=seq * 4 + 2))
        g.commit(
            ObservationBatch(
                tuple(Observation(c, ObservationKind.CLEAR) for c in coords_b),
                sequence=seq * 4 + 3,
            )
        )
        g.commit(
            ObservationBatch(
                tuple(Observation(c, ObservationKind.CLEAR) for c in coords_a),
                sequence=seq * 4 + 4,
            )
        )
    stop.set()
    t.join()
    assert violations == []
