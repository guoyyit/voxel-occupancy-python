"""大规模栅格下未观测区域的有界内存表示。"""

from __future__ import annotations

from voxel_occupancy.config import GridConfig
from voxel_occupancy.grid import VoxelGrid, chunk_key
from voxel_occupancy.types import (
    Observation,
    ObservationBatch,
    ObservationKind,
)


def test_sparse_unobserved_region_costs_nothing():
    g = VoxelGrid()
    # 坐标分布在极广空间（±1e6），但只有少量体素被观测
    observed = [(1_000_000, -1_000_000, 500_000), (-999_999, 3, 42), (0, 0, 0)]
    g.commit(
        ObservationBatch(
            tuple(Observation(c, ObservationKind.OCCUPIED) for c in observed),
            sequence=1,
        )
    )
    assert g.stats().stored_cells == len(observed)
    assert len(g._chunks) == 3
    # 未观测体素即便是邻居也不占存储
    assert g.get((1_000_001, -1_000_000, 500_000)) is None
    # 大范围区域查询只检查包围盒相交的分块，不扫描整个坐标空间
    res = g.query_region((0, 0, 0), 10.0)
    assert {v.coord for v in res} == {(0, 0, 0)}


def test_memory_reclaimed_after_cancel():
    cfg = GridConfig(occupied_log_odds=0.8, free_log_odds=-0.8)
    g = VoxelGrid(cfg)
    coords = [(x, y, z) for x in range(20) for y in range(20) for z in range(20)]
    g.commit(
        ObservationBatch(
            tuple(Observation(c, ObservationKind.OCCUPIED) for c in coords),
            sequence=1,
        )
    )
    assert g.stats().stored_cells == len(coords)
    # 等量反向观测把所有体素拉回先验 -> 全部剪枝，分块释放
    g.commit(
        ObservationBatch(
            tuple(Observation(c, ObservationKind.FREE) for c in coords),
            sequence=2,
        )
    )
    assert g.stats().stored_cells == 0
    assert g._chunks == {}


def test_long_running_cycling_environment_does_not_grow():
    """模拟动态环境：体素被占用 -> 清除 -> 另一位置再占用，反复循环。"""
    g = VoxelGrid()
    window = 400
    for step in range(60):
        active = {(x + step * 7, y + step * 3, 0)
                  for x in range(window) for y in range(2)}
        # 提交当前活跃占用
        g.commit(
            ObservationBatch(
                tuple(Observation(c, ObservationKind.OCCUPIED) for c in active),
                sequence=step * 2 + 1,
            )
        )
        # 下一步把它们全部清除（同时新的位置出现）
        g.commit(
            ObservationBatch(
                tuple(Observation(c, ObservationKind.CLEAR) for c in active),
                sequence=step * 2 + 2,
            )
        )
        assert g.stats().stored_cells == 0
        assert g._chunks == {}

    # 全部结束后栅格依旧为空、可重新占用
    g.commit(
        ObservationBatch(
            (Observation((123456, 654321, 7), ObservationKind.OCCUPIED),),
            sequence=1000,
        )
    )
    assert g.stats().stored_cells == 1
    assert g.get((123456, 654321, 7)).occupied


def test_chunk_distribution_is_compact():
    g = VoxelGrid()
    # 16x16x16 连续区域应恰好落在少量分块中
    coords = [(x, y, z) for x in range(16) for y in range(16) for z in range(16)]
    g.commit(
        ObservationBatch(
            tuple(Observation(c, ObservationKind.OCCUPIED) for c in coords),
            sequence=1,
        )
    )
    assert len({chunk_key(c) for c in coords}) == 1
    assert len(g._chunks) == 1
    # 跨过分块边界后新增一个分块
    g.commit(
        ObservationBatch(
            (Observation((16, 0, 0), ObservationKind.OCCUPIED),), sequence=2
        )
    )
    assert len(g._chunks) == 2


def test_negative_coordinates_share_chunks():
    g = VoxelGrid()
    coords = [(-1, -1, -1), (-16, 0, 0), (-17, 0, 0), (0, 0, 0)]
    g.commit(
        ObservationBatch(
            tuple(Observation(c, ObservationKind.OCCUPIED) for c in coords),
            sequence=1,
        )
    )
    assert g.stats().stored_cells == 4
    # (-1,..) 与 (0,..) 属于不同分块；(-16) 与 (-17) 也不同
    keys = {chunk_key(c) for c in coords}
    assert len(keys) == 4
