"""概率融合、有界性、清除/重占用、剪枝与未观测不变性。"""

from __future__ import annotations

import math

import pytest

from voxel_occupancy.config import GridConfig
from voxel_occupancy.grid import VoxelGrid
from voxel_occupancy.types import (
    Observation,
    ObservationBatch,
    ObservationKind,
)


def occ(coord, n, seq=1, ts=0.0):
    return ObservationBatch(
        tuple(Observation(coord, ObservationKind.OCCUPIED, ts) for _ in range(n)),
        sequence=seq,
        timestamp=ts,
    )


def free(coord, n, seq=1, ts=0.0):
    return ObservationBatch(
        tuple(Observation(coord, ObservationKind.FREE, ts) for _ in range(n)),
        sequence=seq,
        timestamp=ts,
    )


def test_unobserved_stays_prior_and_free():
    g = VoxelGrid()
    # 未观测区域不产生条目，不阻挡
    assert g.get((5, 5, 5)) is None
    assert g.is_occupied((5, 5, 5)) is False
    assert g.query_region((0, 0, 0), 100.0) == []
    r = g.trace_line((0, 0, 0), (10, 10, 10))
    assert r.reachable and r.blocked_coord is None
    assert g.stats().stored_cells == 0


def test_update_only_touches_relevant_voxels():
    g = VoxelGrid()
    g.commit(occ((1, 1, 1), 5, seq=1))
    before = g.get((1, 1, 1))
    g.commit(occ((2, 2, 2), 5, seq=2))
    # 第一批的体素状态不受第二批影响
    assert g.get((1, 1, 1)) == before
    assert g.get((3, 3, 3)) is None


def test_fusion_monotone_and_bounded():
    cfg = GridConfig()
    g = VoxelGrid(cfg)
    probs = []
    for seq in range(1, 201):
        g.commit(occ((0, 0, 0), 1, seq=seq))
        info = g.get((0, 0, 0))
        probs.append(info.probability)
        assert cfg.clamp_min - 1e-12 <= info.log_odds <= cfg.clamp_max + 1e-12
        assert 0.0 < info.probability < 1.0
    assert probs == sorted(probs)  # 占用观测下单调上升
    assert probs[-1] == pytest.approx(1.0 - 1.0 / (1.0 + math.exp(cfg.clamp_max)))

    # 大量 FREE 同样有下界，不会越界
    for seq in range(201, 500):
        g.commit(free((0, 0, 0), 1, seq=seq))
        info = g.get((0, 0, 0))
        assert 0.0 < info.probability < 1.0
        assert info.log_odds >= cfg.clamp_min - 1e-12


def test_repeated_observations_converge():
    g = VoxelGrid()
    g.commit(occ((0, 0, 0), 1, seq=1))
    single = g.get((0, 0, 0)).log_odds
    # 20 次相同观测等价于 1 次 20 条的批次
    g2 = VoxelGrid()
    for seq in range(1, 21):
        g2.commit(occ((0, 0, 0), 1, seq=seq))
    g3 = VoxelGrid()
    g3.commit(occ((0, 0, 0), 20, seq=1))
    assert g2.get((0, 0, 0)).log_odds == g3.get((0, 0, 0)).log_odds
    assert single == pytest.approx(0.85)
    assert g2.get((0, 0, 0)).log_odds > single


def test_clear_then_reoccupy():
    g = VoxelGrid()
    g.commit(occ((1, 2, 3), 5, seq=1))
    assert g.get((1, 2, 3)).occupied
    g.commit(
        ObservationBatch(
            (Observation((1, 2, 3), ObservationKind.CLEAR),), sequence=2
        )
    )
    assert g.get((1, 2, 3)) is None
    assert not g.is_occupied((1, 2, 3))
    # 重新占用
    g.commit(occ((1, 2, 3), 5, seq=3))
    assert g.get((1, 2, 3)).occupied
    assert g.stats().stored_cells == 1


def test_cancel_out_prunes_to_unobserved():
    cfg = GridConfig(occupied_log_odds=0.8, free_log_odds=-0.4)
    g = VoxelGrid(cfg)
    g.commit(occ((0, 0, 0), 1, seq=1))
    g.commit(free((0, 0, 0), 2, seq=2))  # 0.8 - 0.8 = 0 -> 剪枝
    assert g.get((0, 0, 0)) is None
    assert g.stats().stored_cells == 0
    # 再次占用仍然有效（可重新占用）
    g.commit(occ((0, 0, 0), 3, seq=3))
    assert g.get((0, 0, 0)).occupied


def test_intra_batch_order_accumulates():
    g = VoxelGrid()
    # 同批次内先占用再清除：最终为未观测
    g.commit(
        ObservationBatch(
            (
                Observation((0, 0, 0), ObservationKind.OCCUPIED),
                Observation((0, 0, 0), ObservationKind.OCCUPIED),
                Observation((0, 0, 0), ObservationKind.CLEAR),
            ),
            sequence=1,
        )
    )
    assert g.get((0, 0, 0)) is None


def test_empty_batch_is_still_committed():
    g = VoxelGrid()
    result = g.commit(ObservationBatch((), sequence=1))
    assert result.committed and result.touched == 0
    assert g.stats().committed_batches == 1
