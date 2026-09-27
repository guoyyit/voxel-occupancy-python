"""重复提交、乱序/过期观测的丢弃与降权、可复现性。"""

from __future__ import annotations

import pytest

from voxel_occupancy.config import (
    ServiceConfig,
    StalenessConfig,
    StalenessMode,
)
from voxel_occupancy.service import OccupancyService
from voxel_occupancy.staleness import decide_staleness
from voxel_occupancy.types import (
    Observation,
    ObservationBatch,
    ObservationKind,
)


def batch(coord=(0, 0, 0), seq=1, ts=0.0, key=None, n=5):
    return ObservationBatch(
        tuple(Observation(coord, ObservationKind.OCCUPIED, ts) for _ in range(n)),
        sequence=seq,
        timestamp=ts,
        idempotency_key=key,
    )


def test_idempotent_duplicate_batch():
    svc = OccupancyService(ServiceConfig())
    b = batch(seq=1, key="only-once")
    r1 = svc.submit(b)
    r2 = svc.submit(b)
    r3 = svc.submit(b)
    assert r1.committed and not r1.dedup
    assert r2.dedup and r3.dedup
    assert not r2.committed
    assert svc.stats().committed_batches == 1


def test_duplicate_without_key_is_fused():
    svc = OccupancyService(ServiceConfig())
    svc.submit(batch(seq=1))
    svc.submit(batch(seq=1))  # 无幂等键 => 视为真实重复观测，参与融合
    assert svc.get((0, 0, 0)).occupied
    assert svc.stats().committed_batches == 2


def test_out_of_order_within_lag_accepted():
    cfg = ServiceConfig(
        staleness=StalenessConfig(mode=StalenessMode.DROP, max_sequence_lag=5)
    )
    svc = OccupancyService(cfg)
    svc.submit(batch(seq=10, ts=10.0))
    # 乱序但在容忍窗口内
    late = batch(coord=(1, 0, 0), seq=8, ts=9.0)
    r = svc.submit(late)
    assert r.committed
    assert svc.get((1, 0, 0)) is not None


def test_stale_sequence_dropped():
    cfg = ServiceConfig(
        staleness=StalenessConfig(mode=StalenessMode.DROP, max_sequence_lag=2)
    )
    svc = OccupancyService(cfg)
    svc.submit(batch(seq=10, ts=10.0))
    stale = batch(coord=(1, 0, 0), seq=1, ts=1.0)
    r = svc.submit(stale)
    assert not r.committed
    assert r.reason and "stale" in r.reason
    assert svc.get((1, 0, 0)) is None


def test_stale_age_dropped():
    cfg = ServiceConfig(
        staleness=StalenessConfig(
            mode=StalenessMode.DROP, max_age_seconds=5.0
        )
    )
    svc = OccupancyService(cfg)
    svc.submit(batch(seq=1, ts=100.0))
    old = batch(coord=(2, 0, 0), seq=2, ts=50.0)
    r = svc.submit(old)
    assert not r.committed
    assert svc.get((2, 0, 0)) is None


def test_accept_mode_keeps_stale():
    cfg = ServiceConfig(
        staleness=StalenessConfig(mode=StalenessMode.ACCEPT)
    )
    svc = OccupancyService(cfg)
    svc.submit(batch(seq=100))
    r = svc.submit(batch(coord=(3, 0, 0), seq=1))
    assert r.committed
    assert svc.get((3, 0, 0)) is not None


def test_downweight_weakens_stale_observation():
    cfg = ServiceConfig(
        staleness=StalenessConfig(
            mode=StalenessMode.DOWNWEIGHT,
            max_sequence_lag=1,
            downweight_min=0.1,
        )
    )
    svc = OccupancyService(cfg)
    svc.submit(batch(seq=10, n=0))  # 抬高水位
    # 极强过期（lag 远超窗口）-> 权重衰减到接近下限
    stale = ObservationBatch(
        (Observation((0, 0, 0), ObservationKind.OCCUPIED),), sequence=1
    )
    r = svc.submit(stale)
    assert r.committed and r.reason == "stale:downweight"
    # lag 远超窗口 -> 权重衰减到下限 0.1：log_odds = 0.85 * 0.1
    info = svc.get((0, 0, 0))
    assert info is not None
    assert info.log_odds < 0.85
    assert info.log_odds == pytest.approx(0.85 * 0.1, abs=1e-9)

    # 对照组：同样观测在未过期时完整生效
    fresh = OccupancyService(
        ServiceConfig(
            staleness=StalenessConfig(
                mode=StalenessMode.DOWNWEIGHT, max_sequence_lag=1
            )
        )
    )
    fresh.submit(
        ObservationBatch(
            (Observation((0, 0, 0), ObservationKind.OCCUPIED),), sequence=1
        )
    )
    assert fresh.get((0, 0, 0)).log_odds == pytest.approx(0.85)


def test_staleness_decision_is_pure():
    cfg = StalenessConfig(mode=StalenessMode.DROP, max_sequence_lag=2)
    b = batch(seq=5)
    d1 = decide_staleness(
        b, committed_sequence=10, latest_timestamp=0.0, config=cfg
    )
    d2 = decide_staleness(
        b, committed_sequence=10, latest_timestamp=0.0, config=cfg
    )
    assert d1 == d2


def test_same_input_same_policy_reproducible():
    def build():
        cfg = ServiceConfig(
            staleness=StalenessConfig(
                mode=StalenessMode.DOWNWEIGHT,
                max_sequence_lag=3,
                max_age_seconds=10.0,
                downweight_min=0.25,
            )
        )
        svc = OccupancyService(cfg)
        stream = [
            batch(coord=(x % 4, x % 2, 0), seq=20 + x, ts=100.0 + x, n=1)
            for x in range(30)
        ]
        # 故意打乱顺序提交（相同洗牌结果对两个实例一致）
        order = list(range(30))
        order.sort(key=lambda i: (i * 7) % 13)
        for i in order:
            svc.submit(stream[i])
        return svc

    a, b = build(), build()
    cells_a = a.grid.snapshot_cells()
    cells_b = b.grid.snapshot_cells()
    assert set(cells_a) == set(cells_b)
    for coord in cells_a:
        assert cells_a[coord].log_odds == cells_b[coord].log_odds
