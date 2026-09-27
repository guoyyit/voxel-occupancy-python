"""Fusion: bounded probabilities, repeated fusion, stale handling policies."""
import math

import pytest

from voxel_occupancy.fusion import (
    LogOddsFusion,
    log_odds_to_probability,
    probability_to_log_odds,
)
from voxel_occupancy.types import ExpiryPolicy, GridConfig, Observation


def test_logit_roundtrip():
    for p in [0.01, 0.1, 0.3, 0.5, 0.72, 0.999]:
        assert log_odds_to_probability(probability_to_log_odds(p)) == pytest.approx(p)


def test_probability_stays_bounded_under_repeated_evidence():
    fusion = LogOddsFusion(GridConfig())
    lo = 0.0
    obs = Observation((0, 0, 0), 0.999, timestamp=1)
    # Thousands of reinforcing observations must not leave (0, 1).
    for _ in range(10_000):
        d = fusion.evaluate(obs, committed_timestamp=None, batch_newest_timestamp=1)
        assert d is not None
        lo = fusion.fuse(lo, d)
    p = log_odds_to_probability(lo)
    assert 0.0 < p < 1.0
    assert lo == pytest.approx(8.0)


def test_contradictory_evidence_returns_toward_prior():
    fusion = LogOddsFusion(GridConfig())
    lo = 0.0
    for ts in range(1, 30):
        d = fusion.evaluate(
            Observation((1, 1, 1), 0.95, ts), None, ts
        )
        lo = fusion.fuse(lo, d)
    assert log_odds_to_probability(lo) > 0.9
    for ts in range(30, 60):
        d = fusion.evaluate(
            Observation((1, 1, 1), 0.05, ts), None, ts
        )
        lo = fusion.fuse(lo, d)
    assert log_odds_to_probability(lo) < 0.1


def test_reject_drops_duplicate_and_older_observations():
    fusion = LogOddsFusion(GridConfig(expiry_policy=ExpiryPolicy.REJECT))
    obs = Observation((0, 0, 0), 0.9, timestamp=5)
    # Equal timestamp is a duplicate -> dropped.
    assert fusion.evaluate(obs, committed_timestamp=5, batch_newest_timestamp=5) is None
    # Older -> dropped.
    assert fusion.evaluate(obs, committed_timestamp=6, batch_newest_timestamp=5) is None
    # Strictly newer -> applied.
    assert fusion.evaluate(
        Observation((0, 0, 0), 0.9, 7), 6, 7
    ) is not None


def test_reject_drops_out_of_order_within_batch():
    fusion = LogOddsFusion(GridConfig(expiry_policy=ExpiryPolicy.REJECT))
    # committed None; batch carries ts 10 and an older ts 8
    assert fusion.evaluate(
        Observation((0, 0, 0), 0.9, 8), None, batch_newest_timestamp=10
    ) is None
    assert fusion.evaluate(
        Observation((0, 0, 0), 0.9, 10), None, batch_newest_timestamp=10
    ) is not None


def test_downweight_applies_fractional_weight():
    fusion = LogOddsFusion(
        GridConfig(expiry_policy=ExpiryPolicy.DOWNWEIGHT, downweight_decay=0.5)
    )
    fresh = fusion.evaluate(Observation((0, 0, 0), 0.9, 10), 9, 10)
    aged = fusion.evaluate(Observation((0, 0, 0), 0.9, 8), 10, 8)
    assert fresh is not None and aged is not None
    # committed mark is 10 in the second call: age 2 -> 0.5**2
    assert aged == pytest.approx(fresh * 0.25)


def test_downweight_horizon_drops_very_old():
    fusion = LogOddsFusion(
        GridConfig(
            expiry_policy=ExpiryPolicy.DOWNWEIGHT,
            downweight_decay=0.5,
            expiry_horizon=3,
        )
    )
    assert fusion.evaluate(
        Observation((0, 0, 0), 0.9, 5), committed_timestamp=10, batch_newest_timestamp=5
    ) is None
    assert fusion.evaluate(
        Observation((0, 0, 0), 0.9, 8), committed_timestamp=10, batch_newest_timestamp=8
    ) is not None


def test_ignore_time_always_fuses():
    fusion = LogOddsFusion(GridConfig(expiry_policy=ExpiryPolicy.IGNORE_TIME))
    for ts in (0, 1, 100):
        d = fusion.evaluate(
            Observation((0, 0, 0), 0.9, ts), committed_timestamp=1000,
            batch_newest_timestamp=ts,
        )
        assert d is not None and d > 0


def test_reproducibility():
    # Replay a fixed out-of-order stream against a committed mark of 5.
    def run():
        fusion = LogOddsFusion(
            GridConfig(expiry_policy=ExpiryPolicy.DOWNWEIGHT, downweight_decay=0.7)
        )
        lo = 0.0
        for ts, p in [(1, 0.9), (5, 0.2), (3, 0.8), (8, 0.1)]:
            d = fusion.evaluate(
                Observation((0, 0, 0), p, ts), committed_timestamp=5,
                batch_newest_timestamp=8,
            )
            if d is not None:
                lo = fusion.fuse(lo, d)
        return lo

    assert run() == run()
