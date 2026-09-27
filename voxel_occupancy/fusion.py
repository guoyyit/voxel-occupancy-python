"""Probability fusion primitives operating in the log-odds domain.

The occupancy grid stores each cell's belief as a log-odds value

    L = ln(p / (1 - p))

With an independent binary measurement model, fusing one observation is a
simple addition:

    L_new = L_old + logit(p_hit) - L_prior

which keeps the math numerically well behaved. Bounds ``[log_odds_min,
log_odds_max]`` guarantee the occupancy probability stays strictly inside
(0, 1) no matter how many observations are fused - repeated, re-ordered or
stale updates can never push it outside the legal interval.

Time/staleness model
--------------------
``evaluate`` is given an observation, the cell's already-committed high-water
timestamp (``None`` for an unseen cell) and the newest timestamp carried by
the current batch. It returns a log-odds contribution or ``None`` when the
observation must be dropped:

* ``REJECT``      - an observation is kept only when its timestamp is strictly
                    newer than the committed high-water mark. Equal timestamps
                    are treated as duplicates. Within a batch, older entries
                    re-ordered before the batch's newest are also dropped.
* ``DOWNWEIGHT``  - equal timestamps fuse at full confidence; older ones fuse
                    with ``downweight_decay ** age``; anything older than
                    ``expiry_horizon`` is dropped.
* ``IGNORE_TIME`` - every observation fuses at full confidence.

Given identical inputs and policy the result is bit-for-bit reproducible.
"""
from __future__ import annotations

import math

from .types import PROB_MAX, PROB_MIN, GridConfig, Observation

# Probabilities never reach exactly 0/1 before taking logit.
_EPS = 1e-6


def probability_to_log_odds(p: float) -> float:
    """Convert an occupancy probability in (0, 1) to log-odds."""
    p = min(max(p, PROB_MIN + _EPS), PROB_MAX - _EPS)
    return math.log(p / (1.0 - p))


def log_odds_to_probability(log_odds: float) -> float:
    """Convert log-odds back to an occupancy probability via the sigmoid."""
    if log_odds >= 0.0:
        e = math.exp(-log_odds)
        return 1.0 / (1.0 + e)
    e = math.exp(log_odds)
    return e / (1.0 + e)


def clamp_log_odds(log_odds: float, lo: float, hi: float) -> float:
    """Clamp a log-odds value into the configured bounded interval."""
    if log_odds < lo:
        return lo
    if log_odds > hi:
        return hi
    return log_odds


class LogOddsFusion:
    """Applies independent binary observations in the log-odds domain."""

    def __init__(self, config: GridConfig) -> None:
        self.config = config

    def is_fresh(self, obs: Observation, committed_timestamp: int | None) -> bool:
        """Whether the observation is acceptable vs the committed water mark."""
        policy = self.config.expiry_policy
        if policy.value == "ignore_time" or committed_timestamp is None:
            return True
        if policy.value == "reject":
            return obs.timestamp > committed_timestamp
        # DOWNWEIGHT: same instant is fine; only horizon-too-old is dropped.
        if obs.timestamp >= committed_timestamp:
            return True
        age = committed_timestamp - obs.timestamp
        return not (
            self.config.expiry_horizon is not None
            and age > self.config.expiry_horizon
        )

    def time_weight(
        self, obs: Observation, reference_timestamp: int
    ) -> float:
        """Confidence multiplier relative to a (newer/equal) reference time."""
        policy = self.config.expiry_policy
        if policy.value == "ignore_time" or obs.timestamp >= reference_timestamp:
            return 1.0
        age = reference_timestamp - obs.timestamp
        if policy.value == "reject":
            # The caller already screened committed staleness; this only
            # rejects re-ordered entries within a REJECT batch.
            return 0.0
        if (
            self.config.expiry_horizon is not None
            and age > self.config.expiry_horizon
        ):
            return 0.0
        return self.config.downweight_decay ** age

    def evaluate(
        self,
        obs: Observation,
        committed_timestamp: int | None,
        batch_newest_timestamp: int,
    ) -> float | None:
        """Return the log-odds contribution, or ``None`` if to be dropped."""
        if not self.is_fresh(obs, committed_timestamp):
            return None
        if committed_timestamp is not None and obs.timestamp <= committed_timestamp:
            # DOWNWEIGHT of data older than the committed mark.
            reference = committed_timestamp
        elif obs.timestamp < batch_newest_timestamp:
            # Re-ordered within the batch.
            reference = batch_newest_timestamp
        else:
            reference = obs.timestamp
        weight = self.time_weight(obs, reference)
        if weight <= 0.0:
            return None
        confidence = obs.weight * weight
        delta = probability_to_log_odds(obs.prob_hit) - self.config.prior_log_odds
        return delta * confidence

    def fuse(self, current: float, update: float) -> float:
        """Add an update to an accumulated log-odds value with clamping."""
        return clamp_log_odds(
            current + update,
            self.config.log_odds_min,
            self.config.log_odds_max,
        )
