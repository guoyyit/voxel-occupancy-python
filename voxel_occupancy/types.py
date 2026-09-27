"""Core data types for the voxel occupancy service."""
from __future__ import annotations

from dataclasses import dataclass, field

from enum import Enum
from typing import TypeAlias

# Integer voxel coordinate (x, y, z).
Cell: TypeAlias = tuple[int, int, int]

# Probability bounds.
PROB_MIN = 0.0
PROB_MAX = 1.0
PROB_UNKNOWN = 0.5


class CellState(str, Enum):
    """Discrete occupancy classification derived from probability."""

    OCCUPIED = "occupied"
    FREE = "free"
    UNKNOWN = "unknown"


class ExpiryPolicy(str, Enum):
    """How stale observations are handled."""

    REJECT = "reject"   # drop observations older than the latest applied timestamp
    DOWNWEIGHT = "downweight"  # apply them with a reduced, age-scaled weight
    IGNORE_TIME = "ignore_time"  # fuse regardless of timestamp (no expiry)


@dataclass(frozen=True, slots=True)
class Observation:
    """A single voxel observation.

    prob_hit: probability that the cell is occupied, in (0, 1).
    timestamp: monotonic observation time; higher means newer.
    weight: optional confidence multiplier (defaults to 1.0).
    """

    cell: Cell
    prob_hit: float
    timestamp: int
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class VoxelSnapshot:
    """Immutable view of one voxel's state."""

    cell: Cell
    log_odds: float
    probability: float
    state: CellState
    last_update: int
    update_count: int


@dataclass(slots=True)
class GridConfig:
    """Configuration of the occupancy grid fusion behaviour."""

    # Log-odds clipping bounds keep probabilities bounded away from 0/1.
    log_odds_min: float = -8.0
    log_odds_max: float = 8.0
    # Probability thresholds for the discrete state classification.
    occupied_threshold: float = 0.7
    free_threshold: float = 0.3
    # Prior log-odds for every cell (logit(0.5) = 0).
    prior_log_odds: float = 0.0
    # Cells whose probability returns close to prior are pruned to bound memory.
    prune_log_odds_eps: float = 1e-3
    # Maximum number of voxel updates applied atomically in one commit.
    max_observations_per_commit: int = 1_000_000
    # Stale observation strategy.
    expiry_policy: ExpiryPolicy = ExpiryPolicy.REJECT
    # Under DOWNWEIGHT, weight is multiplied by decay_base ** age_steps (>=0 age
    # is ignored because older timestamps are rejected under REJECT).
    downweight_decay: float = 0.5
    # An observation older than newest - expiry_horizon is fully dropped even
    # under DOWNWEIGHT. None means horizon disabled.
    expiry_horizon: int | None = None


@dataclass(slots=True)
class PersistenceConfig:
    """Configuration of crash-safe persistence/recovery."""

    path: str = "voxel_state.db"
    # fsync each commit before acknowledging it (durability).
    fsync: bool = True
    # If the on-disk state is from an incompatible version, fail loudly.
    expected_version: int = 1


@dataclass(slots=True)
class QueryResult:
    """Result of a region query."""

    cells: dict[Cell, VoxelSnapshot] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RayResult:
    """Result of a straight-line reachability query."""

    reachable: bool
    # First blocking voxel, if any.
    blocked_by: Cell | None
    # Every voxel visited along the line (in traversal order).
    traced_cells: tuple[Cell, ...]


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Outcome of submitting one observation batch."""

    sequence: int
    applied: int
    rejected_expired: int
