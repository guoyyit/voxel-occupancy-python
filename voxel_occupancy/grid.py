"""Thread-safe in-memory voxel grid with a sparse, bounded representation.

Storage model
-------------
Only *observed* cells are kept in a dict. Each entry stores::

    (log_odds, last_update_timestamp, update_count)

Unobserved cells consume no memory at all and are indistinguishable from the
flat prior. A cell whose fused log-odds returns within ``prune_log_odds_eps``
of the prior is removed again, which bounds memory even in scenarios where
observations are later retracted (a cleared cell may be re-occupied by later
observations since reads of missing cells report the prior).

Concurrency
-----------
A single reentrant lock serializes writers. A multi-cell commit is prepared,
merged and handed to persistence *while the lock is held*, so concurrent
updates are linearizable in some serial order and readers never observe a
partial commit. Readers capture the live mapping under the same lock, giving
them one consistent committed version for the whole query.
"""
from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from collections.abc import Iterator

from .types import Cell, GridConfig

# Entry: (log_odds, last_update_ts, observation_count).
CellEntry = tuple[float, int, int]
# Prepared per-commit update: (log_odds_delta, timestamp, n_observations).
CellUpdate = tuple[float, int, int]


class VoxelGrid:
    """Sparse 3D occupancy grid guarded by a reentrant lock."""

    def __init__(self, config: GridConfig) -> None:
        self.config = config
        self._lock = RLock()
        self._cells: dict[Cell, CellEntry] = {}
        self._sequence = 0

    @property
    def sequence(self) -> int:
        """Monotonic number of committed updates (the snapshot version)."""
        return self._sequence

    @contextmanager
    def exclusive(self) -> Iterator[tuple[int, dict[Cell, CellEntry]]]:
        """Acquire the grid lock; yield (sequence, live cell mapping).

        Callers may inspect and mutate via ``merge`` while holding it. The
        mapping must not be structurally modified by the caller.
        """
        with self._lock:
            yield self._sequence, self._cells

    def get_entry(self, cell: Cell) -> CellEntry | None:
        with self._lock:
            return self._cells.get(cell)

    def known_cells(self) -> int:
        with self._lock:
            return len(self._cells)

    def merge(
        self, updates: dict[Cell, CellUpdate]
    ) -> tuple[int, int]:
        """Merge prepared per-cell updates atomically.

        Must be called while ``exclusive()`` is held by the same thread.
        Returns (number_of_observations_applied, number_of_cells_touched).
        """
        cells = self._cells
        cfg = self.config
        applied_obs = 0
        touched = 0
        for cell, (delta, ts, n_obs) in updates.items():
            entry = cells.get(cell)
            if entry is None:
                log_odds = cfg.prior_log_odds + delta
                last_ts = ts
                count = n_obs
            else:
                old_lo, old_ts, old_count = entry
                log_odds = old_lo + delta
                last_ts = max(old_ts, ts)
                count = old_count + n_obs
            # Bounded log-odds -> probability always stays inside (0, 1).
            if log_odds > cfg.log_odds_max:
                log_odds = cfg.log_odds_max
            elif log_odds < cfg.log_odds_min:
                log_odds = cfg.log_odds_min
            # Memory bound: belief indistinguishable from prior -> drop cell.
            if abs(log_odds - cfg.prior_log_odds) <= cfg.prune_log_odds_eps:
                cells.pop(cell, None)
            else:
                cells[cell] = (log_odds, last_ts, count)
            applied_obs += n_obs
            touched += 1
        self._sequence += 1
        return applied_obs, touched

    def load_all(
        self, sequence: int, cells: dict[Cell, CellEntry]
    ) -> None:
        """Replace grid contents with recovered durable state (startup only)."""
        with self._lock:
            self._cells = dict(cells)
            self._sequence = sequence

    def export_all(self) -> tuple[int, dict[Cell, CellEntry]]:
        """Return a detached copy of (sequence, cells) for serialization."""
        with self._lock:
            return self._sequence, dict(self._cells)
