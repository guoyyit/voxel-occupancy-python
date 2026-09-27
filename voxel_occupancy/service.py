"""Occupancy service: grid + fusion + queries + persistence + feeds.

Commit semantics
----------------
``submit(batch)`` is one atomic transaction:

  1. all observations are reduced to per-cell log-odds updates, applying the
     configured expiry policy (drop or down-weight stale observations);
  2. the grid lock is taken once and held while updates merge *and* the new
     state is durably persisted;
  3. the monotonically increasing commit sequence is advanced inside that
     critical section.

Concurrent batches therefore interleave only at whole-batch boundaries: the
result is equivalent to some serial ordering. Queries acquire the same lock
and build their answer against one mapping, so they always see an entire
committed state - never a half-applied batch.
"""
from __future__ import annotations

from .feeds import LoopbackFeed, MemoryFeed
from .fusion import LogOddsFusion
from .grid import VoxelGrid
from .persistence import PersistenceStore
from .queries import QueryEngine
from .types import (
    Cell,
    CommitResult,
    GridConfig,
    Observation,
    PersistenceConfig,
    QueryResult,
    RayResult,
    VoxelSnapshot,
)


class OccupancyService:
    """Local 3D occupancy service with concurrency, recovery and queries."""

    def __init__(
        self,
        grid_config: GridConfig | None = None,
        persistence_config: PersistenceConfig | None = None,
    ) -> None:
        self.config = grid_config or GridConfig()
        self._grid = VoxelGrid(self.config)
        self._fusion = LogOddsFusion(self.config)
        self._queries = QueryEngine(self.config)
        self._store: PersistenceStore | None = (
            PersistenceStore(persistence_config)
            if persistence_config is not None
            else None
        )

    # ----- lifecycle / recovery -----------------------------------------

    def recover(self) -> int:
        """Load the last durable snapshot into the grid.

        Returns the recovered commit sequence (0 when nothing was persisted).
        Called automatically when the service is built via
        :meth:`start`, but safe to call explicitly at startup.
        """
        if self._store is None:
            return 0
        recovered = self._store.load()
        if recovered is None:
            return 0
        sequence, cells = recovered
        self._grid.load_all(sequence, cells)
        return sequence

    @classmethod
    def start(
        cls,
        grid_config: GridConfig | None = None,
        persistence_config: PersistenceConfig | None = None,
    ) -> "OccupancyService":
        """Create a service and immediately recover durable state."""
        service = cls(grid_config, persistence_config)
        service.recover()
        return service

    def close(self) -> None:
        """Release resources (snapshots are fully replaced; nothing to flush)."""
        return None

    # ----- commits -------------------------------------------------------

    def _prepare(
        self, observations: list[Observation]
    ) -> tuple[dict, int]:
        """Reduce a raw batch to per-cell updates under the expiry policy.

        Returns (updates, rejected_count). ``updates`` maps each affected cell
        to (log_odds_delta, timestamp, n_applied_observations).

        Expiry is evaluated against both the newest timestamp already
        committed for each cell and the newest timestamp carried by this batch,
        so out-of-order and duplicate observations are handled consistently.
        """
        # Committed high-water mark per touched cell (before this batch).
        committed_ts: dict[Cell, int] = {}
        affected = {obs.cell for obs in observations}
        for cell in affected:
            entry = self._grid.get_entry(cell)
            if entry is not None:
                committed_ts[cell] = entry[1]
        # Newest timestamp carried by this batch itself.
        batch_newest: dict[Cell, int] = {}
        for obs in observations:
            prev = batch_newest.get(obs.cell)
            if prev is None or obs.timestamp > prev:
                batch_newest[obs.cell] = obs.timestamp
        updates: dict[Cell, list] = {}
        rejected = 0
        for obs in observations:
            committed = committed_ts.get(obs.cell)
            in_batch = batch_newest[obs.cell]
            delta = self._fusion.evaluate(obs, committed, in_batch)
            if delta is None:
                rejected += 1
                continue
            if delta == 0.0:
                # Observation exactly at the prior: no information.
                continue
            acc = updates.get(obs.cell)
            if acc is None:
                # acc = [sum_delta, max_ts, count]
                updates[obs.cell] = [delta, obs.timestamp, 1]
            else:
                acc[0] += delta
                acc[1] = max(acc[1], obs.timestamp)
                acc[2] += 1
        prepared = {cell: (acc[0], acc[1], acc[2]) for cell, acc in updates.items()}
        return prepared, rejected

    def submit(self, observations: list[Observation]) -> CommitResult:
        """Atomically fuse one batch and persist the committed state."""
        if len(observations) > self.config.max_observations_per_commit:
            raise ValueError(
                f"batch size {len(observations)} exceeds configured limit "
                f"{self.config.max_observations_per_commit}"
            )
        prepared, rejected = self._prepare(observations)
        with self._grid.exclusive() as (sequence, cells):
            applied, _touched = self._grid.merge(prepared)
            new_sequence = self._grid.sequence
            if self._store is not None:
                # Persist inside the same critical section: readers cannot see
                # a version that is not already durable, and a crash either
                # never happened (old file intact) or committed fully.
                self._store.save(new_sequence, cells)
        return CommitResult(
            sequence=new_sequence, applied=applied, rejected_expired=rejected
        )

    # ----- queries -------------------------------------------------------

    def _snapshot_of(
        self, entry: tuple[float, int, int], cell: Cell
    ) -> VoxelSnapshot:
        return self._queries._to_snapshot(cell, entry)  # noqa: SLF001

    def get_cell(self, cell: Cell) -> VoxelSnapshot | None:
        """Return the committed state of one voxel, or None if unobserved."""
        entry = self._grid.get_entry(cell)
        if entry is None:
            return None
        return self._snapshot_of(entry, cell)

    def query_region(self, center: Cell, max_range: float) -> QueryResult:
        """All observed cells within the inclusive max range."""
        with self._grid.exclusive() as (_seq, cells):
            result = self._queries.region_query(cells, center, max_range)
        return QueryResult(cells=result)

    def query_ray(self, start: Cell, end: Cell) -> RayResult:
        """Reachability along a straight line of voxel coordinates."""
        with self._grid.exclusive() as (_seq, cells):
            return self._queries.ray_query(cells, start, end)

    def snapshot_version(self) -> int:
        """Commit sequence of the currently visible state."""
        return self._grid.sequence

    def run_feed(self, feed: MemoryFeed | LoopbackFeed) -> list[CommitResult]:
        """Drain a local feed to completion, committing every batch."""
        return [self.submit(batch) for batch in feed.batches()]
