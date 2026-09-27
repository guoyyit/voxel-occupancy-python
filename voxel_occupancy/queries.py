"""Spatial queries: bounded region scan and 3D Bresenham ray traversal.

All queries operate on an immutable *snapshot* of the grid captured under the
grid lock, which guarantees they reflect exactly one committed state and can
never observe a partially applied commit.
"""
from __future__ import annotations

from collections.abc import Iterable

from .fusion import log_odds_to_probability

from .types import (
    Cell,
    CellState,
    GridConfig,
    RayResult,
    VoxelSnapshot,
)

# Snapshot entry: (log_odds, last_update_ts, update_count).
Snapshot = dict[Cell, tuple[float, int, int]]


def region_cells(center: Cell, max_range: float) -> Iterable[Cell]:
    """Enumerate integer cells within a Chebyshev (cube) max range.

    Chebyshev distance matches the "maximum per-axis range" semantics of a 3D
    range sensor; ``max_range`` is inclusive.
    """
    if max_range < 0:
        return ()
    r = int(max_range)
    cx, cy, cz = center
    return (
        (cx + dx, cy + dy, cz + dz)
        for dx in range(-r, r + 1)
        for dy in range(-r, r + 1)
        for dz in range(-r, r + 1)
    )


def bresenham_line(start: Cell, end: Cell) -> list[Cell]:
    """Integer 3D Bresenham line from ``start`` to ``end`` inclusive.

    Symmetric variant: every voxel the segment passes through is returned, in
    traversal order, with no duplicates.
    """
    x0, y0, z0 = start
    x1, y1, z1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    sz = 1 if z0 < z1 else -1

    # Driving axis is the one with the largest delta.
    dm = max(dx, dy, dz)
    cells: list[Cell] = []
    x, y, z = x0, y0, z0
    if dm == 0:
        return [start]
    # Precompute per-axis error driving.
    err_x = dm // 2
    err_y = dm // 2
    err_z = dm // 2
    for _ in range(dm + 1):
        cells.append((x, y, z))
        err_x -= dx
        err_y -= dy
        err_z -= dz
        if err_x < 0:
            x += sx
            err_x += dm
        if err_y < 0:
            y += sy
            err_y += dm
        if err_z < 0:
            z += sz
            err_z += dm
    return cells


class QueryEngine:
    """Read-only spatial queries against a consistent grid snapshot."""

    def __init__(self, config: GridConfig) -> None:
        self.config = config

    def classify(self, log_odds: float) -> CellState:
        p = log_odds_to_probability(log_odds)
        if p >= self.config.occupied_threshold:
            return CellState.OCCUPIED
        if p <= self.config.free_threshold:
            return CellState.FREE
        return CellState.UNKNOWN

    def _to_snapshot(self, cell: Cell, entry: tuple[float, int, int]) -> VoxelSnapshot:
        log_odds, last_update, count = entry
        prob = log_odds_to_probability(log_odds)
        return VoxelSnapshot(
            cell=cell,
            log_odds=log_odds,
            probability=prob,
            state=self.classify(log_odds),
            last_update=last_update,
            update_count=count,
        )

    def region_query(self, snapshot: Snapshot, center: Cell, max_range: float):
        """Return observed cells within ``max_range`` of ``center``.

        The grid is sparse, so this iterates the *stored* cells and keeps those
        inside the axis-aligned Chebyshev bounding box - O(known cells) -
        rather than scanning every voxel in the cube, which would be
        O(max_range**3). Only cells present in the sparse grid are returned;
        unobserved cells consume no storage and are simply absent.
        """
        r = int(max_range)
        cx, cy, cz = center
        out: dict[Cell, VoxelSnapshot] = {}
        for cell, entry in snapshot.items():
            x, y, z = cell
            if abs(x - cx) <= r and abs(y - cy) <= r and abs(z - cz) <= r:
                out[cell] = self._to_snapshot(cell, entry)
        return out

    def ray_query(self, snapshot: Snapshot, start: Cell, end: Cell) -> RayResult:
        """Straight-line reachability between two voxel coordinates.

        The destination is reachable iff no voxel *before or at the
        destination* along the line is classified occupied. Observation-free
        cells are treated as non-blocking (they carry no evidence of
        occupancy).
        """
        traced = bresenham_line(start, end)
        for cell in traced:
            entry = snapshot.get(cell)
            if entry is not None and self.classify(entry[0]) is CellState.OCCUPIED:
                return RayResult(
                    reachable=False, blocked_by=cell, traced_cells=tuple(traced)
                )
        return RayResult(reachable=True, blocked_by=None, traced_cells=tuple(traced))

