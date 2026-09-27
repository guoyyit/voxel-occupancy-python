"""Sparse grid memory bounds, clamping, pruning and spatial queries."""
from voxel_occupancy.grid import VoxelGrid
from voxel_occupancy.queries import QueryEngine, bresenham_line, region_cells
from voxel_occupancy.types import CellState, GridConfig


def test_unobserved_cells_use_no_storage():
    grid = VoxelGrid(GridConfig())
    with grid.exclusive():
        grid.merge({(1, 2, 3): (1.0, 1, 1)})
    assert grid.known_cells() == 1
    assert grid.get_entry((10_000, -10_000, 0)) is None


def test_log_odds_clamped():
    grid = VoxelGrid(GridConfig(log_odds_min=-4.0, log_odds_max=4.0))
    with grid.exclusive():
        grid.merge({(0, 0, 0): (100.0, 1, 1)})
        grid.merge({(0, 0, 0): (-100.0, 2, 1)})
    lo, _, count = grid.get_entry((0, 0, 0))
    assert lo == -4.0
    assert count == 2


def test_pruning_bounds_memory_and_allows_reoccupation():
    grid = VoxelGrid(GridConfig(prior_log_odds=0.0, prune_log_odds_eps=1e-3))
    with grid.exclusive():
        # +2 then -2 returns exactly to prior -> cell is removed.
        grid.merge({(0, 0, 0): (2.0, 1, 1)})
        grid.merge({(0, 0, 0): (-2.0, 2, 1)})
    assert grid.get_entry((0, 0, 0)) is None
    assert grid.known_cells() == 0
    # A cleared cell can be re-occupied by later evidence.
    with grid.exclusive():
        grid.merge({(0, 0, 0): (5.0, 3, 1)})
    assert grid.get_entry((0, 0, 0))[0] == 5.0


def test_large_sparse_grid_memory_is_proportional_to_observed_cells():
    grid = VoxelGrid(GridConfig())
    with grid.exclusive():
        grid.merge({(x, x, x): (1.0, 1, 1) for x in range(10_000)})
    assert grid.known_cells() == 10_000
    # Querying deep in empty space returns nothing.
    engine = QueryEngine(GridConfig())
    with grid.exclusive() as (_s, cells):
        result = engine.region_query(cells, (9_000_000, 0, 0), 5.0)
    assert result == {}


def test_region_cells_inclusive_cube():
    cells = list(region_cells((0, 0, 0), 1))
    assert len(cells) == 27
    assert (0, 0, 0) in cells and (1, -1, 1) in cells
    assert list(region_cells((0, 0, 0), 0)) == [(0, 0, 0)]


def test_bresenham_endpoints_and_order():
    line = bresenham_line((0, 0, 0), (5, 0, 0))
    assert line == [(i, 0, 0) for i in range(6)]
    diag = bresenham_line((0, 0, 0), (3, 3, 3))
    assert diag[0] == (0, 0, 0) and diag[-1] == (3, 3, 3)
    assert len(diag) == len(set(diag)) == 4
    assert bresenham_line((2, 2, 2), (2, 2, 2)) == [(2, 2, 2)]


def test_ray_blocked_by_occupied_only():
    engine = QueryEngine(GridConfig())
    snapshot = {
        (2, 0, 0): (8.0, 1, 1),   # occupied
        (1, 0, 0): (-8.0, 1, 1),  # free
    }
    blocked = engine.ray_query(snapshot, (0, 0, 0), (5, 0, 0))
    assert blocked.reachable is False
    assert blocked.blocked_by == (2, 0, 0)
    clear = engine.ray_query(snapshot, (3, 0, 0), (5, 0, 0))
    assert clear.reachable is True
    # Unobserved cells never block.
    empty = engine.ray_query({}, (0, 0, 0), (9, 9, 9))
    assert empty.reachable is True


def test_snapshot_classification_thresholds():
    engine = QueryEngine(
        GridConfig(occupied_threshold=0.7, free_threshold=0.3)
    )
    assert engine.classify(8.0) is CellState.OCCUPIED
    assert engine.classify(-8.0) is CellState.FREE
    assert engine.classify(0.0) is CellState.UNKNOWN
