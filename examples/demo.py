"""End-to-end local demo: loopback feed -> service -> queries -> restart.

Run:  uv run python examples/demo.py
No network, sensors or external services are used (HTTP layer excluded).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxel_occupancy.feeds import LoopbackFeed  # noqa: E402
from voxel_occupancy.service import OccupancyService
from voxel_occupancy.types import (
    ExpiryPolicy,
    GridConfig,
    Observation,
    PersistenceConfig,
)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="voxel-demo-"))
    db_path = workdir / "state.bin"

    config = GridConfig(
        expiry_policy=ExpiryPolicy.REJECT,
        log_odds_min=-8.0,
        log_odds_max=8.0,
        occupied_threshold=0.7,
        free_threshold=0.3,
    )

    # Scripted "sensor" replay: occupy (2,0,0), then later mark it free.
    script: list[list[Observation]] = [
        [Observation((2, 0, 0), 0.95, 1)],
        [Observation((2, 0, 0), 0.02, ts) for ts in range(2, 14)],
        [Observation((5, 0, 0), 0.97, 100)],
    ]

    service = OccupancyService.start(config, PersistenceConfig(path=str(db_path)))
    service.run_feed(LoopbackFeed(script))

    print("version:", service.snapshot_version())
    print("(2,0,0):", service.get_cell((2, 0, 0)).state.value)
    print("(5,0,0):", service.get_cell((5, 0, 0)).state.value)
    ray = service.query_ray((0, 0, 0), (6, 0, 0))
    print("ray reachable:", ray.reachable, "blocked_by:", ray.blocked_by)
    region = service.query_region((0, 0, 0), 5)
    print("cells within range 5:", sorted(region.cells))

    # Simulate a full process restart.
    restarted = OccupancyService.start(
        config, PersistenceConfig(path=str(db_path))
    )
    print("recovered version:", restarted.snapshot_version())
    assert restarted.get_cell((5, 0, 0)).state.value == "occupied"


if __name__ == "__main__":
    main()
