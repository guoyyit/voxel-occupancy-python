"""Local feeds (memory pipe / loopback) and the HTTP API surface."""
import threading

from fastapi.testclient import TestClient

from voxel_occupancy.api import create_app
from voxel_occupancy.feeds import LoopbackFeed, MemoryFeed
from voxel_occupancy.service import OccupancyService
from voxel_occupancy.types import (
    GridConfig,
    Observation,
    PersistenceConfig,
)


def test_memory_feed_cross_thread_pipeline():
    feed = MemoryFeed()
    service = OccupancyService(GridConfig())

    def producer():
        feed.push([Observation((0, 0, 0), 0.9, 1)])
        feed.push([Observation((1, 0, 0), 0.1, 2)])
        feed.close()

    threading.Thread(target=producer).start()
    results = service.run_feed(feed)
    assert [r.sequence for r in results] == [1, 2]
    assert service.get_cell((0, 0, 0)) is not None


def test_loopback_feed_is_deterministic():
    script = [[Observation((0, 0, 0), 0.9, 1)], [Observation((1, 1, 1), 0.1, 2)]]
    feed = LoopbackFeed(script)
    s1 = OccupancyService(GridConfig())
    s1.run_feed(feed)
    # Replay from the beginning.
    feed2 = LoopbackFeed(script)
    s2 = OccupancyService(GridConfig())
    s2.run_feed(feed2)
    assert s1.get_cell((0, 0, 0)).log_odds == s2.get_cell((0, 0, 0)).log_odds
    assert s1.get_cell((1, 1, 1)).log_odds == s2.get_cell((1, 1, 1)).log_odds


def test_api_submit_query_region_ray():
    service = OccupancyService(GridConfig())
    client = TestClient(create_app(service))
    assert client.get("/health").json()["version"] == 0

    resp = client.post(
        "/observations",
        json={
            "observations": [
                {"x": 2, "y": 0, "z": 0, "prob_hit": 0.95, "timestamp": 1}
            ]
        },
    )
    assert resp.json()["sequence"] == 1

    cell = client.get("/cells/2/0/0").json()
    assert cell["state"] == "occupied"

    unknown = client.get("/cells/8/8/8").json()
    assert unknown["state"] == "unknown" and unknown["probability"] == 0.5

    region = client.get("/region", params={"x": 0, "y": 0, "z": 0, "range": 2})
    assert [2, 0, 0] in region.json()["cells"]

    blocked = client.get(
        "/ray",
        params={"x0": 0, "y0": 0, "z0": 0, "x1": 5, "y1": 0, "z1": 0},
    ).json()
    assert blocked["reachable"] is False
    assert blocked["blocked_by"] == [2, 0, 0]

    clear = client.get(
        "/ray",
        params={"x0": 3, "y0": 0, "z0": 0, "x1": 5, "y1": 0, "z1": 0},
    ).json()
    assert clear["reachable"] is True


def test_api_persistent_service_recovers(tmp_path):
    db = str(tmp_path / "api-state.bin")

    def build():
        return OccupancyService.start(
            grid_config=GridConfig(),
            persistence_config=PersistenceConfig(path=db),
        )

    client = TestClient(create_app(build()))
    client.post(
        "/observations",
        json={"observations": [
            {"x": 0, "y": 0, "z": 0, "prob_hit": 0.9, "timestamp": 1}
        ]},
    )
    # A brand-new app over the same file sees recovered state.
    client2 = TestClient(create_app(build()))
    assert client2.get("/cells/0/0/0").json()["state"] == "occupied"
    assert client2.get("/health").json()["version"] == 1
