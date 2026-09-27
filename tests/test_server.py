"""HTTP 端到端集成测试。"""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from voxel_occupancy.config import (  # noqa: E402
    PersistenceConfig,
    ServiceConfig,
)
from voxel_occupancy.server import create_app  # noqa: E402
from voxel_occupancy.service import OccupancyService  # noqa: E402


@pytest.fixture
def client(tmp_path):
    svc = OccupancyService(
        ServiceConfig(
            persistence=PersistenceConfig(
                path=str(tmp_path / "g.dat"), restore_on_start=False
            )
        )
    )
    return TestClient(create_app(svc))


def _occ(x, y, z, n=5):
    return {"x": x, "y": y, "z": z, "kind": "occupied"}


def test_submit_and_query_flow(client):
    r = client.post(
        "/batches",
        json={
            "sequence": 1,
            "timestamp": 10.0,
            "idempotency_key": "b1",
            "observations": [_occ(3, 0, 0)] * 5
            + [{"x": 1, "y": 0, "z": 0, "kind": "free"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["committed"] and body["touched"] == 6

    # 幂等
    r2 = client.post(
        "/batches",
        json={"sequence": 1, "idempotency_key": "b1",
              "observations": [_occ(3, 0, 0)]},
    )
    assert r2.json()["dedup"] is True

    v = client.get("/voxels/3/0/0").json()
    assert v["observed"] and v["occupied"]
    assert 0.0 < v["probability"] < 1.0

    unknown = client.get("/voxels/100/100/100").json()
    assert unknown["observed"] is False and unknown["occupied"] is False


def test_region_and_line_endpoints(client):
    client.post(
        "/batches",
        json={
            "sequence": 1,
            "observations": [_occ(x, 0, 0) for x in range(0, 6)],
        },
    )
    r = client.get("/regions", params={"x": 2, "y": 0, "z": 0, "range": 1.5})
    coords = sorted(tuple(v["coord"]) for v in r.json()["voxels"])
    assert coords == [(1, 0, 0), (2, 0, 0), (3, 0, 0)]

    r = client.get(
        "/lines",
        params={"x0": 0, "y0": 1, "z0": 0, "x1": 5, "y1": 1, "z1": 0},
    )
    assert r.json()["reachable"] is True
    r = client.get(
        "/lines", params={"x0": 0, "y0": 0, "z0": 0, "x1": 5, "y1": 0, "z1": 0}
    )
    body = r.json()
    assert body["reachable"] is False
    # 起点不计为自身阻挡点，首个占用体素为 (1,0,0)
    assert body["blocked_coord"] == [1, 0, 0]


def test_stats_and_flush(client):
    client.post(
        "/batches",
        json={"sequence": 7, "observations": [_occ(0, 0, 0)]},
    )
    s = client.get("/stats").json()
    assert s["committed_sequence"] == 7 and s["stored_cells"] == 1
    assert client.post("/flush").json()["flushed"] is True


def test_clear_via_http(client):
    client.post("/batches", json={"sequence": 1, "observations": [_occ(1, 1, 1)] * 5})
    assert client.get("/voxels/1/1/1").json()["occupied"]
    client.post(
        "/batches",
        json={
            "sequence": 2,
            "observations": [{"x": 1, "y": 1, "z": 1, "kind": "clear"}],
        },
    )
    assert client.get("/voxels/1/1/1").json()["observed"] is False
