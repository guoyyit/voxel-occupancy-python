"""FastAPI HTTP surface for the local occupancy service.

Endpoints
---------
GET  /health                            liveness + current commit version
POST /observations                      submit one atomic observation batch
GET  /cells/{x}/{y}/{z}                 single voxel state
GET  /region?x&y&z&range                bounded region query
GET  /ray?x0&y0&z0&x1&y1&z1             straight-line reachability
POST /recover                           reload the durable snapshot

All requests operate on the same service instance (and therefore the same
serialized commit log); every response reflects a fully committed state.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import OccupancyService
from .types import (
    CellState,
    GridConfig,
    Observation,
    PersistenceConfig,
    VoxelSnapshot,
)


class ObservationIn(BaseModel):
    x: int
    y: int
    z: int
    prob_hit: float = Field(gt=0.0, lt=1.0)
    timestamp: int
    weight: float = 1.0


class SubmitIn(BaseModel):
    observations: list[ObservationIn]


def _snapshot_dict(snap: VoxelSnapshot) -> dict:
    return {
        "cell": list(snap.cell),
        "log_odds": snap.log_odds,
        "probability": snap.probability,
        "state": snap.state.value,
        "last_update": snap.last_update,
        "update_count": snap.update_count,
    }


def create_app(
    service: OccupancyService | None = None,
    *,
    grid_config: GridConfig | None = None,
    persistence_config: PersistenceConfig | None = None,
) -> FastAPI:
    """Build the HTTP app, recovering durable state when a store is given."""
    app = FastAPI(title="local-voxel-occupancy", version="1.0.0")

    if service is None:
        service = OccupancyService.start(
            grid_config=grid_config, persistence_config=persistence_config
        )
    app.state.service = service

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": service.snapshot_version()}

    @app.post("/observations")
    def submit(body: SubmitIn) -> dict:
        batch = [
            Observation(
                cell=(o.x, o.y, o.z),
                prob_hit=o.prob_hit,
                timestamp=o.timestamp,
                weight=o.weight,
            )
            for o in body.observations
        ]
        result = service.submit(batch)
        return {
            "sequence": result.sequence,
            "applied": result.applied,
            "rejected_expired": result.rejected_expired,
        }

    @app.exception_handler(ValueError)
    def value_error_handler(request, exc):  # noqa: ANN001
        raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/cells/{x}/{y}/{z}")
    def get_cell(x: int, y: int, z: int) -> dict:
        snap = service.get_cell((x, y, z))
        if snap is None:
            # Unobserved cell: flat prior, unknown state, no storage used.
            return {
                "cell": [x, y, z],
                "log_odds": 0.0,
                "probability": 0.5,
                "state": CellState.UNKNOWN.value,
                "last_update": None,
                "update_count": 0,
            }
        return _snapshot_dict(snap)

    @app.get("/region")
    def region(x: int, y: int, z: int, range: float) -> dict:  # noqa: A002
        if range < 0:
            raise HTTPException(status_code=400, detail="range must be >= 0")
        result = service.query_region((x, y, z), range)
        return {
            "center": [x, y, z],
            "max_range": range,
            "cells": [
                list(cell) for cell in result.cells
            ],
            "states": {
                ",".join(str(v) for v in cell): _snapshot_dict(snap)
                for cell, snap in result.cells.items()
            },
        }

    @app.get("/ray")
    def ray(x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> dict:
        result = service.query_ray((x0, y0, z0), (x1, y1, z1))
        return {
            "reachable": result.reachable,
            "blocked_by": list(result.blocked_by) if result.blocked_by else None,
            "traced_cells": [list(c) for c in result.traced_cells],
        }

    @app.post("/recover")
    def recover() -> dict:
        sequence = service.recover()
        return {"recovered_version": sequence}

    return app
