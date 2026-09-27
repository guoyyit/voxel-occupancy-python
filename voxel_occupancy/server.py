"""FastAPI HTTP 接口：观测提交、占用/区域/直线查询、统计。

所有接口直接作用于 OccupancyService：
- POST /batches 提交一批原子观测（可带幂等键）；
- GET  /voxels/{x}/{y}/{z} 单体素查询；
- GET  /regions 带球心与最大范围的区域查询；
- GET  /lines 给定起止点的直线可达查询；
- GET  /stats 栅格统计；
- POST /flush 显式落盘。
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import ServiceConfig
from .service import OccupancyService
from .types import (
    Observation,
    ObservationBatch,
    ObservationKind,
)

_KIND_MAP = {
    "occupied": ObservationKind.OCCUPIED,
    "free": ObservationKind.FREE,
    "clear": ObservationKind.CLEAR,
}


class ObservationIn(BaseModel):
    x: int
    y: int
    z: int
    kind: Literal["occupied", "free", "clear"] = "occupied"
    timestamp: float = 0.0
    weight: float = Field(default=1.0, ge=0.0)


class BatchIn(BaseModel):
    observations: list[ObservationIn]
    sequence: int = 0
    timestamp: float = 0.0
    idempotency_key: str | None = None


def create_app(service: OccupancyService | None = None) -> FastAPI:
    """构造 FastAPI 应用，可注入自定义服务（测试用）。"""
    svc = service or OccupancyService(ServiceConfig())
    app = FastAPI(title="voxel-occupancy", version="0.1.0")

    @app.post("/batches")
    def submit_batch(batch_in: BatchIn) -> dict:
        batch = ObservationBatch(
            observations=tuple(
                Observation(
                    coord=(o.x, o.y, o.z),
                    kind=_KIND_MAP[o.kind],
                    timestamp=o.timestamp,
                    weight=o.weight,
                )
                for o in batch_in.observations
            ),
            sequence=batch_in.sequence,
            timestamp=batch_in.timestamp,
            idempotency_key=batch_in.idempotency_key,
        )
        result = svc.submit(batch)
        return {
            "committed": result.committed,
            "reason": result.reason,
            "touched": result.touched,
            "sequence": result.sequence,
            "dedup": result.dedup,
        }

    @app.get("/voxels/{x}/{y}/{z}")
    def get_voxel(x: int, y: int, z: int) -> dict:
        info = svc.get((x, y, z))
        if info is None:
            # 未观测 / 已清除：显式表达，占用判定为 false
            return {
                "coord": [x, y, z],
                "observed": False,
                "occupied": False,
                "probability": 0.5,
                "log_odds": 0.0,
            }
        return {
            "coord": list(info.coord),
            "observed": True,
            "occupied": info.occupied,
            "probability": info.probability,
            "log_odds": info.log_odds,
        }

    @app.get("/regions")
    def query_region(
        x: int = 0,
        y: int = 0,
        z: int = 0,
        range: float = Query(0.0, ge=0.0),
        occupied_only: bool = False,
    ) -> dict:
        voxels = svc.query_region(
            (x, y, z), range, occupied_only=occupied_only
        )
        return {"count": len(voxels), "voxels": [_voxel_dict(v) for v in voxels]}

    @app.get("/lines")
    def trace_line(
        x0: int, y0: int, z0: int, x1: int, y1: int, z1: int
    ) -> dict:
        result = svc.trace_line((x0, y0, z0), (x1, y1, z1))
        return {
            "start": list(result.start),
            "end": list(result.end),
            "reachable": result.reachable,
            "blocked_coord": list(result.blocked_coord)
            if result.blocked_coord is not None
            else None,
            "path": [list(c) for c in result.path],
        }

    @app.get("/stats")
    def stats() -> dict:
        s = svc.stats()
        return {
            "stored_cells": s.stored_cells,
            "committed_sequence": s.committed_sequence,
            "committed_batches": s.committed_batches,
        }

    @app.post("/flush")
    def flush() -> dict:
        try:
            svc.flush()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"flushed": True}

    return app


def _voxel_dict(v) -> dict:
    return {
        "coord": list(v.coord),
        "occupied": v.occupied,
        "probability": v.probability,
        "log_odds": v.log_odds,
    }


app = create_app(OccupancyService(ServiceConfig()))
