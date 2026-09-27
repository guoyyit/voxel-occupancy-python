"""服务入口：uvicorn main:app"""

from voxel_occupancy.server import app

__all__ = ["app"]
