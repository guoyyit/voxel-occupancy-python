"""Service entry point: ``uv run uvicorn main:app``."""
from voxel_occupancy.api import create_app

app = create_app()
