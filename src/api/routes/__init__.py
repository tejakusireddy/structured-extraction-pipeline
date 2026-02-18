"""API route aggregation.

All sub-routers are collected into a single api_router that the
app factory mounts under the configured prefix.
"""

from fastapi import APIRouter

from src.api.routes import extraction, graph, health, ingestion, search

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(ingestion.router)
api_router.include_router(extraction.router)
api_router.include_router(search.router)
api_router.include_router(graph.router)
