"""API routes for the RAG system."""

from fastapi import APIRouter

from . import queries, health, metrics

api_router = APIRouter()

# Include route modules
api_router.include_router(queries.router, prefix="/queries", tags=["queries"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
