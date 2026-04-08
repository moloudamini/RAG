"""Health check API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "service": "rag-system"}


@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    # TODO: Add actual health checks for database, Ollama, etc.
    return {
        "status": "healthy",
        "components": {
            "database": "healthy",
            "ollama": "healthy",
            "vector_db": "healthy",
        },
        "version": "0.1.0",
    }
