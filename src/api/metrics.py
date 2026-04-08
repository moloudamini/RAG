"""Metrics API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_metrics():
    """Get application metrics."""
    # TODO: Return actual metrics from Prometheus
    return {
        "total_queries": 0,
        "average_response_time_ms": 0,
        "sql_accuracy": 0.0,
        "document_relevance": 0.0,
    }


@router.get("/evaluation")
async def get_evaluation_metrics():
    """Get evaluation metrics summary."""
    # TODO: Return aggregated evaluation metrics
    return {
        "total_evaluations": 0,
        "average_sql_accuracy": 0.0,
        "average_link_accuracy": 0.0,
        "average_relevance": 0.0,
    }
