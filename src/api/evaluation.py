"""Evaluation API endpoints."""

from typing import Optional

from fastapi import APIRouter, Query
import structlog

from ..evaluation.golden_set import GoldenSetEvaluator

router = APIRouter()
logger = structlog.get_logger()


@router.post("/golden-set")
async def run_golden_set(
    tag: Optional[str] = Query(default=None, description="Filter entries by tag"),
) -> dict:
    """
    Run the SQL golden set evaluation against the live database.

    Returns per-entry scores and aggregate pass rates by tag.
    """
    evaluator = GoldenSetEvaluator()
    result = await evaluator.run(tag_filter=tag)
    logger.info(
        "Golden set evaluation complete",
        total=result["total"],
        passed=result["passed"],
        pass_rate=result["pass_rate"],
    )
    return result
