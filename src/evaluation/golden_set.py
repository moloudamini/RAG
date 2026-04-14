"""Golden set evaluator for SQL analytics queries."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import structlog

from ..core.database import get_db_session_context
from ..text_to_sql.service import TextToSQLService

logger = structlog.get_logger()

_GOLDEN_SET_PATH = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "sql_golden_set.json"


def _load_golden_set(path: Path = _GOLDEN_SET_PATH) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _normalize_result(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize result rows for comparison: lowercase keys, sort rows by string repr."""
    normalized = [{k.lower(): v for k, v in row.items()} for row in data]
    return sorted(normalized, key=lambda r: str(sorted(r.items())))


def _score_entry(
    entry: Dict[str, Any],
    generated_sql: str,
    sql_result: Dict[str, Any],
    reference_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score a single golden set entry against the generated SQL and its result."""
    sql_upper = generated_sql.upper()
    result_columns = [c.lower() for c in sql_result.get("columns", [])]
    row_count = sql_result.get("row_count", 0)
    has_error = "error" in sql_result

    scores: Dict[str, float] = {}

    # 1. Execution success
    scores["execution_success"] = 0.0 if has_error else 1.0

    # 2. SQL keyword coverage
    keywords = entry.get("expected_sql_keywords", [])
    if keywords:
        matched = sum(1 for kw in keywords if kw.upper() in sql_upper)
        scores["keyword_coverage"] = matched / len(keywords)
    else:
        scores["keyword_coverage"] = 1.0

    # 3. Column presence (at least one expected column appears in result)
    expected_cols = [c.lower() for c in entry.get("expected_columns", [])]
    if expected_cols and not has_error:
        matched_cols = sum(
            1 for ec in expected_cols
            if any(ec in rc for rc in result_columns)
        )
        scores["column_match"] = matched_cols / len(expected_cols)
    else:
        scores["column_match"] = 1.0 if not expected_cols else 0.0

    # 4. Row count check
    exact = entry.get("expected_row_count")
    minimum = entry.get("expected_row_count_min")
    if exact is not None:
        scores["row_count_match"] = 1.0 if row_count == exact else 0.0
    elif minimum is not None:
        scores["row_count_match"] = 1.0 if row_count >= minimum else 0.0
    else:
        scores["row_count_match"] = 1.0

    # 5. No dangerous operations
    dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    scores["safety"] = 0.0 if any(kw in sql_upper for kw in dangerous) else 1.0

    # 6. Result match against reference SQL (only when reference executed successfully)
    if reference_result is not None and "error" not in reference_result and not has_error:
        gen_normalized = _normalize_result(sql_result.get("data", []))
        ref_normalized = _normalize_result(reference_result.get("data", []))
        scores["result_match"] = 1.0 if gen_normalized == ref_normalized else 0.0

    overall = sum(scores.values()) / len(scores)

    return {
        "id": entry["id"],
        "query": entry["query"],
        "tags": entry.get("tags", []),
        "generated_sql": generated_sql,
        "scores": scores,
        "overall": overall,
        "passed": overall >= 0.7,
    }


class GoldenSetEvaluator:
    """Runs the SQL golden set against the live AnalyticsAgent pipeline."""

    def __init__(self, golden_set_path: Optional[Path] = None):
        self.text_to_sql = TextToSQLService()
        self._path = golden_set_path or _GOLDEN_SET_PATH

    async def run(self, tag_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Run all golden set entries through text-to-SQL and score results.

        Args:
            tag_filter: Only run entries whose tags include this value.

        Returns:
            Summary dict with per-entry results and aggregate scores by tag.
        """
        entries = _load_golden_set(self._path)
        if tag_filter:
            entries = [e for e in entries if tag_filter in e.get("tags", [])]

        results = []
        for entry in entries:
            result = await self._evaluate_entry(entry)
            results.append(result)
            logger.info(
                "Golden set entry evaluated",
                id=entry["id"],
                overall=result["overall"],
                passed=result["passed"],
            )

        return {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "pass_rate": sum(1 for r in results if r["passed"]) / len(results) if results else 0.0,
            "aggregate_by_tag": _aggregate_by_tag(results),
            "entries": results,
        }

    async def _evaluate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        generated_sql = ""
        sql_result: Dict[str, Any] = {"error": "not executed", "data": [], "row_count": 0}
        reference_result: Optional[Dict[str, Any]] = None

        try:
            async with get_db_session_context() as session:
                # Generate and execute the LLM SQL
                sql_result_raw = await self.text_to_sql.generate_sql(entry["query"], session)
                generated_sql = sql_result_raw.get("sql", "")

                if generated_sql:
                    sql_result = await self.text_to_sql.execute_sql(generated_sql, session)
                else:
                    sql_result = {"error": "no SQL generated", "data": [], "row_count": 0}

                # Execute reference SQL if provided
                ref_sql = entry.get("reference_sql")
                if ref_sql:
                    try:
                        reference_result = await self.text_to_sql.execute_sql(ref_sql, session)
                    except Exception as e:
                        logger.warning(
                            "Reference SQL execution failed",
                            id=entry["id"],
                            error=str(e),
                        )

        except Exception as e:
            logger.error("Golden set entry failed", id=entry["id"], error=str(e))
            sql_result = {"error": str(e), "data": [], "row_count": 0}

        return _score_entry(entry, generated_sql, sql_result, reference_result)


def _aggregate_by_tag(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    tag_map: Dict[str, List[float]] = {}
    for r in results:
        for tag in r.get("tags", []):
            tag_map.setdefault(tag, []).append(r["overall"])

    return {
        tag: {
            "count": len(scores),
            "avg_score": sum(scores) / len(scores),
            "pass_rate": sum(1 for s in scores if s >= 0.7) / len(scores),
        }
        for tag, scores in tag_map.items()
    }
