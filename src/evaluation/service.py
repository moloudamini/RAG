"""Evaluation service for measuring RAG system performance."""

from typing import TYPE_CHECKING, Dict, List, Optional, Any
import structlog

from ..core.config import settings

if TYPE_CHECKING:
    from ..llm.service import LLMService

logger = structlog.get_logger()


class EvaluationService:
    """Service for evaluating RAG system performance metrics."""

    def __init__(self, llm: Optional["LLMService"] = None):
        """Initialize the evaluation service.

        Args:
            llm: Optional LLMService used for LLM-as-judge scoring. When
                 provided, faithfulness/answer_relevance/completeness metrics
                 are computed via a second LLM call instead of heuristics.
        """
        self.metrics_config = settings.evaluation_metrics
        self._llm = llm

    async def evaluate_query(
        self,
        query: str,
        response: Any,
        sql_result: Optional[Dict] = None,
        retrieved_docs: List[Dict] = [],
        cited_indices: List[int] = [],
    ) -> Dict[str, float]:
        """
        Evaluate a query response and return metrics.

        Args:
            query: The original natural language query
            response: The complete API response object
            sql_result: SQL generation results
            retrieved_docs: Retrieved documents

        Returns:
            Dict of metric names to values
        """
        logger.info("Evaluating query response", query=query[:50])

        metrics = {}

        try:
            # SQL Accuracy (if SQL was generated)
            if sql_result and "sql_accuracy" in self.metrics_config:
                metrics["sql_accuracy"] = self._evaluate_sql_accuracy(query, sql_result)

            # Link Accuracy (relevance of retrieved documents)
            if retrieved_docs and "link_accuracy" in self.metrics_config:
                metrics["link_accuracy"] = self._evaluate_link_accuracy(
                    query, retrieved_docs, cited_indices
                )

            # Citation Accuracy (fraction of cited indices that are valid)
            if cited_indices and "citation_accuracy" in self.metrics_config:
                metrics["citation_accuracy"] = self._evaluate_citation_accuracy(
                    cited_indices, len(retrieved_docs)
                )

            # LLM-as-judge metrics (faithfulness, answer_relevance, completeness)
            judge_metrics_requested = any(
                m in self.metrics_config
                for m in ("faithfulness", "answer_relevance", "completeness")
            )
            if self._llm is not None and judge_metrics_requested and retrieved_docs:
                answer_text = (
                    response.answer if hasattr(response, "answer") else str(response)
                )
                judge_scores = await self._llm.judge_answer(
                    query, answer_text, retrieved_docs
                )
                for metric in ("faithfulness", "answer_relevance", "completeness"):
                    if metric in self.metrics_config:
                        metrics[metric] = judge_scores.get(metric, 0.0)

            # Response Time (if available)
            if hasattr(response, "response_time_ms") and response.response_time_ms:
                metrics["response_time_ms"] = response.response_time_ms

            logger.info("Evaluation completed", metrics=metrics)
            return metrics

        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            return {"evaluation_error": 1.0}

    def _evaluate_sql_accuracy(self, query: str, sql_result: Dict[str, Any]) -> float:
        """Evaluate the accuracy of generated SQL."""
        if not sql_result.get("is_valid", False):
            return 0.0

        sql = sql_result.get("sql", "").upper()
        query_upper = query.upper()

        score = 0.0

        # Basic keyword matching
        sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY"]
        query_keywords = []

        # Extract potential SQL keywords from query
        for keyword in sql_keywords:
            if keyword in query_upper:
                query_keywords.append(keyword)

        # Check if SQL contains expected keywords
        for keyword in query_keywords:
            if keyword in sql:
                score += 0.2

        # Check for dangerous operations (should be low score)
        dangerous_ops = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
        for op in dangerous_ops:
            if op in sql:
                score -= 0.5

        # Length appropriateness (not too short, not too long)
        sql_length = len(sql_result.get("sql", ""))
        if 20 <= sql_length <= 500:
            score += 0.2

        # Confidence from SQL generation
        confidence = sql_result.get("confidence", 0.5)
        score = (score + confidence) / 2

        return max(0.0, min(1.0, score))

    def _evaluate_link_accuracy(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        cited_indices: List[int] = [],
    ) -> float:
        """Evaluate retrieval quality using vector similarity scores.

        Returns the average similarity score of retrieved docs, gated by the
        fraction that meet the similarity threshold. This purely reflects the
        retrieval system's output — no heuristic string matching.
        """
        if not retrieved_docs:
            return 0.0

        scores = [doc.get("similarity_score", 0.0) for doc in retrieved_docs]
        avg_score = sum(scores) / len(scores)

        above_threshold = sum(
            1 for s in scores if s >= settings.similarity_threshold
        )
        coverage_ratio = above_threshold / len(scores)

        return min(1.0, avg_score * coverage_ratio)

    def _evaluate_citation_accuracy(
        self, cited_indices: List[int], doc_count: int
    ) -> float:
        """Evaluate whether cited document indices are valid (non-hallucinated).

        A citation is valid if its index is within the range of retrieved documents.
        Score = fraction of cited indices that are in [1, doc_count].
        """
        if not cited_indices:
            return 0.0

        valid = sum(1 for idx in cited_indices if 1 <= idx <= doc_count)
        return valid / len(cited_indices)


    async def evaluate_batch(
        self, evaluations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of query-response pairs and return aggregate metrics.

        Args:
            evaluations: List of evaluation data

        Returns:
            Dict of aggregate metric names to values
        """
        if not evaluations:
            return {}

        metrics_sum = {}
        metrics_count = {}

        for eval_data in evaluations:
            metrics = await self.evaluate_query(
                eval_data.get("query", ""),
                eval_data.get("response", ""),
                eval_data.get("sql_result"),
                eval_data.get("retrieved_docs", []),
            )

            for metric_name, value in metrics.items():
                if metric_name not in metrics_sum:
                    metrics_sum[metric_name] = 0.0
                    metrics_count[metric_name] = 0

                metrics_sum[metric_name] += value
                metrics_count[metric_name] += 1

        # Calculate averages
        aggregate_metrics = {}
        for metric_name in metrics_sum:
            aggregate_metrics[f"avg_{metric_name}"] = (
                metrics_sum[metric_name] / metrics_count[metric_name]
            )

        return aggregate_metrics
