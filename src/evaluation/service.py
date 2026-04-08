"""Evaluation service for measuring RAG system performance."""

from typing import Dict, List, Optional, Any
import structlog
from difflib import SequenceMatcher
import re

from ..core.config import settings

logger = structlog.get_logger()


class EvaluationService:
    """Service for evaluating RAG system performance metrics."""

    def __init__(self):
        """Initialize the evaluation service."""
        self.metrics_config = settings.evaluation_metrics

    async def evaluate_query(
        self,
        query: str,
        response: Any,
        sql_result: Optional[Dict] = None,
        retrieved_docs: List[Dict] = [],
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
                    query, retrieved_docs
                )

            # Answer Relevance
            if "relevance" in self.metrics_config:
                metrics["relevance"] = self._evaluate_answer_relevance(
                    query,
                    response.answer if hasattr(response, "answer") else str(response),
                )

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
        self, query: str, retrieved_docs: List[Dict[str, Any]]
    ) -> float:
        """Evaluate the relevance of retrieved documents."""
        if not retrieved_docs:
            return 0.0

        total_relevance = 0.0
        query_lower = query.lower()

        for doc in retrieved_docs:
            content = doc.get("content", "").lower()
            title = doc.get("title", "").lower()

            # Calculate text similarity
            content_similarity = SequenceMatcher(None, query_lower, content).ratio()
            title_similarity = SequenceMatcher(None, query_lower, title).ratio()

            # Weighted combination
            relevance = (content_similarity * 0.7) + (title_similarity * 0.3)

            # Boost if document has high similarity score from vector search
            vector_similarity = doc.get("similarity_score", 0.0)
            relevance = (relevance + vector_similarity) / 2

            total_relevance += relevance

        # Average relevance across all documents
        avg_relevance = total_relevance / len(retrieved_docs)

        # Apply threshold - documents below similarity threshold don't count
        relevant_docs = [
            doc
            for doc in retrieved_docs
            if doc.get("similarity_score", 0.0) >= settings.similarity_threshold
        ]

        coverage_ratio = len(relevant_docs) / max(len(retrieved_docs), 1)

        return avg_relevance * coverage_ratio

    def _evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Evaluate how relevant the answer is to the query."""
        if not answer or not query:
            return 0.0

        query_lower = query.lower()
        answer_lower = answer.lower()

        # Direct text overlap
        query_words = set(re.findall(r"\b\w+\b", query_lower))
        answer_words = set(re.findall(r"\b\w+\b", answer_lower))

        overlap = len(query_words.intersection(answer_words))
        total_unique = len(query_words.union(answer_words))

        if total_unique == 0:
            return 0.0

        overlap_score = overlap / total_unique

        # Sequence similarity
        seq_similarity = SequenceMatcher(None, query_lower, answer_lower).ratio()

        # Length appropriateness (answers shouldn't be too short or too long)
        answer_length = len(answer)
        length_score = 1.0
        if answer_length < 10:
            length_score = 0.3  # Too short
        elif answer_length > 2000:
            length_score = 0.7  # Too long

        # Combine scores
        relevance = (
            (overlap_score * 0.4) + (seq_similarity * 0.4) + (length_score * 0.2)
        )

        return max(0.0, min(1.0, relevance))

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
