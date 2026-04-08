"""Monitoring and metrics service."""

from typing import Dict, Optional
import structlog

try:
    from prometheus_client import Counter, Histogram, Gauge, Info

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = structlog.get_logger()


class MonitoringService:
    """Service for application monitoring and metrics collection."""

    def __init__(self):
        """Initialize monitoring service."""
        self.metrics = None
        if PROMETHEUS_AVAILABLE:
            self._init_metrics()

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        try:
            # Query metrics
            self.query_total = Counter(
                "rag_queries_total", "Total number of queries processed", ["status"]
            )

            self.query_duration = Histogram(
                "rag_query_duration_seconds", "Query processing duration", ["operation"]
            )

            # SQL generation metrics
            self.sql_generation_total = Counter(
                "rag_sql_generation_total", "Total SQL generation attempts", ["result"]
            )

            self.sql_accuracy = Histogram(
                "rag_sql_accuracy", "SQL generation accuracy distribution"
            )

            # Retrieval metrics
            self.retrieval_total = Counter(
                "rag_retrieval_total", "Total document retrieval operations"
            )

            self.documents_retrieved = Histogram(
                "rag_documents_retrieved_count",
                "Number of documents retrieved per query",
            )

            # LLM metrics
            self.llm_requests_total = Counter(
                "rag_llm_requests_total",
                "Total LLM API requests",
                ["model", "operation"],
            )

            self.llm_tokens_used = Counter(
                "rag_llm_tokens_total", "Total tokens used by LLM", ["model"]
            )

            # System health metrics
            self.db_connection_pool_size = Gauge(
                "rag_db_connection_pool_size", "Database connection pool size"
            )

            self.ollama_health = Gauge(
                "rag_ollama_health", "Ollama service health (1=healthy, 0=unhealthy)"
            )

            # Application info
            self.app_info = Info("rag_app", "RAG application information")
            self.app_info.info({"version": "0.1.0", "service": "rag-system"})

            logger.info("Prometheus metrics initialized")

        except Exception as e:
            logger.error("Failed to initialize metrics", error=str(e))

    async def record_query(
        self,
        status: str = "success",
        duration: Optional[float] = None,
        operation: str = "query",
    ):
        """Record a query operation."""
        if not self.metrics:
            return

        try:
            self.query_total.labels(status=status).inc()

            if duration is not None:
                self.query_duration.labels(operation=operation).observe(duration)

        except Exception as e:
            logger.error("Failed to record query metric", error=str(e))

    async def record_sql_generation(
        self, result: str = "success", accuracy: Optional[float] = None
    ):
        """Record SQL generation metrics."""
        if not self.metrics:
            return

        try:
            self.sql_generation_total.labels(result=result).inc()

            if accuracy is not None:
                self.sql_accuracy.observe(accuracy)

        except Exception as e:
            logger.error("Failed to record SQL metric", error=str(e))

    async def record_retrieval(self, document_count: int):
        """Record document retrieval metrics."""
        if not self.metrics:
            return

        try:
            self.retrieval_total.inc()
            self.documents_retrieved.observe(document_count)

        except Exception as e:
            logger.error("Failed to record retrieval metric", error=str(e))

    async def record_llm_usage(
        self, model: str, operation: str, tokens_used: Optional[int] = None
    ):
        """Record LLM usage metrics."""
        if not self.metrics:
            return

        try:
            self.llm_requests_total.labels(model=model, operation=operation).inc()

            if tokens_used is not None:
                self.llm_tokens_used.labels(model=model).inc(tokens_used)

        except Exception as e:
            logger.error("Failed to record LLM metric", error=str(e))

    async def update_health_metrics(
        self, db_pool_size: Optional[int] = None, ollama_healthy: Optional[bool] = None
    ):
        """Update system health metrics."""
        if not self.metrics:
            return

        try:
            if db_pool_size is not None:
                self.db_connection_pool_size.set(db_pool_size)

            if ollama_healthy is not None:
                self.ollama_health.set(1.0 if ollama_healthy else 0.0)

        except Exception as e:
            logger.error("Failed to update health metrics", error=str(e))

    def get_metrics_summary(self) -> Dict[str, float]:
        """Get a summary of current metrics values."""
        # This is a simplified version - in production you'd collect
        # actual metric values from Prometheus
        return {
            "total_queries": 0,  # Would be collected from Prometheus
            "average_response_time": 0.0,
            "sql_accuracy_avg": 0.0,
            "documents_retrieved_avg": 0.0,
        }


# Global monitoring instance
monitoring = MonitoringService()
