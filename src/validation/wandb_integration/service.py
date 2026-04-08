"""Weights & Biases integration service for experiment tracking."""

from typing import Dict, List, Optional, Any
import structlog

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ...core.config import settings

logger = structlog.get_logger()


class WandbService:
    """Service for Weights & Biases experiment tracking and logging."""

    def __init__(self):
        """Initialize the W&B service."""
        self.enabled = self._is_enabled()
        if self.enabled:
            self._init_wandb()

    def _is_enabled(self) -> bool:
        """Check if W&B is enabled and available."""
        if not WANDB_AVAILABLE:
            logger.warning("wandb package not available")
            return False

        if not settings.wandb_api_key:
            logger.info("W&B API key not configured")
            return False

        return True

    def _init_wandb(self):
        """Initialize W&B with configuration."""
        try:
            wandb.login(key=settings.wandb_api_key)

            # Initialize with default settings
            wandb.init(
                project=settings.wandb_project,
                entity=settings.wandb_entity,
                config={
                    "model": settings.ollama_model,
                    "embedding_model": settings.ollama_embedding_model,
                    "vector_dimension": settings.vector_dimension,
                    "similarity_threshold": settings.similarity_threshold,
                },
                reinit=True,  # Allow multiple initializations
            )

            logger.info("W&B initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize W&B", error=str(e))
            self.enabled = False

    async def log_query_evaluation(
        self,
        query: str,
        response: Any,
        metrics: Dict[str, float],
        response_time_ms: int,
    ):
        """
        Log a query evaluation to W&B.

        Args:
            query: The original query
            response: The API response object
            metrics: Evaluation metrics
            response_time_ms: Response time in milliseconds
        """
        if not self.enabled:
            return

        try:
            # Prepare log data
            log_data = {
                "query": query,
                "response_length": len(response.answer)
                if hasattr(response, "answer")
                else 0,
                "response_time_ms": response_time_ms,
                "has_sql": bool(
                    hasattr(response, "sql_response") and response.sql_response
                ),
                "documents_retrieved": len(response.documents)
                if hasattr(response, "documents")
                else 0,
            }

            # Add metrics
            log_data.update(metrics)

            # Add SQL-specific metrics if available
            if hasattr(response, "sql_response") and response.sql_response:
                sql_resp = response.sql_response
                log_data.update(
                    {
                        "sql_confidence": sql_resp.confidence,
                        "sql_length": len(sql_resp.sql),
                    }
                )

            # Log to W&B
            wandb.log(log_data)

            logger.info("Query evaluation logged to W&B", metrics_count=len(metrics))

        except Exception as e:
            logger.error("Failed to log to W&B", error=str(e))

    async def log_batch_evaluation(
        self,
        batch_metrics: Dict[str, float],
        batch_size: int,
        batch_id: Optional[str] = None,
    ):
        """
        Log batch evaluation metrics to W&B.

        Args:
            batch_metrics: Aggregate metrics for the batch
            batch_size: Number of evaluations in the batch
            batch_id: Optional batch identifier
        """
        if not self.enabled:
            return

        try:
            log_data = {
                "batch_size": batch_size,
                "batch_id": batch_id or "unknown",
            }
            log_data.update(batch_metrics)

            wandb.log(log_data)

            logger.info("Batch evaluation logged to W&B", batch_size=batch_size)

        except Exception as e:
            logger.error("Failed to log batch to W&B", error=str(e))

    async def log_model_comparison(
        self, model_configs: List[Dict[str, Any]], comparison_metrics: Dict[str, Any]
    ):
        """
        Log model comparison results to W&B.

        Args:
            model_configs: List of model configurations compared
            comparison_metrics: Comparison results and metrics
        """
        if not self.enabled:
            return

        try:
            # Create a summary table
            table_data = []
            for i, config in enumerate(model_configs):
                row = {
                    "model_index": i,
                    "model_name": config.get("model", "unknown"),
                    "config": str(config),
                }
                # Add metrics for this model if available
                for metric_name, values in comparison_metrics.items():
                    if isinstance(values, list) and i < len(values):
                        row[metric_name] = values[i]

                table_data.append(row)

            # Log comparison table
            wandb.log(
                {
                    "model_comparison": wandb.Table(
                        dataframe=table_data,
                        columns=list(table_data[0].keys()) if table_data else [],
                    ),
                    **comparison_metrics,
                }
            )

            logger.info(
                "Model comparison logged to W&B", models_compared=len(model_configs)
            )

        except Exception as e:
            logger.error("Failed to log model comparison", error=str(e))

    async def finish_run(self):
        """Finish the current W&B run."""
        if self.enabled and wandb.run is not None:
            try:
                wandb.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.error("Failed to finish W&B run", error=str(e))

    async def start_new_run(
        self, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ):
        """Start a new W&B run."""
        if not self.enabled:
            return

        try:
            await self.finish_run()  # Finish any existing run

            wandb.init(
                project=settings.wandb_project,
                entity=settings.wandb_entity,
                name=run_name,
                config=config or {},
                reinit=True,
            )

            logger.info("New W&B run started", run_name=run_name)

        except Exception as e:
            logger.error("Failed to start new W&B run", error=str(e))
