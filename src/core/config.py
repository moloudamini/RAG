"""Core configuration and settings for the RAG system."""

import os
from typing import Optional

from pydantic import Field, computed_field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Settings
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=1, description="Number of API workers")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # Database Settings
    database_url: str = Field(
        default="sqlite+aiosqlite:///./rag.db",
        description="Database URL (SQLite for development)",
    )

    # Ollama Settings
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    ollama_model: str = Field(
        default="llama3.2", description="Default Ollama model for text generation"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", description="Ollama model for embeddings"
    )

    # Vector Database Settings
    vector_dimension: int = Field(default=768, description="Vector embedding dimension")
    similarity_threshold: float = Field(
        default=0.3, description="Minimum similarity threshold for retrieval"
    )

    # Hybrid Retrieval Settings
    bm25_weight: float = Field(
        default=0.3, description="BM25 score weight in hybrid fusion"
    )
    vector_weight: float = Field(
        default=0.7, description="Vector score weight in hybrid fusion"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )
    use_reranker: bool = Field(
        default=True, description="Enable cross-encoder reranking"
    )

    # Security Settings
    secret_key: str = Field(
        default_factory=lambda: os.urandom(32).hex(),
        description="Secret key for JWT tokens",
    )
    jwt_expiration_hours: int = Field(
        default=24, description="JWT token expiration in hours"
    )

    # W&B Settings
    wandb_project: str = Field(default="rag-system", description="W&B project name")
    wandb_entity: Optional[str] = Field(
        default=None, description="W&B entity/team name"
    )
    wandb_api_key: Optional[str] = Field(default=None, description="W&B API key")

    # Evaluation Settings
    evaluation_batch_size: int = Field(
        default=10, description="Batch size for evaluation"
    )
    evaluation_metrics: list[str] = Field(
        default=["sql_accuracy", "link_accuracy", "relevance", "response_time"],
        description="Evaluation metrics to compute",
    )

    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")

    # Performance Settings
    max_concurrent_requests: int = Field(
        default=10, description="Max concurrent requests"
    )
    request_timeout: int = Field(default=30, description="Request timeout in seconds")

    @computed_field
    @property
    def database_config(self) -> dict:
        """Parsed database configuration."""
        # Parse database URL for connection details
        return {
            "url": self.database_url,
            "pool_pre_ping": True,
            "echo": self.log_level == "DEBUG",
        }

    @computed_field
    @property
    def ollama_config(self) -> dict:
        """Ollama client configuration."""
        return {
            "base_url": self.ollama_base_url,
            "model": self.ollama_model,
            "embedding_model": self.ollama_embedding_model,
        }


# Global settings instance
settings = Settings()
