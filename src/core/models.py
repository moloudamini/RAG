"""Database models for the RAG system."""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Query(Base):
    """User query log for evaluation and analytics."""

    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(100))
    natural_language_query: Mapped[str] = mapped_column(Text, nullable=False)
    generated_sql: Mapped[Optional[str]] = mapped_column(Text)
    sql_execution_result: Mapped[Optional[dict]] = mapped_column(JSON)
    llm_response: Mapped[Optional[str]] = mapped_column(Text)
    retrieved_documents: Mapped[Optional[list]] = mapped_column(JSON)
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    evaluations: Mapped[List["QueryEvaluation"]] = relationship(back_populates="query")


class QueryEvaluation(Base):
    """Evaluation results for queries."""

    __tablename__ = "query_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    query_id: Mapped[int] = mapped_column(Integer, ForeignKey("queries.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query: Mapped["Query"] = relationship(back_populates="evaluations")


# Indexes
Index("ix_queries_created_at", Query.created_at)
