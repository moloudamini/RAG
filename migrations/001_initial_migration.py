"""Initial migration - create all tables."""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables for the RAG system."""

    # Create queries table
    op.create_table(
        "queries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=100), nullable=True),
        sa.Column("natural_language_query", sa.Text(), nullable=False),
        sa.Column("generated_sql", sa.Text(), nullable=True),
        sa.Column("sql_execution_result", sa.JSON(), nullable=True),
        sa.Column("llm_response", sa.Text(), nullable=True),
        sa.Column("retrieved_documents", sa.JSON(), nullable=True),
        sa.Column("response_time_ms", sa.Integer(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_queries_id", "queries", ["id"], unique=False)
    op.create_index("ix_queries_created_at", "queries", ["created_at"], unique=False)

    # Create query_evaluations table
    op.create_table(
        "query_evaluations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("query_id", sa.Integer(), nullable=False),
        sa.Column("metric_name", sa.String(length=100), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("metric_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["query_id"],
            ["queries.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_query_evaluations_id", "query_evaluations", ["id"], unique=False
    )
    op.create_index(
        "ix_query_evaluations_query_id", "query_evaluations", ["query_id"], unique=False
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("query_evaluations")
    op.drop_table("queries")
