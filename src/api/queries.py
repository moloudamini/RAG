"""Query API endpoints for text-to-SQL and Q&A using LangGraph agents."""

from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..agents.orchestrator import AgentOrchestrator
from ..core.database import get_db_session

logger = structlog.get_logger()
router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for natural language queries."""

    query: str = Field(
        ..., description="Natural language query", min_length=1, max_length=1000
    )
    company_id: Optional[int] = Field(None, description="Company ID for context")
    force_agent: Optional[str] = Field(
        None, description="Force specific agent: 'qa' or 'analytics'"
    )


class SQLResponse(BaseModel):
    """Response model for SQL generation."""

    sql: Optional[str] = Field(None, description="Generated SQL query")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    is_valid: Optional[bool] = Field(None, description="Whether SQL is valid")


class DocumentResponse(BaseModel):
    """Response model for retrieved documents."""

    id: int
    title: str
    content: str
    similarity_score: float
    metadata: dict


class QueryResponse(BaseModel):
    """Complete response for natural language queries."""

    query: str
    agent_used: str = Field(..., description="Agent that processed the query")
    sql_response: Optional[SQLResponse] = None
    documents: List[DocumentResponse] = []
    answer: str = Field(..., description="Generated answer")
    response_time_ms: int
    tokens_used: Optional[int] = None
    evaluation_metrics: Optional[dict] = None


@router.post("/", response_model=QueryResponse)
async def process_query(
    request: QueryRequest, db: AsyncSession = Depends(get_db_session)
) -> QueryResponse:
    """Process a natural language query using LangGraph agents."""

    import asyncio

    start_time = asyncio.get_running_loop().time()
    logger.info("Processing query", query=request.query[:100])

    try:
        # Initialize agent orchestrator
        orchestrator = AgentOrchestrator()

        # Process query with appropriate agent
        result = await orchestrator.process_query(
            request.query, request.company_id, request.force_agent
        )

        # Convert result to API response format
        sql_response = None
        if result.get("sql_query"):
            sql_response = SQLResponse(
                sql=result["sql_query"],
                confidence=0.8,  # TODO: Get from actual validation
                is_valid=result["sql_query"] is not None,
            )

        documents = [
            DocumentResponse(
                id=doc["id"],
                title=doc["title"],
                content=doc["content"][:500],  # Truncate for response
                similarity_score=doc["similarity_score"],
                metadata=doc["metadata"],
            )
            for doc in result.get("documents", [])
        ]

        response = QueryResponse(
            query=request.query,
            agent_used=result.get("agent_used", "unknown"),
            sql_response=sql_response,
            documents=documents,
            answer=result.get("answer", ""),
            response_time_ms=result.get("response_time_ms", 0),
            tokens_used=result.get("tokens_used"),
            evaluation_metrics=result.get("evaluation_metrics"),
        )

        total_time = int((asyncio.get_running_loop().time() - start_time) * 1000)
        logger.info(
            "Query processed",
            agent=result.get("agent_used"),
            total_time_ms=total_time,
            response_time_ms=result.get("response_time_ms", 0),
        )

        return response

    except Exception as e:
        logger.error("Query processing failed", error=str(e))
        raise HTTPException(status_code=500, detail="Query processing failed")


@router.post("/qa", response_model=QueryResponse)
async def process_qa_query(
    request: QueryRequest, db: AsyncSession = Depends(get_db_session)
) -> QueryResponse:
    """Process a query using the Q&A agent specifically."""
    forced = QueryRequest(
        query=request.query, company_id=request.company_id, force_agent="qa"
    )
    return await process_query(forced, db)


@router.post("/analytics", response_model=QueryResponse)
async def process_analytics_query(
    request: QueryRequest, db: AsyncSession = Depends(get_db_session)
) -> QueryResponse:
    """Process a query using the Analytics agent specifically."""
    forced = QueryRequest(
        query=request.query, company_id=request.company_id, force_agent="analytics"
    )
    return await process_query(forced, db)


@router.get("/agents")
async def list_agents():
    """List available agents and their capabilities."""
    return {
        "agents": {
            "qa": {
                "description": "Question-Answering agent for general queries using document retrieval",
                "capabilities": [
                    "document_search",
                    "knowledge_base_qa",
                    "general_questions",
                ],
            },
            "analytics": {
                "description": "Analytics agent for data analysis queries using text-to-SQL",
                "capabilities": [
                    "sql_generation",
                    "data_analysis",
                    "reporting",
                    "metrics",
                ],
            },
        },
        "auto_classification": {
            "description": "Automatically route queries to appropriate agent based on content",
            "method": "keyword_and_pattern_matching",
        },
    }
