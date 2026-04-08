"""LangGraph agents for Q&A and Analytics."""

from typing import Dict, List, Optional, Any, TypedDict
import structlog

from langgraph.graph import StateGraph, END

from ..retrieval.service import RetrievalService
from ..llm.service import LLMService
from ..text_to_sql.service import TextToSQLService
from ..evaluation.service import EvaluationService
from ..validation.wandb_integration.service import WandbService
from ..core.database import get_db_session_context

logger = structlog.get_logger()


class AgentState(TypedDict):
    """State for LangGraph agents."""

    query: str
    company_id: Optional[int]
    context: str
    sql_query: Optional[str]
    sql_result: Optional[Dict[str, Any]]
    documents: List[Dict[str, Any]]
    answer: str
    evaluation_metrics: Dict[str, float]
    response_time_ms: int
    tokens_used: int


class QAAgent:
    """Agent for answering questions based on company knowledge base."""

    def __init__(self):
        """Initialize the Q&A agent."""
        self.retrieval = RetrievalService()
        self.llm = LLMService()
        self.evaluation = EvaluationService()
        self.wandb = WandbService()
        self._graph = None  # Cache compiled graph

    def create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for Q&A."""
        if self._graph is not None:
            return self._graph

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("evaluate_response", self._evaluate_response)

        # Define flow
        workflow.set_entry_point("retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_answer")
        workflow.add_edge("generate_answer", "evaluate_response")
        workflow.add_edge("evaluate_response", END)

        self._graph = workflow.compile()
        return self._graph

    async def process_query(
        self, query: str, company_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a Q&A query using the LangGraph workflow."""
        import asyncio

        start_time = asyncio.get_running_loop().time()

        logger.info("Processing Q&A query", query=query[:100])

        # Initialize state
        initial_state = AgentState(
            query=query,
            company_id=company_id,
            context="",
            sql_query=None,
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        # Execute workflow
        graph = self.create_graph()
        result = await graph.ainvoke(initial_state)

        # Calculate response time
        result["response_time_ms"] = int(
            (asyncio.get_running_loop().time() - start_time) * 1000
        )

        logger.info("Q&A query processed", response_time_ms=result["response_time_ms"])
        return result

    async def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents for the query."""
        try:
            documents = await self.retrieval.retrieve_documents(
                state["query"], state["company_id"], top_k=5
            )

            # Build context from documents
            context_parts = []
            for doc in documents[:3]:  # Use top 3 documents
                context_parts.append(f"Document: {doc['title']}")
                context_parts.append(f"Content: {doc['content'][:1000]}...")  # Truncate
                context_parts.append("")

            state["documents"] = documents
            state["context"] = "\n".join(context_parts)

            logger.info("Documents retrieved for Q&A", count=len(documents))

        except Exception as e:
            logger.error("Document retrieval failed in Q&A", error=str(e))
            # Don't re-raise - allow processing to continue with empty context
            state["context"] = "No relevant documents found."
            state["documents"] = []

        return state

    async def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate answer using retrieved documents."""
        try:
            answer = await self.llm.generate_answer(
                state["query"], state["context"], max_tokens=500, temperature=0.7
            )

            state["answer"] = answer
            state["tokens_used"] = await self.llm.estimate_tokens(
                state["query"] + state["context"] + answer
            )

            logger.info("Answer generated for Q&A", answer_length=len(answer))

        except Exception as e:
            logger.error("Answer generation failed in Q&A", error=str(e))
            state["answer"] = (
                "I apologize, but I was unable to generate an answer at this time."
            )

        return state

    async def _evaluate_response(self, state: AgentState) -> AgentState:
        """Evaluate the Q&A response."""
        try:
            metrics = await self.evaluation.evaluate_query(
                state["query"],
                type(
                    "Response",
                    (),
                    {
                        "answer": state["answer"],
                        "response_time_ms": state["response_time_ms"],
                        "documents": state["documents"],
                        "sql_response": None,
                    },
                )(),
                sql_result=None,
                retrieved_docs=state["documents"],
            )

            state["evaluation_metrics"] = metrics

            # Log to W&B
            if self.wandb.enabled:
                await self.wandb.log_query_evaluation(
                    state["query"],
                    type(
                        "Response",
                        (),
                        {
                            "answer": state["answer"],
                            "response_time_ms": state["response_time_ms"],
                            "documents": state["documents"],
                            "sql_response": None,
                        },
                    )(),
                    metrics,
                    state["response_time_ms"],
                )

        except Exception as e:
            logger.error("Evaluation failed in Q&A", error=str(e))

        return state


class AnalyticsAgent:
    """Agent for handling analytical queries with text-to-SQL."""

    def __init__(self):
        """Initialize the Analytics agent."""
        self.text_to_sql = TextToSQLService()
        self.llm = LLMService()
        self.evaluation = EvaluationService()
        self.wandb = WandbService()
        self._graph = None  # Cache compiled graph

    def create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for Analytics."""
        if self._graph is not None:
            return self._graph  # type: ignore

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("validate_sql", self._validate_sql)
        workflow.add_node("execute_sql", self._execute_sql)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("evaluate_response", self._evaluate_response)

        # Define flow
        workflow.set_entry_point("generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        workflow.add_edge("validate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "generate_insights")
        workflow.add_edge("generate_insights", "evaluate_response")
        workflow.add_edge("evaluate_response", END)

        self._graph = workflow.compile()
        return self._graph  # type: ignore

    async def process_query(
        self, query: str, company_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process an analytics query using the LangGraph workflow."""
        import asyncio

        start_time = asyncio.get_running_loop().time()

        logger.info("Processing Analytics query", query=query[:100])

        # Initialize state
        initial_state = AgentState(
            query=query,
            company_id=company_id,
            context="",
            sql_query=None,
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        # Execute workflow
        graph = self.create_graph()
        result = await graph.ainvoke(initial_state)

        # Calculate response time
        result["response_time_ms"] = int(
            (asyncio.get_running_loop().time() - start_time) * 1000
        )

        logger.info(
            "Analytics query processed", response_time_ms=result["response_time_ms"]
        )
        return result

    async def _generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL from natural language query."""
        try:
            async with get_db_session_context() as session:
                # Check if any schema tables are registered before attempting SQL
                from ..core.models import SchemaTable
                from sqlalchemy import select, func

                count_result = await session.execute(
                    select(func.count()).select_from(SchemaTable).where(
                        SchemaTable.company_id == state["company_id"]
                        if state["company_id"]
                        else True
                    )
                )
                schema_count = count_result.scalar()

                if not schema_count:
                    logger.warning(
                        "No schema registered for SQL generation — skipping SQL",
                        company_id=state["company_id"],
                    )
                    state["sql_query"] = None
                    state["context"] = (
                        "No database schema registered for this company. "
                        "Register table schemas via POST /api/schema/tables to enable NL-to-SQL. "
                        "Answering from documents instead."
                    )
                    return state

                sql_result = await self.text_to_sql.generate_sql(
                    state["query"], state["company_id"], session
                )

            state["sql_query"] = sql_result.get("sql")
            state["context"] = f"Generated SQL: {state['sql_query']}"

            logger.info(
                "SQL generated for Analytics",
                sql=state["sql_query"][:100] if state["sql_query"] else None,
            )

        except Exception as e:
            logger.error("SQL generation failed in Analytics", error=str(e))
            state["sql_query"] = None
            state["context"] = "SQL generation failed."

        return state

    async def _validate_sql(self, state: AgentState) -> AgentState:
        """Validate the generated SQL."""
        if not state["sql_query"]:
            return state

        try:
            validation = await self.llm.validate_sql(
                state["sql_query"],
                state["query"],
                "",  # Could add schema context here
            )

            if not validation.get("is_valid", False):
                logger.warning(
                    "SQL validation failed", issues=validation.get("issues", [])
                )
                # Could attempt to improve SQL here

        except Exception as e:
            logger.error("SQL validation failed in Analytics", error=str(e))

        return state

    async def _execute_sql(self, state: AgentState) -> AgentState:
        """Return the generated SQL. Execution runs only when a business_db_url is configured."""
        if not state["sql_query"]:
            state["sql_result"] = {"status": "no_sql", "data": []}
            return state

        from ..core.config import settings

        business_db_url = settings.business_db_url
        if not business_db_url:
            # SQL generated but no separate business DB configured — return SQL for user to run
            state["sql_result"] = {
                "status": "sql_ready",
                "message": (
                    "SQL generated successfully. Configure BUSINESS_DB_URL env var "
                    "to enable automatic execution against your business database."
                ),
                "data": [],
                "row_count": 0,
            }
            state["context"] += f"\nGenerated SQL (ready to run): {state['sql_query']}"
            logger.info("SQL ready (no business DB configured for execution)")
            return state

        try:
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker

            engine = create_async_engine(business_db_url, pool_pre_ping=True)
            async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

            async with async_session() as session:
                sql_result = await self.text_to_sql.execute_sql(state["sql_query"], session)

            await engine.dispose()
            state["sql_result"] = sql_result
            state["context"] += f"\nSQL Result: {sql_result}"
            logger.info("SQL executed against business DB", row_count=sql_result.get("row_count", 0))

        except Exception as e:
            logger.error("SQL execution failed", error=str(e))
            # DB unreachable — treat as sql_ready so insights are still clean
            state["sql_result"] = {
                "status": "sql_ready",
                "message": "Business database could not be reached. SQL was generated successfully.",
                "data": [],
                "row_count": 0,
            }

        return state

    async def _generate_insights(self, state: AgentState) -> AgentState:
        """Generate insights from SQL results."""
        try:
            sql_result = state["sql_result"] or {}
            sql_status = sql_result.get("status", "")
            has_data = bool(sql_result.get("data"))

            # SQL was never generated — explain why and answer from context
            if not state["sql_query"]:
                reason = state.get("context", "") or "No SQL could be generated for this query."
                answer = await self.llm.generate_answer(
                    state["query"],
                    reason,
                    max_tokens=400,
                    temperature=0.3,
                )
                state["answer"] = answer
                state["tokens_used"] = 0
                logger.info("No SQL generated; answering from context")
                return state

            # SQL generated but no data returned yet — show SQL with instructions
            if not has_data and sql_status in ("sql_ready", "no_sql", ""):
                sql = state["sql_query"]
                msg = sql_result.get(
                    "message",
                    "Configure the BUSINESS_DB_URL environment variable to execute this query automatically.",
                )
                state["answer"] = (
                    f"Here is the SQL query generated for your request:\n\n"
                    f"```sql\n{sql}\n```\n\n"
                    f"{msg}"
                )
                state["tokens_used"] = 0
                logger.info("Returning sql_ready answer (no data to analyze)")
                return state

            # Data available — ask LLM to analyze it
            analysis_context = (
                f"Query: {state['query']}\n"
                f"Generated SQL: {state['sql_query'] or 'N/A'}\n"
                f"SQL Results: {sql_result}\n\n"
                f"Provide a concise, direct answer based on the data above."
            )

            answer = await self.llm.generate_answer(
                f"Answer this question using the SQL results: {state['query']}",
                analysis_context,
                max_tokens=500,
                temperature=0.2,
            )

            state["answer"] = answer
            state["tokens_used"] = await self.llm.estimate_tokens(
                state["query"] + analysis_context + answer
            )

            logger.info("Insights generated for Analytics", answer_length=len(answer))

        except Exception as e:
            logger.error("Insights generation failed", error=str(e))
            state["answer"] = (
                "I apologize, but I was unable to analyze the data at this time."
            )

        return state

    async def _evaluate_response(self, state: AgentState) -> AgentState:
        """Evaluate the Analytics response."""
        try:
            # Create a mock response object for evaluation
            mock_response = type(
                "Response",
                (),
                {
                    "answer": state["answer"],
                    "response_time_ms": state["response_time_ms"],
                    "documents": [],  # Analytics doesn't use documents
                    "sql_response": (
                        type(
                            "SQLResponse",
                            (),
                            {
                                "sql": state["sql_query"] or "",
                                "confidence": 0.8,  # Mock confidence
                                "is_valid": state["sql_query"] is not None,
                            },
                        )()
                        if state["sql_query"]
                        else None
                    ),
                },
            )()

            metrics = await self.evaluation.evaluate_query(
                state["query"],
                mock_response,
                sql_result=(
                    {"sql": state["sql_query"], "is_valid": True}
                    if state["sql_query"]
                    else None
                ),
                retrieved_docs=[],
            )

            state["evaluation_metrics"] = metrics

            # Log to W&B
            if self.wandb.enabled:
                await self.wandb.log_query_evaluation(
                    state["query"], mock_response, metrics, state["response_time_ms"]
                )

        except Exception as e:
            logger.error("Evaluation failed in Analytics", error=str(e))

        return state


class AgentOrchestrator:
    """Orchestrates Q&A and Analytics agents based on query type."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.qa_agent = QAAgent()
        self.analytics_agent = AnalyticsAgent()

    def classify_query(self, query: str) -> str:
        """
        Classify query type: 'qa' for general questions, 'analytics' for data analysis.
        This is a simple heuristic - could be enhanced with ML classification.
        """
        query_lower = query.lower()

        # Analytics keywords
        analytics_keywords = [
            "how many",
            "count",
            "sum",
            "average",
            "total",
            "group by",
            "sales",
            "revenue",
            "profit",
            "performance",
            "metrics",
            "trend",
            "analysis",
            "report",
            "statistics",
            "data",
        ]

        # Q&A keywords
        qa_keywords = [
            "what is",
            "who",
            "when",
            "where",
            "why",
            "how to",
            "explain",
            "describe",
            "tell me about",
            "what are",
        ]

        analytics_score = sum(
            1 for keyword in analytics_keywords if keyword in query_lower
        )
        qa_score = sum(1 for keyword in qa_keywords if keyword in query_lower)

        # SQL-like patterns strongly indicate analytics
        sql_patterns = [
            "select",
            "from",
            "join",
            "group by",
            "order by",
            "having",
            "limit",
        ]
        sql_pattern_count = sum(1 for pattern in sql_patterns if pattern in query_lower)
        if (
            sql_pattern_count >= 1
        ):  # Require at least 1 SQL pattern to classify as analytics
            analytics_score += 3

        # Analytics keywords get slight preference in ties
        if analytics_score > qa_score:
            return "analytics"
        elif qa_score > analytics_score:
            return "qa"
        else:
            # Tie: prefer analytics for data-related queries
            return (
                "analytics"
                if any(
                    kw in query_lower
                    for kw in ["statistics", "data", "metrics", "analysis"]
                )
                else "qa"
            )

    async def process_query(
        self,
        query: str,
        company_id: Optional[int] = None,
        force_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a query by routing to the appropriate agent.

        Args:
            query: The natural language query
            company_id: Optional company ID for context
            force_agent: Force use of specific agent ('qa' or 'analytics')

        Returns:
            Processed query result
        """
        # Classify query type
        agent_type = force_agent or self.classify_query(query)

        logger.info("Routing query to agent", query=query[:50], agent=agent_type)

        if agent_type == "analytics":
            result = await self.analytics_agent.process_query(query, company_id)
        else:
            result = await self.qa_agent.process_query(query, company_id)

        result["agent_used"] = agent_type
        return result
