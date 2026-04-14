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
    context: str
    sql_query: Optional[str]
    sql_result: Optional[Dict[str, Any]]
    documents: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
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
        self.evaluation = EvaluationService(llm=self.llm)
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
        self, query: str
    ) -> Dict[str, Any]:
        """Process a Q&A query using the LangGraph workflow."""
        import asyncio

        start_time = asyncio.get_running_loop().time()

        logger.info("Processing Q&A query", query=query[:100])

        # Initialize state
        initial_state = AgentState(
            query=query,
            context="",
            sql_query=None,
            sql_result=None,
            documents=[],
            citations=[],
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
            
            documents = await self.retrieval.retrieve_documents(state["query"], top_k=5)

            # Build context from documents
            context_parts = []
            for doc in documents[:3]:  # Use top 3 documents
                content = doc.get("content") or "No content available."
                context_parts.append(f"Content: {content[:1000]}...") 
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
        """Generate answer with inline citations from retrieved documents."""
        try:
            result = await self.llm.generate_answer_with_citations(
                state["query"],
                state["documents"][:3],  # top 3 docs passed as numbered sources
                max_tokens=500,
                temperature=0.7,
            )

            answer = result["answer"]
            cited_indices = result["cited_indices"]

            # Build citation objects from cited doc indices (1-based)
            citations = []
            for idx in cited_indices:
                doc_index = idx - 1  # convert to 0-based
                if 0 <= doc_index < len(state["documents"]):
                    doc = state["documents"][doc_index]
                    metadata = doc.get("metadata") or {}
                    citations.append(
                        {
                            "index": idx,
                            "source": metadata.get("source")
                            or metadata.get("filename")
                            or f"Document {idx}",
                            "excerpt": (doc.get("content") or "")[:200],
                        }
                    )

            state["answer"] = answer
            state["citations"] = citations
            state["tokens_used"] = await self.llm.estimate_tokens(
                state["query"] + answer
            )

            logger.info(
                "Answer generated for Q&A",
                answer_length=len(answer),
                citation_count=len(citations),
            )

        except Exception as e:
            logger.error("Answer generation failed in Q&A", error=str(e))
            state["answer"] = (
                "I apologize, but I was unable to generate an answer at this time."
            )
            state["citations"] = []

        return state

    async def _evaluate_response(self, state: AgentState) -> AgentState:
        """Evaluate the Q&A response."""
        try:
            mock_response = type(
                "Response",
                (),
                {
                    "answer": state["answer"],
                    "response_time_ms": state["response_time_ms"],
                    "documents": state["documents"],
                    "sql_response": None,
                },
            )()

            cited_indices = [c["index"] for c in state.get("citations", [])]

            metrics = await self.evaluation.evaluate_query(
                state["query"],
                mock_response,
                sql_result=None,
                retrieved_docs=state["documents"],
                cited_indices=cited_indices,
            )

            state["evaluation_metrics"] = metrics

            if self.wandb.enabled:
                await self.wandb.log_query_evaluation(
                    state["query"],
                    mock_response,
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
        self, query: str
    ) -> Dict[str, Any]:
        """Process an analytics query using the LangGraph workflow."""
        import asyncio

        start_time = asyncio.get_running_loop().time()

        logger.info("Processing Analytics query", query=query[:100])

        # Initialize state
        initial_state = AgentState(
            query=query,
            context="",
            sql_query=None,
            sql_result=None,
            documents=[],
            citations=[],
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
        """Generate SQL from natural language query using live DB schema."""
        try:
            async with get_db_session_context() as session:
                sql_result = await self.text_to_sql.generate_sql(state["query"], session)

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
        """Execute the generated SQL against the application database."""
        if not state["sql_query"]:
            state["sql_result"] = {"status": "no_sql", "data": []}
            return state

        try:
            async with get_db_session_context() as session:
                sql_result = await self.text_to_sql.execute_sql(state["sql_query"], session)

            state["sql_result"] = sql_result
            state["context"] += f"\nSQL Result: {sql_result}"
            logger.info("SQL executed", row_count=sql_result.get("row_count", 0))

        except Exception as e:
            logger.error("SQL execution failed", error=str(e))
            state["sql_result"] = {"error": str(e), "data": [], "row_count": 0}

        return state

    async def _generate_insights(self, state: AgentState) -> AgentState:
        """Generate insights from SQL results."""
        try:
            sql_result = state["sql_result"] or {}
            has_data = bool(sql_result.get("data"))

            # SQL was never generated
            if not state["sql_query"]:
                answer = await self.llm.generate_answer(
                    state["query"],
                    state.get("context", "") or "No SQL could be generated for this query.",
                    max_tokens=400,
                    temperature=0.3,
                )
                state["answer"] = answer
                state["tokens_used"] = 0
                return state

            # SQL executed but returned an error or no rows
            if not has_data:
                error = sql_result.get("error", "")
                context = (
                    f"SQL query was generated but returned no results.\n"
                    f"SQL: {state['sql_query']}\n"
                    + (f"Error: {error}" if error else "The query returned 0 rows.")
                )
                answer = await self.llm.generate_answer(
                    state["query"], context, max_tokens=300, temperature=0.3
                )
                state["answer"] = answer
                state["tokens_used"] = 0
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
        force_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a query by routing to the appropriate agent.

        Args:
            query: The natural language query
            force_agent: Force use of specific agent ('qa' or 'analytics')

        Returns:
            Processed query result
        """
        # Classify query type
        agent_type = force_agent or self.classify_query(query)

        logger.info("Routing query to agent", query=query[:50], agent=agent_type)

        if agent_type == "analytics":
            result = await self.analytics_agent.process_query(query)
        else:
            result = await self.qa_agent.process_query(query)

        result["agent_used"] = agent_type
        return result
