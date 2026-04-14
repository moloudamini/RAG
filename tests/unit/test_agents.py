"""Unit tests for LangGraph agent system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.orchestrator import (
    AgentOrchestrator,
    QAAgent,
    AnalyticsAgent,
    AgentState,
)


def make_qa_state(**overrides) -> AgentState:
    """Build a minimal AgentState for Q&A tests."""
    defaults = AgentState(
        query="What is our mission?",
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
    defaults.update(overrides)
    return defaults


def make_analytics_state(**overrides) -> AgentState:
    """Build a minimal AgentState for analytics tests."""
    defaults = AgentState(
        query="How many sales?",
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
    defaults.update(overrides)
    return defaults


class TestAgentOrchestrator:
    """Test cases for AgentOrchestrator singleton and query classification."""

    def test_orchestrator_initialization(self):
        """Test that AgentOrchestrator initializes correctly."""
        orchestrator = AgentOrchestrator()

        assert orchestrator.qa_agent is not None
        assert orchestrator.analytics_agent is not None

    def test_query_classification_qa(self):
        """Test query classification for Q&A queries."""
        orchestrator = AgentOrchestrator()

        qa_queries = [
            "What is our company mission?",
            "Who is the CEO?",
            "When was the company founded?",
            "Where are our offices located?",
            "Why did we choose this strategy?",
            "How to use the product?",
            "Explain the pricing model",
            "Describe our services",
            "Tell me about the team",
            "What are our values?",
        ]

        for query in qa_queries:
            classification = orchestrator.classify_query(query)
            assert (
                classification == "qa"
            ), f"Query '{query}' should be classified as 'qa'"

    def test_query_classification_analytics(self):
        """Test query classification for analytics queries."""
        orchestrator = AgentOrchestrator()

        analytics_queries = [
            "How many sales did we have last month?",
            "What is the total revenue?",
            "Show me the average profit margin",
            "Count the number of customers",
            "Group sales by region",
            "What are our performance metrics?",
            "Show me the trend in user growth",
            "Generate a sales report",
            "What are the key statistics?",
            "Analyze the data",
        ]

        for query in analytics_queries:
            classification = orchestrator.classify_query(query)
            assert (
                classification == "analytics"
            ), f"Query '{query}' should be classified as 'analytics'"

    def test_query_classification_sql_patterns(self):
        """Test that SQL patterns strongly indicate analytics."""
        orchestrator = AgentOrchestrator()

        sql_queries = [
            "SELECT * FROM users",
            "Show me data from the sales table",
            "Query the database for orders",
            "JOIN customers with orders",
            "GROUP BY product category",
            "ORDER BY date descending",
        ]

        for query in sql_queries:
            classification = orchestrator.classify_query(query)
            assert (
                classification == "analytics"
            ), f"SQL query '{query}' should be classified as 'analytics'"

    @pytest.mark.asyncio
    @patch("src.agents.orchestrator.QAAgent")
    @patch("src.agents.orchestrator.AnalyticsAgent")
    async def test_process_query_routes_to_qa(
        self, mock_analytics_agent, mock_qa_agent
    ):
        """Test that process_query routes Q&A queries to QA agent."""
        mock_qa_instance = MagicMock()
        mock_qa_instance.process_query = AsyncMock(
            return_value={"answer": "QA response"}
        )
        mock_qa_agent.return_value = mock_qa_instance

        mock_analytics_instance = MagicMock()
        mock_analytics_agent.return_value = mock_analytics_instance

        orchestrator = AgentOrchestrator()

        result = await orchestrator.process_query("What is our mission?")

        mock_qa_instance.process_query.assert_called_once_with("What is our mission?")
        assert result["answer"] == "QA response"
        assert result["agent_used"] == "qa"

    @pytest.mark.asyncio
    @patch("src.agents.orchestrator.QAAgent")
    @patch("src.agents.orchestrator.AnalyticsAgent")
    async def test_process_query_routes_to_analytics(
        self, mock_analytics_agent, mock_qa_agent
    ):
        """Test that process_query routes analytics queries to analytics agent."""
        mock_qa_instance = MagicMock()
        mock_qa_agent.return_value = mock_qa_instance

        mock_analytics_instance = MagicMock()
        mock_analytics_instance.process_query = AsyncMock(
            return_value={"sql_result": "Analytics response"}
        )
        mock_analytics_agent.return_value = mock_analytics_instance

        orchestrator = AgentOrchestrator()

        result = await orchestrator.process_query("How many sales last month?")

        mock_analytics_instance.process_query.assert_called_once_with(
            "How many sales last month?"
        )
        assert result["sql_result"] == "Analytics response"
        assert result["agent_used"] == "analytics"

    @pytest.mark.asyncio
    @patch("src.agents.orchestrator.QAAgent")
    @patch("src.agents.orchestrator.AnalyticsAgent")
    async def test_process_query_force_agent(self, mock_analytics_agent, mock_qa_agent):
        """Test that force_agent parameter overrides classification."""
        mock_qa_instance = MagicMock()
        mock_qa_instance.process_query = AsyncMock(
            return_value={"answer": "QA response"}
        )
        mock_qa_agent.return_value = mock_qa_instance

        mock_analytics_instance = MagicMock()
        mock_analytics_instance.process_query = AsyncMock(
            return_value={"answer": "Analytics response"}
        )
        mock_analytics_agent.return_value = mock_analytics_instance

        orchestrator = AgentOrchestrator()

        result = await orchestrator.process_query(
            "What is our mission?", force_agent="analytics"
        )

        mock_qa_instance.process_query.assert_not_called()
        assert result["agent_used"] == "analytics"


class TestQAAgent:
    """Test cases for QAAgent functionality."""

    @pytest.fixture
    def qa_agent(self):
        """Create a QA agent instance with mocked dependencies."""
        with (
            patch("src.agents.orchestrator.RetrievalService") as mock_retrieval,
            patch("src.agents.orchestrator.LLMService") as mock_llm,
            patch("src.agents.orchestrator.EvaluationService") as mock_evaluation,
            patch("src.agents.orchestrator.WandbService") as mock_wandb,
        ):
            agent = QAAgent()
            agent.retrieval = mock_retrieval()
            agent.llm = mock_llm()
            agent.evaluation = mock_evaluation()
            agent.wandb = mock_wandb()
            agent.wandb.enabled = False
            return agent

    def test_create_graph_caching(self, qa_agent):
        """Test that create_graph caches the compiled graph."""
        graph1 = qa_agent.create_graph()
        graph2 = qa_agent.create_graph()

        assert graph1 is graph2
        assert qa_agent._graph is not None

    def test_create_graph_structure(self, qa_agent):
        """Test that the graph has the correct nodes and edges."""
        graph = qa_agent.create_graph()

        assert hasattr(graph, "nodes")
        expected_nodes = ["retrieve_documents", "generate_answer", "evaluate_response"]
        for node in expected_nodes:
            assert node in graph.nodes

    @pytest.mark.asyncio
    async def test_process_query_full_workflow(self, qa_agent):
        """Test the complete Q&A workflow execution."""
        qa_agent.retrieval.retrieve_documents = AsyncMock(
            return_value=[
                {"title": "Doc1", "content": "Content1", "metadata": {}},
                {"title": "Doc2", "content": "Content2", "metadata": {}},
            ]
        )
        qa_agent.llm.generate_answer_with_citations = AsyncMock(
            return_value={"answer": "Generated answer", "cited_indices": [1]}
        )
        qa_agent.llm.estimate_tokens = AsyncMock(return_value=150)
        qa_agent.evaluation.evaluate_query = AsyncMock(return_value={"accuracy": 0.9})

        result = await qa_agent.process_query("What is our mission?")

        assert "query" in result
        assert "answer" in result
        assert "documents" in result
        assert "citations" in result
        assert "response_time_ms" in result
        assert "tokens_used" in result
        assert "evaluation_metrics" in result

        assert result["query"] == "What is our mission?"
        assert result["answer"] == "Generated answer"
        assert len(result["documents"]) == 2
        assert result["tokens_used"] == 150
        assert result["evaluation_metrics"] == {"accuracy": 0.9}

    @pytest.mark.asyncio
    async def test_retrieve_documents_success(self, qa_agent):
        """Test successful document retrieval."""
        mock_docs = [
            {"title": "Mission Statement", "content": "Our mission is to innovate"},
            {"title": "About Us", "content": "We are a company that cares"},
        ]
        qa_agent.retrieval.retrieve_documents = AsyncMock(return_value=mock_docs)

        state = make_qa_state()
        result_state = await qa_agent._retrieve_documents(state)

        assert len(result_state["documents"]) == 2
        assert "Our mission is to innovate" in result_state["context"]

    @pytest.mark.asyncio
    async def test_retrieve_documents_failure(self, qa_agent):
        """Test document retrieval failure handling."""
        qa_agent.retrieval.retrieve_documents = AsyncMock(
            side_effect=Exception("Retrieval failed")
        )

        state = make_qa_state()
        result_state = await qa_agent._retrieve_documents(state)

        assert result_state["documents"] == []
        assert result_state["context"] == "No relevant documents found."

    @pytest.mark.asyncio
    async def test_generate_answer_success(self, qa_agent):
        """Test successful answer generation with citations."""
        qa_agent.llm.generate_answer_with_citations = AsyncMock(
            return_value={"answer": "This is the answer [1]", "cited_indices": [1]}
        )
        qa_agent.llm.estimate_tokens = AsyncMock(return_value=200)

        state = make_qa_state(
            context="Mission: To innovate...",
            documents=[{"content": "Mission content", "metadata": {"source": "doc1.pdf"}}],
        )
        result_state = await qa_agent._generate_answer(state)

        assert result_state["answer"] == "This is the answer [1]"
        assert result_state["tokens_used"] == 200
        assert len(result_state["citations"]) == 1
        assert result_state["citations"][0]["index"] == 1

    @pytest.mark.asyncio
    async def test_generate_answer_failure(self, qa_agent):
        """Test answer generation failure handling."""
        qa_agent.llm.generate_answer_with_citations = AsyncMock(
            side_effect=Exception("LLM failed")
        )

        state = make_qa_state(context="Mission: To innovate...")
        result_state = await qa_agent._generate_answer(state)

        assert "apologize" in result_state["answer"].lower()
        assert result_state["citations"] == []

    @pytest.mark.asyncio
    async def test_evaluate_response_success(self, qa_agent):
        """Test successful response evaluation."""
        qa_agent.evaluation.evaluate_query = AsyncMock(
            return_value={"accuracy": 0.85, "faithfulness": 0.9}
        )

        state = make_qa_state(
            context="Mission context",
            documents=[{"title": "Doc1", "content": "Content1"}],
            answer="Generated answer",
            response_time_ms=150,
            tokens_used=100,
        )
        result_state = await qa_agent._evaluate_response(state)

        assert result_state["evaluation_metrics"] == {
            "accuracy": 0.85,
            "faithfulness": 0.9,
        }


class TestAnalyticsAgent:
    """Test cases for AnalyticsAgent functionality."""

    @pytest.fixture
    def analytics_agent(self):
        """Create an Analytics agent instance with mocked dependencies."""
        with (
            patch("src.agents.orchestrator.TextToSQLService") as mock_text_to_sql,
            patch("src.agents.orchestrator.LLMService") as mock_llm,
            patch("src.agents.orchestrator.EvaluationService") as mock_evaluation,
            patch("src.agents.orchestrator.WandbService") as mock_wandb,
        ):
            agent = AnalyticsAgent()
            agent.text_to_sql = mock_text_to_sql()
            agent.llm = mock_llm()
            agent.evaluation = mock_evaluation()
            agent.wandb = mock_wandb()
            agent.wandb.enabled = False
            return agent

    def test_create_graph_caching(self, analytics_agent):
        """Test that create_graph caches the compiled graph."""
        graph1 = analytics_agent.create_graph()
        graph2 = analytics_agent.create_graph()

        assert graph1 is graph2
        assert analytics_agent._graph is not None

    def test_create_graph_structure(self, analytics_agent):
        """Test that the analytics graph has the correct nodes and edges."""
        graph = analytics_agent.create_graph()

        expected_nodes = [
            "generate_sql",
            "validate_sql",
            "execute_sql",
            "generate_insights",
            "evaluate_response",
        ]
        for node in expected_nodes:
            assert node in graph.nodes

    @pytest.mark.asyncio
    async def test_process_query_full_workflow(self, analytics_agent):
        """Test the complete analytics workflow execution."""
        with patch("src.agents.orchestrator.get_db_session_context") as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db.return_value.__aexit__ = AsyncMock(return_value=None)

            analytics_agent.text_to_sql.generate_sql = AsyncMock(
                return_value={"sql": "SELECT * FROM sales"}
            )
            analytics_agent.text_to_sql.execute_sql = AsyncMock(
                return_value={"data": [{"count": 100}], "row_count": 1}
            )
            analytics_agent.llm.validate_sql = AsyncMock(return_value={"is_valid": True})
            analytics_agent.llm.generate_answer = AsyncMock(
                return_value="Analytics insights"
            )
            analytics_agent.llm.estimate_tokens = AsyncMock(return_value=250)
            analytics_agent.evaluation.evaluate_query = AsyncMock(
                return_value={"accuracy": 0.8}
            )

            result = await analytics_agent.process_query("How many sales last month?")

        assert "query" in result
        assert "sql_query" in result
        assert "sql_result" in result
        assert "answer" in result
        assert "response_time_ms" in result
        assert "tokens_used" in result
        assert "evaluation_metrics" in result

        assert result["query"] == "How many sales last month?"
        assert result["sql_query"] == "SELECT * FROM sales"
        assert result["answer"] == "Analytics insights"
        assert result["tokens_used"] == 250

    @pytest.mark.asyncio
    async def test_generate_sql_success(self, analytics_agent):
        """Test successful SQL generation."""
        with patch("src.agents.orchestrator.get_db_session_context") as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db.return_value.__aexit__ = AsyncMock(return_value=None)

            analytics_agent.text_to_sql.generate_sql = AsyncMock(
                return_value={"sql": "SELECT COUNT(*) FROM sales"}
            )

            state = make_analytics_state()
            result_state = await analytics_agent._generate_sql(state)

        assert result_state["sql_query"] == "SELECT COUNT(*) FROM sales"
        assert "SELECT COUNT(*) FROM sales" in result_state["context"]

    @pytest.mark.asyncio
    async def test_generate_sql_failure(self, analytics_agent):
        """Test SQL generation failure handling."""
        with patch("src.agents.orchestrator.get_db_session_context") as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db.return_value.__aexit__ = AsyncMock(return_value=None)

            analytics_agent.text_to_sql.generate_sql = AsyncMock(
                side_effect=Exception("SQL generation failed")
            )

            state = make_analytics_state()
            result_state = await analytics_agent._generate_sql(state)

        assert result_state["sql_query"] is None
        assert "SQL generation failed" in result_state["context"]

    @pytest.mark.asyncio
    async def test_validate_sql_success(self, analytics_agent):
        """Test successful SQL validation."""
        analytics_agent.llm.validate_sql = AsyncMock(
            return_value={"is_valid": True, "issues": []}
        )

        state = make_analytics_state(sql_query="SELECT COUNT(*) FROM sales")
        result_state = await analytics_agent._validate_sql(state)

        assert result_state["sql_query"] == "SELECT COUNT(*) FROM sales"

    @pytest.mark.asyncio
    async def test_execute_sql_success(self, analytics_agent):
        """Test SQL execution with real DB session mock."""
        with patch("src.agents.orchestrator.get_db_session_context") as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db.return_value.__aexit__ = AsyncMock(return_value=None)

            analytics_agent.text_to_sql.execute_sql = AsyncMock(
                return_value={"data": [{"count": 42}], "row_count": 1}
            )

            state = make_analytics_state(
                context="Generated SQL: SELECT COUNT(*) FROM sales",
                sql_query="SELECT COUNT(*) FROM sales",
            )
            result_state = await analytics_agent._execute_sql(state)

        assert result_state["sql_result"]["row_count"] == 1
        assert "SELECT COUNT(*) FROM sales" in result_state["context"]

    @pytest.mark.asyncio
    async def test_execute_sql_no_query(self, analytics_agent):
        """Test SQL execution when no query is available."""
        state = make_analytics_state(sql_query=None)
        result_state = await analytics_agent._execute_sql(state)

        assert result_state["sql_result"]["status"] == "no_sql"

    @pytest.mark.asyncio
    async def test_generate_insights_success(self, analytics_agent):
        """Test successful insights generation."""
        analytics_agent.llm.generate_answer = AsyncMock(
            return_value="Key insights: Sales are up 20%"
        )
        analytics_agent.llm.estimate_tokens = AsyncMock(return_value=300)

        state = make_analytics_state(
            context="SQL: SELECT COUNT(*) FROM sales",
            sql_query="SELECT COUNT(*) FROM sales",
            sql_result={"status": "success", "data": [{"count": 100}]},
        )
        result_state = await analytics_agent._generate_insights(state)

        assert result_state["answer"] == "Key insights: Sales are up 20%"
        assert result_state["tokens_used"] == 300

    @pytest.mark.asyncio
    async def test_generate_insights_failure(self, analytics_agent):
        """Test insights generation failure handling."""
        analytics_agent.llm.generate_answer = AsyncMock(
            side_effect=Exception("LLM failed")
        )

        state = make_analytics_state(
            context="SQL context",
            sql_query="SELECT COUNT(*) FROM sales",
            sql_result={"data": [100]},
        )
        result_state = await analytics_agent._generate_insights(state)

        assert "apologize" in result_state["answer"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_response_analytics(self, analytics_agent):
        """Test analytics response evaluation."""
        analytics_agent.evaluation.evaluate_query = AsyncMock(
            return_value={"accuracy": 0.75, "sql_validity": 0.9}
        )

        state = make_analytics_state(
            context="Analysis context",
            sql_query="SELECT COUNT(*) FROM sales",
            sql_result={"data": [100]},
            answer="Sales count: 100",
            response_time_ms=200,
            tokens_used=150,
        )
        result_state = await analytics_agent._evaluate_response(state)

        assert result_state["evaluation_metrics"] == {
            "accuracy": 0.75,
            "sql_validity": 0.9,
        }


class TestAgentState:
    """Test cases for AgentState TypedDict."""

    def test_agent_state_creation(self):
        """Test that AgentState can be created with all required fields."""
        state = AgentState(
            query="Test query",
            context="Test context",
            sql_query="SELECT * FROM test",
            sql_result={"status": "success"},
            documents=[{"title": "Doc1", "content": "Content1"}],
            citations=[{"index": 1, "source": "doc1.pdf", "excerpt": "..."}],
            answer="Test answer",
            evaluation_metrics={"accuracy": 0.9},
            response_time_ms=150,
            tokens_used=100,
        )

        assert state["query"] == "Test query"
        assert state["context"] == "Test context"
        assert state["sql_query"] == "SELECT * FROM test"
        assert len(state["documents"]) == 1
        assert len(state["citations"]) == 1
        assert state["answer"] == "Test answer"
        assert state["evaluation_metrics"] == {"accuracy": 0.9}
        assert state["response_time_ms"] == 150
        assert state["tokens_used"] == 100

    def test_agent_state_optional_fields(self):
        """Test that optional fields can be None."""
        state = AgentState(
            query="Test query",
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

        assert state["sql_query"] is None
        assert state["sql_result"] is None
        assert state["citations"] == []


class TestIntegrationWithServices:
    """Test integration points with external services."""

    @pytest.mark.asyncio
    async def test_qa_agent_calls_retrieval_service(self):
        """Test that QAAgent properly calls RetrievalService."""
        with (
            patch("src.agents.orchestrator.RetrievalService") as mock_retrieval_class,
            patch("src.agents.orchestrator.LLMService") as mock_llm_class,
            patch("src.agents.orchestrator.EvaluationService") as mock_eval_class,
            patch("src.agents.orchestrator.WandbService") as mock_wandb_class,
        ):
            mock_retrieval = MagicMock()
            mock_retrieval.retrieve_documents = AsyncMock(return_value=[])
            mock_retrieval_class.return_value = mock_retrieval

            mock_llm = MagicMock()
            mock_llm.generate_answer_with_citations = AsyncMock(
                return_value={"answer": "Answer", "cited_indices": []}
            )
            mock_llm.estimate_tokens = AsyncMock(return_value=50)
            mock_llm_class.return_value = mock_llm

            mock_eval = MagicMock()
            mock_eval.evaluate_query = AsyncMock(return_value={})
            mock_eval_class.return_value = mock_eval

            mock_wandb = MagicMock()
            mock_wandb.enabled = False
            mock_wandb_class.return_value = mock_wandb

            agent = QAAgent()
            await agent.process_query("Test query")

            mock_retrieval.retrieve_documents.assert_called_once_with(
                "Test query", top_k=5
            )

    @pytest.mark.asyncio
    async def test_analytics_agent_calls_text_to_sql_service(self):
        """Test that AnalyticsAgent properly calls TextToSQLService."""
        with (
            patch("src.agents.orchestrator.TextToSQLService") as mock_text_to_sql_class,
            patch("src.agents.orchestrator.LLMService") as mock_llm_class,
            patch("src.agents.orchestrator.EvaluationService") as mock_eval_class,
            patch("src.agents.orchestrator.WandbService") as mock_wandb_class,
            patch("src.agents.orchestrator.get_db_session_context") as mock_db_session,
        ):
            mock_text_to_sql = MagicMock()
            mock_text_to_sql.generate_sql = AsyncMock(return_value={"sql": "SELECT 1"})
            mock_text_to_sql.execute_sql = AsyncMock(
                return_value={"data": [{"val": 1}], "row_count": 1}
            )
            mock_text_to_sql_class.return_value = mock_text_to_sql

            mock_llm = MagicMock()
            mock_llm.validate_sql = AsyncMock(return_value={"is_valid": True})
            mock_llm.generate_answer = AsyncMock(return_value="Insights")
            mock_llm.estimate_tokens = AsyncMock(return_value=50)
            mock_llm_class.return_value = mock_llm

            mock_eval = MagicMock()
            mock_eval.evaluate_query = AsyncMock(return_value={})
            mock_eval_class.return_value = mock_eval

            mock_wandb = MagicMock()
            mock_wandb.enabled = False
            mock_wandb_class.return_value = mock_wandb

            mock_session = MagicMock()
            mock_db_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_db_session.return_value.__aexit__ = AsyncMock(return_value=None)

            agent = AnalyticsAgent()
            await agent.process_query("How many users?")

            mock_text_to_sql.generate_sql.assert_called_once()
            call_args = mock_text_to_sql.generate_sql.call_args
            assert call_args[0][0] == "How many users?"

    @pytest.mark.asyncio
    async def test_wandb_logging_when_enabled(self):
        """Test that W&B logging occurs when enabled."""
        with (
            patch("src.agents.orchestrator.RetrievalService") as mock_retrieval_class,
            patch("src.agents.orchestrator.LLMService") as mock_llm_class,
            patch("src.agents.orchestrator.EvaluationService") as mock_eval_class,
            patch("src.agents.orchestrator.WandbService") as mock_wandb_class,
        ):
            mock_retrieval = MagicMock()
            mock_retrieval.retrieve_documents = AsyncMock(return_value=[])
            mock_retrieval_class.return_value = mock_retrieval

            mock_llm = MagicMock()
            mock_llm.generate_answer_with_citations = AsyncMock(
                return_value={"answer": "Answer", "cited_indices": []}
            )
            mock_llm.estimate_tokens = AsyncMock(return_value=50)
            mock_llm_class.return_value = mock_llm

            mock_eval = MagicMock()
            mock_eval.evaluate_query = AsyncMock(return_value={"accuracy": 0.8})
            mock_eval_class.return_value = mock_eval

            mock_wandb = MagicMock()
            mock_wandb.enabled = True
            mock_wandb.log_query_evaluation = AsyncMock()
            mock_wandb_class.return_value = mock_wandb

            agent = QAAgent()
            await agent.process_query("Test query")

            mock_wandb.log_query_evaluation.assert_called_once()


class TestErrorHandling:
    """Test error handling in agent workflows."""

    @pytest.mark.asyncio
    async def test_qa_agent_handles_retrieval_service_error(self):
        """Test that Q&A agent handles retrieval service errors gracefully."""
        with (
            patch("src.agents.orchestrator.RetrievalService") as mock_retrieval_class,
            patch("src.agents.orchestrator.LLMService") as mock_llm_class,
            patch("src.agents.orchestrator.EvaluationService") as mock_eval_class,
            patch("src.agents.orchestrator.WandbService") as mock_wandb_class,
        ):
            mock_retrieval = MagicMock()
            mock_retrieval.retrieve_documents = AsyncMock(
                side_effect=Exception("Network error")
            )
            mock_retrieval_class.return_value = mock_retrieval

            mock_llm = MagicMock()
            mock_llm.generate_answer_with_citations = AsyncMock(
                return_value={"answer": "Fallback answer", "cited_indices": []}
            )
            mock_llm.estimate_tokens = AsyncMock(return_value=30)
            mock_llm_class.return_value = mock_llm

            mock_eval = MagicMock()
            mock_eval.evaluate_query = AsyncMock(return_value={})
            mock_eval_class.return_value = mock_eval

            mock_wandb = MagicMock()
            mock_wandb.enabled = False
            mock_wandb_class.return_value = mock_wandb

            agent = QAAgent()
            result = await agent.process_query("Test query")

            assert "answer" in result
            assert result["documents"] == []

    @pytest.mark.asyncio
    async def test_analytics_agent_handles_sql_generation_error(self):
        """Test that analytics agent handles SQL generation errors gracefully."""
        with (
            patch("src.agents.orchestrator.TextToSQLService") as mock_text_to_sql_class,
            patch("src.agents.orchestrator.LLMService") as mock_llm_class,
            patch("src.agents.orchestrator.EvaluationService") as mock_eval_class,
            patch("src.agents.orchestrator.WandbService") as mock_wandb_class,
            patch("src.agents.orchestrator.get_db_session_context") as mock_db_session,
        ):
            mock_text_to_sql = MagicMock()
            mock_text_to_sql.generate_sql = AsyncMock(
                side_effect=Exception("SQL generation failed")
            )
            mock_text_to_sql_class.return_value = mock_text_to_sql

            mock_llm = MagicMock()
            mock_llm.validate_sql = AsyncMock(return_value={"is_valid": False})
            mock_llm.generate_answer = AsyncMock(return_value="Error insights")
            mock_llm.estimate_tokens = AsyncMock(return_value=40)
            mock_llm_class.return_value = mock_llm

            mock_eval = MagicMock()
            mock_eval.evaluate_query = AsyncMock(return_value={})
            mock_eval_class.return_value = mock_eval

            mock_wandb = MagicMock()
            mock_wandb.enabled = False
            mock_wandb_class.return_value = mock_wandb

            mock_session = MagicMock()
            mock_db_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_db_session.return_value.__aexit__ = AsyncMock(return_value=None)

            agent = AnalyticsAgent()
            result = await agent.process_query("How many users?")

            assert "answer" in result
            assert result["sql_query"] is None

    @pytest.mark.asyncio
    async def test_evaluation_service_error_handling(self):
        """Test that evaluation errors don't break the workflow."""
        with (
            patch("src.agents.orchestrator.RetrievalService") as mock_retrieval_class,
            patch("src.agents.orchestrator.LLMService") as mock_llm_class,
            patch("src.agents.orchestrator.EvaluationService") as mock_eval_class,
            patch("src.agents.orchestrator.WandbService") as mock_wandb_class,
        ):
            mock_retrieval = MagicMock()
            mock_retrieval.retrieve_documents = AsyncMock(return_value=[])
            mock_retrieval_class.return_value = mock_retrieval

            mock_llm = MagicMock()
            mock_llm.generate_answer_with_citations = AsyncMock(
                return_value={"answer": "Answer", "cited_indices": []}
            )
            mock_llm.estimate_tokens = AsyncMock(return_value=50)
            mock_llm_class.return_value = mock_llm

            mock_eval = MagicMock()
            mock_eval.evaluate_query = AsyncMock(
                side_effect=Exception("Evaluation failed")
            )
            mock_eval_class.return_value = mock_eval

            mock_wandb = MagicMock()
            mock_wandb.enabled = False
            mock_wandb_class.return_value = mock_wandb

            agent = QAAgent()
            result = await agent.process_query("Test query")

            assert "answer" in result
            assert result["evaluation_metrics"] == {}
