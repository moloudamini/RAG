"""Unit tests for LangGraph agent system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.orchestrator import (
    AgentOrchestrator,
    QAAgent,
    AnalyticsAgent,
    AgentState,
)


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
        # Setup mocks
        mock_qa_instance = MagicMock()
        mock_qa_instance.process_query = AsyncMock(
            return_value={"answer": "QA response"}
        )
        mock_qa_agent.return_value = mock_qa_instance

        mock_analytics_instance = MagicMock()
        mock_analytics_agent.return_value = mock_analytics_instance

        orchestrator = AgentOrchestrator()

        result = await orchestrator.process_query("What is our mission?")

        mock_qa_instance.process_query.assert_called_once_with(
            "What is our mission?", None
        )
        assert result["answer"] == "QA response"
        assert result["agent_used"] == "qa"

    @pytest.mark.asyncio
    @patch("src.agents.orchestrator.QAAgent")
    @patch("src.agents.orchestrator.AnalyticsAgent")
    async def test_process_query_routes_to_analytics(
        self, mock_analytics_agent, mock_qa_agent
    ):
        """Test that process_query routes analytics queries to analytics agent."""
        # Setup mocks
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
            "How many sales last month?", None
        )
        assert result["sql_result"] == "Analytics response"
        assert result["agent_used"] == "analytics"

    @pytest.mark.asyncio
    @patch("src.agents.orchestrator.QAAgent")
    @patch("src.agents.orchestrator.AnalyticsAgent")
    async def test_process_query_force_agent(self, mock_analytics_agent, mock_qa_agent):
        """Test that force_agent parameter overrides classification."""
        # Setup mocks
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

        # Force analytics agent for a Q&A query
        result = await orchestrator.process_query(
            "What is our mission?", force_agent="analytics"
        )

        mock_qa_instance.process_query.assert_not_called()
        assert result["agent_used"] == "analytics"

    @pytest.mark.asyncio
    @patch("src.agents.orchestrator.QAAgent")
    @patch("src.agents.orchestrator.AnalyticsAgent")
    async def test_process_query_with_company_id(
        self, mock_analytics_agent, mock_qa_agent
    ):
        """Test that company_id is passed to agents."""
        mock_qa_instance = MagicMock()
        mock_qa_instance.process_query = AsyncMock(return_value={"answer": "Response"})
        mock_qa_agent.return_value = mock_qa_instance

        mock_analytics_instance = MagicMock()
        mock_analytics_agent.return_value = mock_analytics_instance

        orchestrator = AgentOrchestrator()

        await orchestrator.process_query("What is our mission?", company_id=123)

        mock_qa_instance.process_query.assert_called_once_with(
            "What is our mission?", 123
        )


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
            agent.wandb.enabled = False  # Disable W&B for testing
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

        # Check that graph has expected nodes
        assert hasattr(graph, "nodes")
        # The graph should have the expected nodes
        expected_nodes = ["retrieve_documents", "generate_answer", "evaluate_response"]
        for node in expected_nodes:
            assert node in graph.nodes

    @pytest.mark.asyncio
    async def test_process_query_full_workflow(self, qa_agent):
        """Test the complete Q&A workflow execution."""
        # Setup mocks
        qa_agent.retrieval.retrieve_documents = AsyncMock(
            return_value=[
                {"title": "Doc1", "content": "Content1"},
                {"title": "Doc2", "content": "Content2"},
            ]
        )
        qa_agent.llm.generate_answer = AsyncMock(return_value="Generated answer")
        qa_agent.llm.estimate_tokens = AsyncMock(return_value=150)
        qa_agent.evaluation.evaluate_query = AsyncMock(return_value={"accuracy": 0.9})

        result = await qa_agent.process_query("What is our mission?", company_id=123)

        # Verify result structure
        assert "query" in result
        assert "company_id" in result
        assert "answer" in result
        assert "documents" in result
        assert "response_time_ms" in result
        assert "tokens_used" in result
        assert "evaluation_metrics" in result

        # Verify values
        assert result["query"] == "What is our mission?"
        assert result["company_id"] == 123
        assert result["answer"] == "Generated answer"
        assert len(result["documents"]) == 2
        assert result["tokens_used"] == 150
        assert result["evaluation_metrics"] == {"accuracy": 0.9}

    @pytest.mark.asyncio
    async def test_retrieve_documents_success(self, qa_agent):
        """Test successful document retrieval."""
        mock_docs = [
            {"title": "Mission Statement", "content": "Our mission is to..."},
            {"title": "About Us", "content": "We are a company that..."},
        ]
        qa_agent.retrieval.retrieve_documents = AsyncMock(return_value=mock_docs)

        state = AgentState(
            query="What is our mission?",
            company_id=123,
            context="",
            sql_query=None,
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        result_state = await qa_agent._retrieve_documents(state)

        assert len(result_state["documents"]) == 2
        assert "Mission Statement" in result_state["context"]
        assert "Our mission is to..." in result_state["context"]

    @pytest.mark.asyncio
    async def test_retrieve_documents_failure(self, qa_agent):
        """Test document retrieval failure handling."""
        qa_agent.retrieval.retrieve_documents = AsyncMock(
            side_effect=Exception("Retrieval failed")
        )

        state = AgentState(
            query="What is our mission?",
            company_id=123,
            context="",
            sql_query=None,
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        result_state = await qa_agent._retrieve_documents(state)

        assert result_state["documents"] == []
        assert result_state["context"] == "No relevant documents found."

    @pytest.mark.asyncio
    async def test_generate_answer_success(self, qa_agent):
        """Test successful answer generation."""
        qa_agent.llm.generate_answer = AsyncMock(return_value="This is the answer")
        qa_agent.llm.estimate_tokens = AsyncMock(return_value=200)

        state = AgentState(
            query="What is our mission?",
            company_id=123,
            context="Mission: To innovate...",
            sql_query=None,
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        result_state = await qa_agent._generate_answer(state)

        assert result_state["answer"] == "This is the answer"
        assert result_state["tokens_used"] == 200

    @pytest.mark.asyncio
    async def test_generate_answer_failure(self, qa_agent):
        """Test answer generation failure handling."""
        qa_agent.llm.generate_answer = AsyncMock(side_effect=Exception("LLM failed"))

        state = AgentState(
            query="What is our mission?",
            company_id=123,
            context="Mission: To innovate...",
            sql_query=None,
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        result_state = await qa_agent._generate_answer(state)

        assert "apologize" in result_state["answer"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_response_success(self, qa_agent):
        """Test successful response evaluation."""
        qa_agent.evaluation.evaluate_query = AsyncMock(
            return_value={"accuracy": 0.85, "relevance": 0.9}
        )

        state = AgentState(
            query="What is our mission?",
            company_id=123,
            context="Mission context",
            sql_query=None,
            sql_result=None,
            documents=[{"title": "Doc1", "content": "Content1"}],
            answer="Generated answer",
            evaluation_metrics={},
            response_time_ms=150,
            tokens_used=100,
        )

        result_state = await qa_agent._evaluate_response(state)

        assert result_state["evaluation_metrics"] == {
            "accuracy": 0.85,
            "relevance": 0.9,
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
            agent.wandb.enabled = False  # Disable W&B for testing
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

        # Check that graph has expected nodes
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
        # Setup mocks
        analytics_agent.text_to_sql.generate_sql = AsyncMock(
            return_value={"sql": "SELECT * FROM sales"}
        )
        analytics_agent.llm.validate_sql = AsyncMock(return_value={"is_valid": True})
        analytics_agent.llm.generate_answer = AsyncMock(
            return_value="Analytics insights"
        )
        analytics_agent.llm.estimate_tokens = AsyncMock(return_value=250)
        analytics_agent.evaluation.evaluate_query = AsyncMock(
            return_value={"accuracy": 0.8}
        )

        result = await analytics_agent.process_query(
            "How many sales last month?", company_id=123
        )

        # Verify result structure
        assert "query" in result
        assert "company_id" in result
        assert "sql_query" in result
        assert "sql_result" in result
        assert "answer" in result
        assert "response_time_ms" in result
        assert "tokens_used" in result
        assert "evaluation_metrics" in result

        # Verify values
        assert result["query"] == "How many sales last month?"
        assert result["company_id"] == 123
        assert result["sql_query"] == "SELECT * FROM sales"
        assert result["answer"] == "Analytics insights"
        assert result["tokens_used"] == 250

    @pytest.mark.asyncio
    async def test_generate_sql_success(self, analytics_agent):
        """Test successful SQL generation."""
        analytics_agent.text_to_sql.generate_sql = AsyncMock(
            return_value={"sql": "SELECT COUNT(*) FROM sales"}
        )

        state = AgentState(
            query="How many sales?",
            company_id=123,
            context="",
            sql_query=None,
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        result_state = await analytics_agent._generate_sql(state)

        assert result_state["sql_query"] == "SELECT COUNT(*) FROM sales"
        assert "SELECT COUNT(*) FROM sales" in result_state["context"]

    @pytest.mark.asyncio
    async def test_generate_sql_failure(self, analytics_agent):
        """Test SQL generation failure handling."""
        analytics_agent.text_to_sql.generate_sql = AsyncMock(
            side_effect=Exception("SQL generation failed")
        )

        state = AgentState(
            query="How many sales?",
            company_id=123,
            context="",
            sql_query=None,
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        result_state = await analytics_agent._generate_sql(state)

        assert result_state["sql_query"] is None
        assert "SQL generation failed" in result_state["context"]

    @pytest.mark.asyncio
    async def test_validate_sql_success(self, analytics_agent):
        """Test successful SQL validation."""
        analytics_agent.llm.validate_sql = AsyncMock(
            return_value={"is_valid": True, "issues": []}
        )

        state = AgentState(
            query="How many sales?",
            company_id=123,
            context="",
            sql_query="SELECT COUNT(*) FROM sales",
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        result_state = await analytics_agent._validate_sql(state)

        # State should remain unchanged for successful validation
        assert result_state["sql_query"] == "SELECT COUNT(*) FROM sales"

    @pytest.mark.asyncio
    async def test_execute_sql_simulated(self, analytics_agent):
        """Test SQL execution (simulated)."""
        state = AgentState(
            query="How many sales?",
            company_id=123,
            context="Generated SQL: SELECT COUNT(*) FROM sales",
            sql_query="SELECT COUNT(*) FROM sales",
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        result_state = await analytics_agent._execute_sql(state)

        assert result_state["sql_result"]["status"] == "simulated"
        assert "simulated" in result_state["sql_result"]["message"]
        assert "SELECT COUNT(*) FROM sales" in result_state["context"]

    @pytest.mark.asyncio
    async def test_execute_sql_no_query(self, analytics_agent):
        """Test SQL execution when no query is available."""
        state = AgentState(
            query="How many sales?",
            company_id=123,
            context="",
            sql_query=None,
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        result_state = await analytics_agent._execute_sql(state)

        assert result_state["sql_result"]["error"] == "No SQL query to execute"

    @pytest.mark.asyncio
    async def test_generate_insights_success(self, analytics_agent):
        """Test successful insights generation."""
        analytics_agent.llm.generate_answer = AsyncMock(
            return_value="Key insights: Sales are up 20%"
        )
        analytics_agent.llm.estimate_tokens = AsyncMock(return_value=300)

        state = AgentState(
            query="How many sales?",
            company_id=123,
            context="SQL: SELECT COUNT(*) FROM sales",
            sql_query="SELECT COUNT(*) FROM sales",
            sql_result={"status": "success", "data": [100]},
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
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

        state = AgentState(
            query="How many sales?",
            company_id=123,
            context="SQL context",
            sql_query="SELECT COUNT(*) FROM sales",
            sql_result={"data": [100]},
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        result_state = await analytics_agent._generate_insights(state)

        assert "apologize" in result_state["answer"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_response_analytics(self, analytics_agent):
        """Test analytics response evaluation."""
        analytics_agent.evaluation.evaluate_query = AsyncMock(
            return_value={"accuracy": 0.75, "sql_validity": 0.9}
        )

        state = AgentState(
            query="How many sales?",
            company_id=123,
            context="Analysis context",
            sql_query="SELECT COUNT(*) FROM sales",
            sql_result={"data": [100]},
            documents=[],
            answer="Sales count: 100",
            evaluation_metrics={},
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
            company_id=123,
            context="Test context",
            sql_query="SELECT * FROM test",
            sql_result={"status": "success"},
            documents=[{"title": "Doc1", "content": "Content1"}],
            answer="Test answer",
            evaluation_metrics={"accuracy": 0.9},
            response_time_ms=150,
            tokens_used=100,
        )

        assert state["query"] == "Test query"
        assert state["company_id"] == 123
        assert state["context"] == "Test context"
        assert state["sql_query"] == "SELECT * FROM test"
        assert state["sql_result"] == {"status": "success"}
        assert len(state["documents"]) == 1
        assert state["answer"] == "Test answer"
        assert state["evaluation_metrics"] == {"accuracy": 0.9}
        assert state["response_time_ms"] == 150
        assert state["tokens_used"] == 100

    def test_agent_state_optional_fields(self):
        """Test that optional fields can be None."""
        state = AgentState(
            query="Test query",
            company_id=None,
            context="",
            sql_query=None,
            sql_result=None,
            documents=[],
            answer="",
            evaluation_metrics={},
            response_time_ms=0,
            tokens_used=0,
        )

        assert state["company_id"] is None
        assert state["sql_query"] is None
        assert state["sql_result"] is None


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
            mock_llm.generate_answer = AsyncMock(return_value="Answer")
            mock_llm.estimate_tokens = AsyncMock(return_value=50)
            mock_llm_class.return_value = mock_llm

            mock_eval = MagicMock()
            mock_eval.evaluate_query = AsyncMock(return_value={})
            mock_eval_class.return_value = mock_eval

            mock_wandb = MagicMock()
            mock_wandb.enabled = False
            mock_wandb_class.return_value = mock_wandb

            agent = QAAgent()
            await agent.process_query("Test query", company_id=123)

            mock_retrieval.retrieve_documents.assert_called_once_with(
                "Test query", 123, top_k=5
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
            await agent.process_query("How many users?", company_id=123)

            mock_text_to_sql.generate_sql.assert_called_once()
            # Verify it was called with the query and company_id
            call_args = mock_text_to_sql.generate_sql.call_args
            assert call_args[0][0] == "How many users?"  # query
            assert call_args[0][1] == 123  # company_id

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
            mock_llm.generate_answer = AsyncMock(return_value="Answer")
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
            mock_llm.generate_answer = AsyncMock(return_value="Fallback answer")
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

            # Should still complete with fallback behavior
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

            # Should still complete despite SQL generation failure
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
            mock_llm.generate_answer = AsyncMock(return_value="Answer")
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

            # Should complete despite evaluation failure
            assert "answer" in result
            assert result["evaluation_metrics"] == {}
