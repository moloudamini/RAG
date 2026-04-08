# RAG System - Production Text-to-SQL and Q&A with LangGraph Agents

A multi-agent system with Retrieval-Augmented Generation (RAG) that uses LangGraph to orchestrate specialized agents for handling different types of queries. The system converts natural language to SQL and answers questions based on company information using Ollama for local LLM inference.

## Features

- **LangGraph Agent Orchestration**: Two specialized agents (Q&A and Analytics) with structured workflows
- **Q&A Agent**: Answers general questions using document retrieval and knowledge base
- **Analytics Agent**: Handles analytical queries with text-to-SQL conversion and data insights
- **Automatic Query Routing**: Intelligently routes queries to appropriate agents
- **Ollama Integration**: Use local LLM inference for cost-effective, private deployment
- **Evaluation Pipeline**: Comprehensive metrics including link accuracy, relevance, and response time
- **W&B Integration**: Experiment tracking and systematic performance monitoring
- **Production Ready**: REST API, Docker containerization, health monitoring, and structured logging
- **uv Package Management**: Modern Python dependency management with fast installs and lockfiles

## RAG Pipeline Architecture

![RAG System Architecture](assets/diagram.png)

*For a detailed view of the system architecture, see the diagram above showing the LangGraph agent orchestration with Q&A and Analytics agents.*

**Note**: The diagram shows the complete RAG pipeline with automatic query routing, specialized agents, and evaluation metrics.

### Pipeline Components

1. **Query Classification**: Automatically determines if query is Q&A or Analytics based on keywords and patterns
2. **Q&A Agent**: Uses retrieval-augmented generation for general questions
3. **Analytics Agent**: Converts natural language to SQL with validation and execution
4. **Evaluation**: Comprehensive metrics for accuracy, relevance, and performance
5. **Monitoring**: W&B integration for experiment tracking and system observability

### Agents

#### Q&A Agent (LangGraph Workflow)
- **Purpose**: Answer general questions using company knowledge base
- **Workflow**: `retrieve_documents → generate_answer → evaluate_response`
- **Capabilities**: Document search, knowledge base Q&A, general questions

#### Analytics Agent (LangGraph Workflow)
- **Purpose**: Handle analytical queries requiring data analysis
- **Workflow**: `generate_sql → validate_sql → execute_sql → generate_insights → evaluate_response`
- **Capabilities**: SQL generation, data analysis, reporting, metrics

### Query Classification

Queries are automatically classified based on:
- **Analytics**: Keywords like "how many", "count", "sum", "sales", "revenue", SQL patterns
- **Q&A**: Keywords like "what is", "explain", "describe", general questions

## Quick Start

### Prerequisites

- Python 3.12+
- uv (modern Python package manager)
- Docker and Docker Compose (optional, for full production setup)
- Ollama (optional, for local LLM inference)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd rag
```

### 2. Install Dependencies with uv

```bash
# Install uv if not already installed
pip install uv

# Install all dependencies (creates virtual environment automatically)
uv sync

# Activate the virtual environment (optional, uv run handles this)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Development Setup (SQLite)

For quick development and testing, the system uses SQLite by default:

```bash
# Start the application
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# The system will automatically create SQLite database and tables
```

### 4. Setup (PostgreSQL on Docker)

```bash
# Start PostgreSQL
docker-compose -f docker/docker-compose.yml up -d

# Start Ollama separately (on host)
ollama serve

# Pull required Ollama models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 5. Configure Environment (Optional)

Create a `.env` file for custom configuration:

```env
# Database (PostgreSQL primary, SQLite for development)
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_db

# Ollama (for local LLM inference)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# W&B Integration (optional)
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=rag-system
```

### 6. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

## API Usage

### Automatic Query Processing

The system automatically routes queries to the appropriate agent based on content analysis.

```bash
# Analytics query (will route to Analytics Agent)
curl -X POST "http://localhost:8000/api/v1/queries/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the total sales by product category?",
    "company_id": 1
  }'

# Q&A query (will route to Q&A Agent)
curl -X POST "http://localhost:8000/api/v1/queries/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is our company mission statement?",
    "company_id": 1
  }'
```

### Force Specific Agent

You can override automatic classification by specifying the agent:

```bash
# Force Q&A agent for an analytics-style query
curl -X POST "http://localhost:8000/api/v1/queries/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the total sales by product category?",
    "company_id": 1,
    "force_agent": "qa"
  }'

# Force Analytics agent for a general question
curl -X POST "http://localhost:8000/api/v1/queries/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain our company culture",
    "company_id": 1,
    "force_agent": "analytics"
  }'
```

### Response Format

```json
{
  "query": "What are the total sales by product category?",
  "agent_used": "analytics",
  "sql_response": {
    "sql": "SELECT category, SUM(sales) FROM products GROUP BY category;",
    "confidence": 0.85,
    "is_valid": true
  },
  "documents": [],
  "answer": "Based on the sales data analysis...",
  "response_time_ms": 1250,
  "tokens_used": 450,
  "evaluation_metrics": {
    "sql_accuracy": 0.85,
    "relevance": 0.88
  }
}
```

## Configuration

### Core Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default LLM model | `llama3.2` |
| `VECTOR_DIMENSION` | Embedding vector size | `768` |
| `SIMILARITY_THRESHOLD` | Document retrieval threshold | `0.7` |

### W&B Integration

```env
WANDB_API_KEY=your_api_key
WANDB_PROJECT=rag-system
WANDB_ENTITY=your_team
```

## Development

### Running Tests

```bash
# Unit tests
uv run pytest tests/unit/

# Integration tests
uv run pytest tests/integration/

# End-to-end tests
uv run pytest tests/e2e/

# With coverage
uv run pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Linting and formatting
uv run ruff check .
uv run ruff format .
```

### Database Management

The system uses SQLAlchemy's `create_all()` for schema management. Tables are created automatically on startup.

```bash
# Create tables manually if needed
python -c "import asyncio; from src.core.database import create_tables; asyncio.run(create_tables())"
```

### Running the Application

```bash
# Development mode with auto-reload
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## LangGraph Workflows

### Q&A Agent Workflow

```python
from src.agents.orchestrator import QAAgent

agent = QAAgent()
graph = agent.create_graph()

# The workflow handles:
# 1. Document retrieval based on query
# 2. Context building from relevant documents
# 3. Answer generation using LLM
# 4. Response evaluation and metrics
```

### Analytics Agent Workflow

```python
from src.agents.orchestrator import AnalyticsAgent

agent = AnalyticsAgent()
graph = agent.create_graph()

# The workflow handles:
# 1. SQL generation from natural language
# 2. SQL validation and improvement
# 3. SQL execution (simulated currently)
# 4. Insights generation from results
# 5. Response evaluation and metrics
```

## Docker Deployment

### Build and Run

```bash
# Build the application image
docker build -f docker/Dockerfile -t rag-system .

# Run with docker-compose (PostgreSQL only)
docker-compose -f docker/docker-compose.yml up -d

# Or run standalone with PostgreSQL
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  rag-system
```

**Note**: Ollama runs separately on the host system for local LLM inference.

## Evaluation & Monitoring

### Metrics

The system tracks comprehensive metrics for each agent:

- **Q&A Agent**: Link accuracy, answer relevance, response time
- **Analytics Agent**: SQL accuracy, answer relevance, response time
- **System-wide**: Token usage, error rates, agent selection accuracy

### W&B Dashboard

View experiment results and metrics at: https://wandb.ai/[your-entity]/rag-system

### Health Checks

- **Application Health**: `/health`
- **Detailed Health**: `/health/detailed`
- **Metrics**: `/metrics`

## Security Considerations

- Input validation and sanitization
- SQL injection prevention through parameterized queries
- Rate limiting on API endpoints
- Structured logging (no sensitive data)
- Environment variable configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Your License Here]
  "answer": "Based on the sales data, the total sales by category are...",
  "response_time_ms": 1250,
  "evaluation_metrics": {
    "sql_accuracy": 0.85,
    "link_accuracy": 0.92,
    "relevance": 0.88
  }
}
```

## Configuration

### Core Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default LLM model | `llama3.2:3b` |
| `VECTOR_DIMENSION` | Embedding vector size | `768` |
| `SIMILARITY_THRESHOLD` | Document retrieval threshold | `0.7` |

### W&B Integration

```env
WANDB_API_KEY=your_api_key
WANDB_PROJECT=rag-system
WANDB_ENTITY=your_team
```

## Docker Deployment

### Build and Run

```bash
# Build the application image
docker build -f docker/Dockerfile -t rag-system .

# Run with docker-compose
docker-compose -f docker/docker-compose.yml up -d

# Or run standalone
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  rag-system
```

## Evaluation & Monitoring

### Metrics

The system tracks comprehensive metrics:

- **SQL Accuracy**: Correctness of generated SQL queries
- **Link Accuracy**: Relevance of retrieved documents
- **Answer Relevance**: Quality of generated responses
- **Response Time**: Query processing latency
- **Token Usage**: LLM resource consumption

### W&B Dashboard

View experiment results and metrics at: https://wandb.ai/[your-entity]/rag-system

### Health Checks

- **Application Health**: `/health`
- **Detailed Health**: `/health/detailed`
- **Metrics**: `/metrics`

## Security Considerations

- Input validation and sanitization
- SQL injection prevention through parameterized queries
- Rate limiting on API endpoints
- Structured logging (no sensitive data)
- Environment variable configuration

