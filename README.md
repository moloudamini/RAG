# Multi-agent system (Text-to-SQL and Q&A)

A multi-agent system with Retrieval-Augmented Generation (RAG) that uses LangGraph to orchestrate two specialized agents: a **Q&A agent** backed by a vector store and a **text-to-SQL analytics agent** that convert NL-to-SQL and execute SQL query.

## Features

- **LangGraph Agent Orchestration**: Q&A and Analytics agents with structured workflows
- **Q&A Agent**: Answers general questions using vector store document retrieval
- **Analytics Agent**: Converts natural language to SQL, executes it, and returns insights — all against one database
- **Auto Schema Introspection**: Reads live table schemas directly from the database — no manual registration needed
- **Automatic Query Routing**: Intelligently routes queries to the appropriate agent based on content
- **Ollama Integration**: Local LLM inference for cost-effective, private deployment
- **Evaluation Pipeline**: Metrics including link accuracy, relevance, and response time
- **W&B Integration**: Experiment tracking and performance monitoring
- **Production Ready**: REST API, Docker containerization, health monitoring, and structured logging

## Architecture
![RAG System Architecture](assets/diagram.png)


**Note**: The diagram shows the complete RAG pipeline with automatic query routing, specialized agents, and evaluation metrics.

### Pipeline Components

1. **Query Classification**: Automatically determines if query is Q&A or Analytics based on keywords and patterns
2. **Q&A Agent**: Uses retrieval-augmented generation for general questions
3. **Analytics Agent**: Converts natural language to SQL with validation and execution
4. **Evaluation**: Comprehensive metrics for accuracy, relevance, and performance
5. **Monitoring**: W&B integration for experiment tracking and system observability

### Agents

#### Q&A Agent
- **Purpose**: Answer general questions using the document knowledge base
- **Backend**: Vector store (ChromaDB/pgvector) for semantic document retrieval
- **Workflow**: `retrieve_documents → generate_answer → evaluate_response`

#### Analytics Agent
- **Purpose**: Handle analytical queries requiring SQL data access
- **Workflow**: `generate_sql → validate_sql → execute_sql → generate_insights → evaluate_response`

### Query Routing

Queries are automatically classified based on keywords and patterns:
- **Analytics**: "how many", "count", "sum", "sales", "revenue", SQL patterns
- **Q&A**: "what is", "explain", "describe", general questions

## Quick Start

### Prerequisites

- Python 3.12+
- uv (modern Python package manager)
- Docker and Docker Compose (optional, for PostgreSQL)
- Ollama (for local LLM inference)

### 1. Install Dependencies

```bash
pip install uv
uv sync
```

### 2. Configure Environment

Create a `.env` file:

```env
# Database
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_db

# W&B (optional)
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=rag-system
```

### 3. Start the Application

```bash
# Pull required Ollama models
ollama pull llama3.2
ollama pull nomic-embed-text

# Start the API
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. PostgreSQL via Docker

```bash
docker-compose -f docker/docker-compose.yml up -d
ollama serve
```

## API Usage

### Submit a Query

The system automatically routes to the correct agent:

```bash
# Analytics query → Analytics Agent
curl -X POST "http://localhost:8000/api/v1/queries/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the total sales by product category?"}'

# Q&A query → Q&A Agent
curl -X POST "http://localhost:8000/api/v1/queries/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our company mission statement?"}'

# Force a specific agent
curl -X POST "http://localhost:8000/api/v1/queries/" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain our sales trends", "force_agent": "analytics"}'
```

### Response Format

```json
{
  "query": "What are the total sales by product category?",
  "agent_used": "analytics",
  "answer": "Here is the SQL query generated for your request:\n\n```sql\nSELECT category, SUM(amount) FROM orders GROUP BY category;\n```",
  "sql_response": {
    "sql": "SELECT category, SUM(amount) FROM orders GROUP BY category;",
    "confidence": 0.85,
    "is_valid": true
  },
  "documents": [],
  "response_time_ms": 1250,
  "tokens_used": 450,
  "evaluation_metrics": {
    "sql_accuracy": 0.85,
    "relevance": 0.88
  }
}
```

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `DATABASE_URL` | App database (schema registry, query logs) | Required |
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | LLM model for generation | `llama3.2` |
| `VECTOR_DIMENSION` | Embedding vector size | `768` |
| `SIMILARITY_THRESHOLD` | Document retrieval threshold | `0.7` |
| `WANDB_API_KEY` | Weights & Biases API key | Optional |

## Development

```bash
# Run tests
uv run pytest tests/ --cov=src --cov-report=html

# Lint and format
uv run ruff check .
uv run ruff format .


## Docker Deployment

```bash
docker build -f docker/Dockerfile -t rag-system .

docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  rag-system
```

**Note**: Ollama runs on the host; use `host.docker.internal` to reach it from Docker.

## Monitoring

- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **W&B Dashboard**: https://wandb.ai/[your-entity]/rag-system

## Security

- All user input validated before processing
- SQL injection prevention (SELECT-only queries, parameterized execution)
- No sensitive data in logs
- Secrets via environment variables only

