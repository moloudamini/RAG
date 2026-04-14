# Multi-agent system (Text-to-SQL and Q&A)

A multi-agent system with Retrieval-Augmented Generation (RAG) that uses LangGraph to orchestrate two specialized agents: a **Q&A agent** backed by a vector store and a **text-to-SQL analytics agent** that convert NL-to-SQL and execute SQL query.

## Features

- **LangGraph Agent Orchestration**: Q&A and Analytics agents with structured workflows
- **Q&A Agent**: Answers general questions using hybrid search and cross-encoder reranking
- **Analytics Agent**: Converts natural language to SQL, executes it, and returns insights 
- **Auto Schema Introspection**: Reads live table schemas directly from the database 
- **Automatic Query Routing**: Intelligently routes queries to the appropriate agent based on content
- **Ollama Integration**: Local LLM inference for cost-effective, private deployment
- **Evaluation Pipeline**: LLM-as-judge metrics (faithfulness, answer relevance, completeness), link accuracy via vector similarity, citation accuracy, and SQL golden set validation
- **SQL Golden Set**: Reference-SQL-based evaluation that validates generated SQL returns the same data as hand-written ground truth queries
- **W&B Integration**: Experiment tracking and performance monitoring

## Architecture
![RAG System Architecture](assets/diagram.png)


**Note**: The diagram shows the complete RAG pipeline with automatic query routing, specialized agents, and evaluation metrics.

### Pipeline Components

1. **Query Classification**: Automatically determines if query is Q&A or Analytics based on keywords and patterns
2. **Q&A Agent**: Uses retrieval-augmented generation for general questions
3. **Analytics Agent**: Converts natural language to SQL with validation and execution
4. **Evaluation**: LLM-as-judge scoring (faithfulness, answer relevance, completeness) + SQL golden set with reference SQL comparison
5. **Monitoring**: W&B integration for experiment tracking and system observability

### Agents

#### Q&A Agent
- **Purpose**: Answer general questions using the document knowledge base
- **Workflow**: `retrieve_documents → generate_answer → evaluate_response`

#### Analytics Agent
- **Purpose**: Handle analytical queries requiring SQL data access
- **Workflow**: `generate_sql → validate_sql → execute_sql → generate_insights → evaluate_response`

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

### 5. Database Migrations and Seeding

Before running the application for the first time, you need to set up the database schema and add some sample data:

```bash
# Run database migrations to create tables
uv run alembic upgrade head

# Seed the database with sample company and product data
uv run python -m src.scripts.seed_business_data

# Check 
docker exec -it rag-postgres psql -U rag_user -d rag_db

SELECT * FROM products;

# Ingest initial documents for the Q&A agent
uv run python -m src.scripts.ingest
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
  "sql_response": {
    "sql": "SELECT SUM(p.price) AS total_sales, p.category FROM products p GROUP BY p.category",
    "confidence": 0.8,
    "is_valid": true
  },
  "documents": [],
  "answer": "Based on the SQL results, the total sales by product category are: Storage: $2400.00, Hardware: $850.50",
  "response_time_ms": 8077,
  "tokens_used": 206,
  "evaluation_metrics": {
    "sql_accuracy": 0.65,
    "link_accuracy": 0.72,
    "citation_accuracy": 1.0,
    "faithfulness": 0.91,
    "answer_relevance": 0.88,
    "completeness": 0.85
  }
}
```

### SQL Golden Set Evaluation

Validate the analytics agent's SQL generation against hand-written reference queries:

```bash
# Run all 15 golden set entries
curl -X POST "http://localhost:8000/api/v1/evaluation/golden-set"

# Run a subset by tag (aggregation, filter, join, group_by, temporal)
curl -X POST "http://localhost:8000/api/v1/evaluation/golden-set?tag=join"
```

Each entry is scored on: execution success, SQL keyword coverage, column match, row count, safety, and **result match** — whether the generated SQL returns the same data as the reference query against the live database.

To add golden set entries, edit `tests/fixtures/sql_golden_set.json`:

```json
{
  "id": "my_query",
  "query": "Natural language question",
  "reference_sql": "SELECT ... FROM ...;",
  "expected_sql_keywords": ["SELECT", "table_name"],
  "expected_columns": ["col"],
  "expected_row_count_min": 0,
  "tags": ["aggregation"]
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
```

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
- **W&B Dashboard**: https://wandb.ai/[your-entity]/rag-system

## Security

- All user input validated before processing
- SQL injection prevention (SELECT-only queries, parameterized execution)
- No sensitive data in logs
- Secrets via environment variables only

