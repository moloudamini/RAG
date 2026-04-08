"""Text-to-SQL conversion service using Ollama."""

import re
from typing import Dict, Optional, Any, List
import structlog

import ollama
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings

logger = structlog.get_logger()


class TextToSQLService:
    """Service for converting natural language queries to SQL using Ollama."""

    def __init__(self):
        self.client = ollama.AsyncClient(host=settings.ollama_base_url)
        self.model = settings.ollama_model

    async def generate_sql(
        self,
        natural_query: str,
        company_id: Optional[int] = None,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """Convert natural language query to SQL and execute it."""
        logger.info("Generating SQL", query=natural_query[:100])

        schema_context = await self._get_schema_context(company_id, db)

        prompt = self._create_sql_prompt(natural_query, schema_context)

        response = await self.client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.1, "top_p": 0.9, "num_predict": 500},
        )

        raw_text = response.get("response", "").strip()
        if not raw_text:
            raise ValueError("Ollama returned empty response for SQL generation")

        sql_query = self._extract_sql(raw_text)
        if not sql_query:
            raise ValueError(f"Could not extract SQL from model response: {raw_text[:200]}")

        is_valid = self._validate_sql(sql_query)
        confidence = self._calculate_confidence(sql_query)

        result = {
            "sql": sql_query,
            "confidence": confidence,
            "is_valid": is_valid,
            "raw_response": raw_text,
            "schema_used": schema_context,
        }

        logger.info("SQL generated", sql=sql_query[:100], confidence=confidence, is_valid=is_valid)
        return result

    async def execute_sql(
        self,
        sql_query: str,
        db: AsyncSession,
        max_rows: int = 100,
    ) -> Dict[str, Any]:
        """Execute a validated SQL query and return results."""
        if not self._validate_sql(sql_query):
            return {"error": "SQL failed safety validation", "rows": [], "row_count": 0}

        try:
            result = await db.execute(text(sql_query))
            rows = result.fetchmany(max_rows)
            columns = list(result.keys()) if result.keys() else []

            data = [dict(zip(columns, row)) for row in rows]

            logger.info("SQL executed", row_count=len(data))
            return {
                "columns": columns,
                "data": data,
                "row_count": len(data),
                "truncated": len(rows) == max_rows,
            }

        except Exception as e:
            logger.error("SQL execution failed", error=str(e), sql=sql_query[:200])
            return {"error": str(e), "rows": [], "row_count": 0}

    async def _get_schema_context(
        self, company_id: Optional[int], db: Optional[AsyncSession]
    ) -> str:
        """Build schema context from SchemaTable/SchemaColumn rows, with document-based fallback."""
        if not db:
            return "No schema context available."

        try:
            from ..core.models import SchemaTable, SchemaColumn, Document
            from sqlalchemy import select

            # Try structured schema tables first
            stmt = select(SchemaTable)
            if company_id:
                stmt = stmt.where(SchemaTable.company_id == company_id)

            tables_result = await db.execute(stmt)
            tables = tables_result.scalars().all()

            if tables:
                return await self._format_schema_from_tables(tables, db)

            # Fallback: derive schema hint from document titles
            return await self._schema_from_documents(company_id, db)

        except Exception as e:
            logger.error("Failed to get schema context", error=str(e))
            return "Schema context unavailable."

    async def _format_schema_from_tables(self, tables, db: AsyncSession) -> str:
        """Format schema context from SchemaTable + SchemaColumn rows."""
        from ..core.models import SchemaColumn
        from sqlalchemy import select

        parts = []
        for table in tables:
            cols_result = await db.execute(
                select(SchemaColumn).where(SchemaColumn.table_id == table.id)
            )
            cols = cols_result.scalars().all()
            col_defs = []
            for col in cols:
                markers = []
                if col.is_primary_key:
                    markers.append("PK")
                if col.is_foreign_key:
                    markers.append("FK")
                suffix = f" ({', '.join(markers)})" if markers else ""
                desc = f" -- {col.description}" if col.description else ""
                col_defs.append(f"  {col.name} {col.data_type}{suffix}{desc}")

            parts.append(f"Table: {table.name}")
            if table.description:
                parts.append(f"  Description: {table.description}")
            parts.extend(col_defs)
            parts.append("")

        return "\n".join(parts)

    async def _schema_from_documents(
        self, company_id: Optional[int], db: AsyncSession
    ) -> str:
        """Fallback: use document titles as a hint about available information."""
        from ..core.models import Document
        from sqlalchemy import select

        stmt = select(Document.title)
        if company_id:
            stmt = stmt.where(Document.company_id == company_id)
        stmt = stmt.limit(20)

        result = await db.execute(stmt)
        titles = [row[0] for row in result.all()]

        if not titles:
            return "No schema or documents found for this company."

        return (
            "Available document topics (use these as reference, not table names):\n"
            + "\n".join(f"- {t}" for t in titles)
        )

    def _create_sql_prompt(self, query: str, schema_context: str) -> str:
        return f"""You are an expert SQL developer. Convert the natural language query below to a valid SQL SELECT statement.

Database Schema / Context:
{schema_context}

Natural Language Query: {query}

Rules:
1. Output ONLY the SQL query — no explanation, no markdown fences
2. Use only SELECT statements (no INSERT, UPDATE, DELETE, DROP)
3. Use proper table and column names from the schema above
4. End the query with a semicolon

SQL Query:"""

    def _extract_sql(self, response: str) -> str:
        """Extract clean SQL from LLM response."""
        # Strip markdown code fences
        sql = re.sub(r"```sql\s*", "", response, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*", "", sql)
        sql = sql.strip()

        # Take only lines up to the first explanation
        lines = []
        for line in sql.split("\n"):
            lower = line.lower().strip()
            if any(lower.startswith(m) for m in ("explanation:", "note:", "this query", "the sql", "--")):
                break
            lines.append(line)

        return "\n".join(lines).strip()

    def _validate_sql(self, sql: str) -> bool:
        """Ensure SQL is a safe SELECT-only query."""
        if not sql:
            return False
        upper = sql.upper()
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
        return "SELECT" in upper and "FROM" in upper and not any(kw in upper for kw in dangerous)

    def _calculate_confidence(self, sql: str) -> float:
        confidence = 0.5
        if sql.strip().endswith(";"):
            confidence += 0.1
        if re.search(r"SELECT\s+.+\s+FROM\s+", sql, re.IGNORECASE):
            confidence += 0.2
        if "JOIN" in sql.upper():
            confidence += 0.1
        if len(sql) < 20:
            confidence -= 0.2
        if len(sql) > 500:
            confidence -= 0.1
        return max(0.0, min(1.0, confidence))
