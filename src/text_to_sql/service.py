"""Text-to-SQL conversion service using Ollama."""

import re
from typing import Dict, Any

import structlog
import ollama
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings

logger = structlog.get_logger()

_SYSTEM_TABLES = {"queries", "query_evaluations"}


class TextToSQLService:
    """Converts natural language queries to SQL using schema introspection."""

    def __init__(self):
        self.client = ollama.AsyncClient(host=settings.ollama_base_url)
        self.model = settings.ollama_model

    async def generate_sql(
        self,
        natural_query: str,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Convert natural language query to SQL and execute it."""
        logger.info("Generating SQL", query=natural_query[:100])

        schema_context = await self._introspect_schema(db)
        prompt = self._build_prompt(natural_query, schema_context)

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

        logger.info("SQL generated", sql=sql_query[:100], confidence=confidence, is_valid=is_valid)

        return {
            "sql": sql_query,
            "confidence": confidence,
            "is_valid": is_valid,
            "raw_response": raw_text,
            "schema_used": schema_context,
        }

    async def execute_sql(self, sql_query: str, db: AsyncSession, max_rows: int = 100) -> Dict[str, Any]:
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

    async def _introspect_schema(self, db: AsyncSession) -> str:
        """Build schema context by reading table/column info from database."""
        try:
            tables_sql = text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            result = await db.execute(tables_sql)
            table_names = [row[0] for row in result.fetchall()]
        except Exception:
            # SQLite fallback
            try:
                result = await db.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                table_names = [row[0] for row in result.fetchall()]
            except Exception as e:
                logger.error("Schema introspection failed", error=str(e))
                return "Schema introspection unavailable."

        # Filter out system tables
        table_names = [t for t in table_names if t not in _SYSTEM_TABLES]

        if not table_names:
            return "No business tables found in the database."

        parts = []
        for table in table_names:
            try:
                cols = await self._get_columns(db, table)
                parts.append(f"Table: {table}")
                parts.extend(f"  {col}" for col in cols)
                parts.append("")
            except Exception as e:
                logger.warning("Could not introspect table", table=table, error=str(e))

        return "\n".join(parts)

    async def _get_columns(self, db: AsyncSession, table: str) -> list[str]:
        """Get column definitions for a table."""
        try:
            result = await db.execute(text("""
                SELECT column_name, data_type,
                       is_nullable,
                       column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = :table
                ORDER BY ordinal_position
            """), {"table": table})
            rows = result.fetchall()
            if rows:
                return [f"{r[0]} {r[1].upper()}" for r in rows]
        except Exception:
            pass

        # SQLite fallback using PRAGMA
        result = await db.execute(text(f"PRAGMA table_info({table})"))
        rows = result.fetchall()
        return [f"{r[1]} {r[2].upper()}" for r in rows]

    def _build_prompt(self, query: str, schema_context: str) -> str:
        return f"""You are an expert SQL developer. Convert the natural language query below to a valid SQL SELECT statement.

Database Schema:
{schema_context}

Natural Language Query: {query}

Rules:
1. Output ONLY the SQL query — no explanation, no markdown fences
2. Use only SELECT statements (no INSERT, UPDATE, DELETE, DROP)
3. Use only table and column names from the schema above
4. End the query with a semicolon

SQL Query:"""

    def _extract_sql(self, response: str) -> str:
        sql = re.sub(r"```sql\s*", "", response, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*", "", sql).strip()

        lines = []
        for line in sql.split("\n"):
            lower = line.lower().strip()
            if any(lower.startswith(m) for m in ("explanation:", "note:", "this query", "the sql", "--")):
                break
            lines.append(line)

        return "\n".join(lines).strip()

    def _validate_sql(self, sql: str) -> bool:
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
