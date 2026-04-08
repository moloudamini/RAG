"""LLM service for text generation using Ollama."""

import json
from typing import Dict, Any
import structlog

import ollama

from ..core.config import settings

logger = structlog.get_logger()


class LLMService:
    """Service for LLM text generation using Ollama."""

    def __init__(self):
        """Initialize the LLM service."""
        self.client = ollama.AsyncClient(host=settings.ollama_base_url)
        self.model = settings.ollama_model

    async def generate_answer(
        self,
        query: str,
        context: str = "",
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate an answer using the LLM with optional context.

        Args:
            query: The user's question
            context: Additional context (SQL results, documents, etc.)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated answer text
        """
        logger.info("Generating answer", query=query[:100], context_length=len(context))

        try:
            # Create prompt with context
            prompt = self._create_answer_prompt(query, context)

            # Generate response
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": max_tokens,
                },
            )

            answer = response["response"].strip()

            # Clean up the response
            answer = self._clean_response(answer)

            logger.info("Answer generated", answer_length=len(answer))
            return answer

        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            return "I apologize, but I was unable to generate an answer at this time."

    async def validate_sql(
        self, sql_query: str, natural_query: str, schema_context: str = ""
    ) -> Dict[str, Any]:
        """
        Validate and potentially improve a generated SQL query.

        Args:
            sql_query: The SQL query to validate
            natural_query: The original natural language query
            schema_context: Database schema information

        Returns:
            Dict with validation results and improved query if needed
        """
        logger.info("Validating SQL query", sql_length=len(sql_query))

        try:
            prompt = f"""You are an expert SQL validator. Analyze the following SQL query for correctness and safety.

Original Question: {natural_query}
Generated SQL: {sql_query}

Schema Context: {schema_context}

Please analyze:
1. Is the SQL syntax correct?
2. Does it match the intent of the question?
3. Is it safe to execute (no dangerous operations)?
4. Any improvements needed?

Provide your analysis in JSON format:
{{
    "is_valid": boolean,
    "is_safe": boolean,
    "matches_intent": boolean,
    "improved_sql": "improved query if needed, otherwise empty string",
    "issues": ["list of issues found"],
    "confidence": 0.0-1.0
}}"""

            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": 0.1,
                    "num_predict": 300,
                },
            )

            result = json.loads(response["response"])

            logger.info(
                "SQL validation completed", is_valid=result.get("is_valid", False)
            )
            return result

        except Exception as e:
            logger.error("SQL validation failed", error=str(e))
            return {
                "is_valid": False,
                "is_safe": False,
                "matches_intent": False,
                "improved_sql": "",
                "issues": ["Validation failed"],
                "confidence": 0.0,
            }

    def _create_answer_prompt(self, query: str, context: str = "") -> str:
        """Create the prompt for answer generation."""
        base_prompt = f"""You are a helpful AI assistant that answers questions based on available information.

Question: {query}

"""

        if context:
            base_prompt += f"""Context Information:
{context}

"""

        base_prompt += """Instructions:
- Answer the question based on the provided context when available
- If no relevant context is provided, use your general knowledge
- Be concise but comprehensive
- If the question involves data analysis, explain your reasoning
- If you're unsure about something, say so clearly

Answer:"""

        return base_prompt

    def _clean_response(self, response: str) -> str:
        """Clean up the LLM response."""
        # Remove any system prompts that might have leaked through
        response = response.strip()

        # Remove common artifacts
        artifacts = [
            "Answer:",
            "Response:",
            "Based on the context:",
            "According to the information:",
        ]

        for artifact in artifacts:
            if response.startswith(artifact):
                response = response[len(artifact) :].strip()
                break

        return response

    async def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        This is a rough approximation for monitoring purposes.
        """
        # Simple approximation: ~4 characters per token for English text
        return len(text) // 4
