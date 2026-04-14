"""LLM service for text generation using Ollama."""

import json
import re
from typing import Dict, Any, List
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

    async def generate_answer_with_citations(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate an answer that cites retrieved documents inline.

        Args:
            query: The user's question
            docs: Retrieved documents with 'content' and 'metadata' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with 'answer' (str) and 'cited_indices' (list of 1-based ints)
        """
        logger.info(
            "Generating answer with citations",
            query=query[:100],
            doc_count=len(docs),
        )

        try:
            numbered_context = self._build_numbered_context(docs)
            prompt = self._create_citation_prompt(query, numbered_context, len(docs))

            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": max_tokens,
                },
            )

            answer = self._clean_response(response["response"].strip())
            cited_indices = self._extract_citation_indices(answer)

            logger.info(
                "Answer with citations generated",
                answer_length=len(answer),
                cited_count=len(cited_indices),
            )
            return {"answer": answer, "cited_indices": cited_indices}

        except Exception as e:
            logger.error("Citation answer generation failed", error=str(e))
            return {
                "answer": "I apologize, but I was unable to generate an answer at this time.",
                "cited_indices": [],
            }

    def _build_numbered_context(self, docs: List[Dict[str, Any]]) -> str:
        """Build a numbered context string from retrieved documents."""
        parts = []
        for i, doc in enumerate(docs, start=1):
            metadata = doc.get("metadata") or {}
            source = metadata.get("source") or metadata.get("filename") or f"Document {i}"
            content = (doc.get("content") or "")[:800]
            parts.append(f"[{i}] Source: {source}\n{content}")
        return "\n\n".join(parts)

    def _create_citation_prompt(
        self, query: str, numbered_context: str, doc_count: int
    ) -> str:
        """Create a prompt that instructs the LLM to cite sources."""
        return f"""You are a helpful AI assistant. Answer the question using ONLY the numbered sources below.
Cite each source inline using its number in square brackets, e.g. [1] or [2].
Only cite sources that are directly relevant to your answer.
Do not cite a source number higher than {doc_count}.

Sources:
{numbered_context}

Question: {query}

Instructions:
- Answer based on the sources above
- Use inline citations like [1], [2] where applicable
- If no source is relevant, state that clearly
- Be concise but complete

Answer:"""

    def _extract_citation_indices(self, answer: str) -> List[int]:
        """Extract unique 1-based citation indices from an answer string."""
        matches = re.findall(r"\[(\d+)\]", answer)
        seen: set[int] = set()
        result = []
        for m in matches:
            idx = int(m)
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
        return result

    async def judge_answer(
        self,
        query: str,
        answer: str,
        context_docs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Use the LLM as a judge to evaluate answer quality.

        Scores three dimensions on a 0.0–1.0 scale:
        - faithfulness: answer is grounded in the provided documents, not hallucinated
        - answer_relevance: answer addresses what the question actually asks
        - completeness: answer covers the key points available in the sources

        Args:
            query: The original user question
            answer: The generated answer to evaluate
            context_docs: Retrieved documents that were available to the LLM

        Returns:
            Dict with float scores for faithfulness, answer_relevance, completeness
        """
        logger.info("Running LLM-as-judge evaluation", query=query[:100])

        context_text = self._build_numbered_context(context_docs) if context_docs else "No documents provided."

        prompt = f"""You are an expert evaluator for a retrieval-augmented generation (RAG) system.
Evaluate the answer below on three dimensions. Return ONLY a JSON object with float scores between 0.0 and 1.0.

Question: {query}

Retrieved Sources:
{context_text}

Answer to evaluate:
{answer}

Scoring criteria:
- faithfulness: Is every claim in the answer supported by the sources? (1.0 = fully grounded, 0.0 = completely hallucinated)
- answer_relevance: Does the answer directly address what was asked? (1.0 = fully on-topic, 0.0 = completely off-topic)
- completeness: Does the answer cover all key points available in the sources for this question? (1.0 = comprehensive, 0.0 = misses everything important)

Return exactly this JSON structure:
{{
    "faithfulness": <float 0.0-1.0>,
    "answer_relevance": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>
}}"""

        try:
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.0, "num_predict": 100},
            )

            scores = json.loads(response["response"])

            result = {
                "faithfulness": float(max(0.0, min(1.0, scores.get("faithfulness", 0.0)))),
                "answer_relevance": float(max(0.0, min(1.0, scores.get("answer_relevance", 0.0)))),
                "completeness": float(max(0.0, min(1.0, scores.get("completeness", 0.0)))),
            }

            logger.info("LLM judge scores", **result)
            return result

        except Exception as e:
            logger.error("LLM judge evaluation failed", error=str(e))
            return {"faithfulness": 0.0, "answer_relevance": 0.0, "completeness": 0.0}

    async def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        This is a rough approximation for monitoring purposes.
        """
        # Simple approximation: ~4 characters per token for English text
        return len(text) // 4
