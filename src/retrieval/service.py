"""Retrieval service: Focuses exclusively on querying ChromaDB and reranking candidates."""

from typing import List, Dict, Any
import structlog
import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from src.core.config import settings

logger = structlog.get_logger()


class RetrievalService:
    """Service for retrieving documents from vector store."""

    def __init__(self):
        self.embedding_model = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model
        )
        self.persist_directory = "db/chroma"
        self._vector_store = None
        self._reranker = None

    @property
    def vector_store(self) -> Chroma:
        """Lazy-load Chroma vector store."""
        if self._vector_store is None:
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory, exist_ok=True)
            
            self._vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name="documents"
            )
        return self._vector_store

    async def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents from ChromaDB"""
        logger.info("Retrieving documents", query=query[:100])

        
        vector_results = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=top_k * 2
        )

        if not vector_results:
            return []

        candidates = []
        for doc, score in vector_results:
            candidates.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
            })

        # Cross-encoder reranking
        if settings.use_reranker and len(candidates) > 1:
            candidates = self._rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        return candidates

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Rerank candidates using a cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            if self._reranker is None:
                self._reranker = CrossEncoder(settings.reranker_model)

            pairs = [(query, c["content"]) for c in candidates]
            ce_scores = self._reranker.predict(pairs)

            for i, c in enumerate(candidates):
                c["similarity_score"] = float(ce_scores[i])

            candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
            return candidates[:top_k]
        except Exception as e:
            logger.error("Reranking failed", error=str(e))
            return candidates[:top_k]
