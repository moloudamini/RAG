"""Hybrid retrieval service: BM25 + vector similarity + cross-encoder reranking."""

from typing import List, Dict, Optional, Any
import numpy as np
import structlog

import ollama
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db_session_context
from ..core.config import settings
from ..core.models import DocumentChunk, Document

logger = structlog.get_logger()


class RetrievalService:
    """Hybrid retrieval: BM25 keyword + dense vector + cross-encoder reranking."""

    def __init__(self):
        self.client = ollama.AsyncClient(host=settings.ollama_base_url)
        self.embedding_model = settings.ollama_embedding_model
        self.similarity_threshold = settings.similarity_threshold
        self.bm25_weight = settings.bm25_weight
        self.vector_weight = settings.vector_weight
        self._reranker = None  # lazy-loaded

    # =========================
    # PUBLIC METHODS
    # =========================

    async def retrieve_documents(
        self,
        query: str,
        company_id: Optional[int] = None,
        top_k: int = 5,
        db: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        logger.info("Retrieving documents (hybrid)", query=query[:100], company_id=company_id)

        if db:
            return await self._retrieve(query, company_id, top_k, db)

        async with get_db_session_context() as session:
            return await self._retrieve(query, company_id, top_k, session)

    async def index_document(
        self,
        document_id: int,
        content: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        db: Optional[AsyncSession] = None,
    ) -> bool:
        logger.info("Indexing document", document_id=document_id)

        if db:
            return await self._index(document_id, content, chunk_size, overlap, db)

        async with get_db_session_context() as session:
            return await self._index(document_id, content, chunk_size, overlap, session)

    # =========================
    # INTERNAL: RETRIEVAL
    # =========================

    async def _retrieve(
        self,
        query: str,
        company_id: Optional[int],
        top_k: int,
        db: AsyncSession,
    ) -> List[Dict[str, Any]]:
        # Load all chunks for this company
        chunks = await self._load_chunks(company_id, db)
        if not chunks:
            logger.warning("No chunks found for retrieval", company_id=company_id)
            return []

        texts = [c["content"] for c in chunks]

        # Step 1: BM25 scores
        bm25_scores = self._bm25_scores(query, texts)

        # Step 2: Dense vector scores
        query_embedding = await self._embed(query)
        vector_scores = self._cosine_scores(query_embedding, chunks)

        # Step 3: Hybrid fusion (weighted combination after normalization)
        fused = self._fuse_scores(bm25_scores, vector_scores, len(chunks))

        # Take top candidates before reranking
        candidate_k = min(top_k * 3, len(chunks))
        top_indices = np.argsort(fused)[::-1][:candidate_k]
        candidates = [
            {**chunks[i], "hybrid_score": float(fused[i])}
            for i in top_indices
            if fused[i] >= self.similarity_threshold
        ]

        if not candidates:
            logger.info("No candidates above threshold", threshold=self.similarity_threshold)
            return []

        # Step 4: Cross-encoder reranking
        if settings.use_reranker and len(candidates) > 1:
            candidates = self._rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        logger.info("Documents retrieved", count=len(candidates))
        return candidates

    async def _load_chunks(
        self, company_id: Optional[int], db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Load all document chunks from DB (with company filter)."""
        stmt = (
            select(DocumentChunk, Document.title, Document.company_id)
            .join(Document, DocumentChunk.document_id == Document.id)
        )
        if company_id:
            stmt = stmt.where(Document.company_id == company_id)

        result = await db.execute(stmt)
        rows = result.all()

        chunks = []
        for chunk, title, cid in rows:
            embedding = self._parse_embedding(chunk.embedding)
            chunks.append(
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.chunk_metadata or {},
                    "title": title,
                    "company_id": cid,
                    "embedding": embedding,
                    "similarity_score": 0.0,
                }
            )
        return chunks

    def _bm25_scores(self, query: str, texts: List[str]) -> np.ndarray:
        """Compute BM25 scores for query against all texts."""
        try:
            from rank_bm25 import BM25Okapi

            tokenized_corpus = [t.lower().split() for t in texts]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(query.lower().split())
            return np.array(scores, dtype=float)
        except Exception as e:
            logger.error("BM25 scoring failed, falling back to zeros", error=str(e))
            return np.zeros(len(texts))

    def _cosine_scores(self, query_embedding: List[float], chunks: List[Dict]) -> np.ndarray:
        """Compute cosine similarity between query and all chunk embeddings."""
        q = np.array(query_embedding, dtype=float)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return np.zeros(len(chunks))

        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            emb = chunk.get("embedding")
            if emb is None:
                continue
            v = np.array(emb, dtype=float)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            scores[i] = float(np.dot(q, v) / (q_norm * v_norm))
        return scores

    def _fuse_scores(
        self, bm25: np.ndarray, vector: np.ndarray, n: int
    ) -> np.ndarray:
        """Normalize each score array to [0,1] then combine with weights."""
        def normalize(arr: np.ndarray) -> np.ndarray:
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        return self.bm25_weight * normalize(bm25) + self.vector_weight * normalize(vector)

    def _rerank(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        """Rerank candidates using a cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            if self._reranker is None:
                logger.info("Loading cross-encoder reranker", model=settings.reranker_model)
                self._reranker = CrossEncoder(settings.reranker_model)

            pairs = [(query, c["content"]) for c in candidates]
            ce_scores = self._reranker.predict(pairs)

            for i, c in enumerate(candidates):
                c["rerank_score"] = float(ce_scores[i])
                c["similarity_score"] = float(ce_scores[i])

            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return candidates[:top_k]

        except Exception as e:
            logger.error("Reranking failed, returning hybrid scores", error=str(e))
            for c in candidates:
                c["similarity_score"] = c["hybrid_score"]
            return candidates[:top_k]

    # =========================
    # INTERNAL: INDEXING
    # =========================

    async def _index(
        self,
        document_id: int,
        content: str,
        chunk_size: int,
        overlap: int,
        db: AsyncSession,
    ) -> bool:
        try:
            chunks = self._chunk_text(content, chunk_size, overlap)

            for i, chunk in enumerate(chunks):
                embedding = await self._embed(chunk)
                embedding_str = f"[{','.join(map(str, embedding))}]"

                chunk_obj = DocumentChunk(
                    document_id=document_id,
                    content=chunk,
                    chunk_index=i,
                    embedding=embedding_str,
                    chunk_metadata={"chunk_size": chunk_size, "overlap": overlap},
                )
                db.add(chunk_obj)

            await db.commit()
            logger.info("Document indexed", chunks=len(chunks))
            return True

        except Exception as e:
            logger.error("Document indexing failed", error=str(e))
            await db.rollback()
            return False

    # =========================
    # HELPERS
    # =========================

    async def _embed(self, text: str) -> List[float]:
        """Generate embedding via Ollama."""
        try:
            response = await self.client.embeddings(
                model=self.embedding_model,
                prompt=text,
            )
            return response["embedding"]
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise

    def _parse_embedding(self, raw: Optional[str]) -> Optional[List[float]]:
        """Parse stored embedding string back to list of floats."""
        if not raw:
            return None
        try:
            # Strip brackets and parse
            cleaned = raw.strip().lstrip("[").rstrip("]")
            return [float(x) for x in cleaned.split(",") if x.strip()]
        except Exception:
            return None

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                search_end = min(end + 100, len(text))
                sentence_end = text.rfind(".", end, search_end)
                if sentence_end != -1:
                    end = sentence_end + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
            if start >= end:
                start = end
        return chunks
