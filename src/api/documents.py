"""Document management API for the RAG pipeline."""

from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db_session
from ..core.models import Company, Document
from ..retrieval.service import RetrievalService

logger = structlog.get_logger()
router = APIRouter()


# =========================
# REQUEST / RESPONSE MODELS
# =========================


class DocumentCreate(BaseModel):
    company_name: str = Field(
        ..., min_length=1, max_length=255
    )  # Changed from company_id
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    content_type: str = Field(default="text", pattern="^(text|pdf|html|markdown)$")
    source_url: Optional[str] = Field(default=None, max_length=1000)
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    overlap: int = Field(default=200, ge=0, le=1000)


class DocumentResponse(BaseModel):
    id: int
    company_id: int
    title: str
    content_type: str
    source_url: Optional[str]
    chunk_count: int

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


# =========================
# ROUTES
# =========================


@router.post(
    "/",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a document for a company",
    description="Stores the document, chunks the content, generates embeddings, and indexes for retrieval. Creates the company if it doesn't exist.",
)
async def add_document(
    payload: DocumentCreate,
    db: AsyncSession = Depends(get_db_session),
) -> DocumentResponse:
    # Find or create company
    company = await db.execute(
        select(Company).where(Company.name == payload.company_name)
    )
    company = company.scalar_one_or_none()

    if company is None:
        # Create new company
        company = Company(name=payload.company_name)
        db.add(company)
        await db.flush()  # Get company.id before committing
        logger.info(
            "Company created automatically", company_id=company.id, name=company.name
        )

    # Persist document record
    document = Document(
        company_id=company.id,
        title=payload.title,
        content=payload.content,
        content_type=payload.content_type,
        source_url=payload.source_url,
    )
    db.add(document)
    await db.flush()  # get document.id before indexing

    # Chunk, embed and index
    retrieval_service = RetrievalService()
    success = await retrieval_service.index_document(
        document_id=document.id,
        content=payload.content,
        chunk_size=payload.chunk_size,
        overlap=payload.overlap,
        db=db,
    )

    if not success:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index document chunks",
        )

    await db.commit()
    await db.refresh(document)

    chunk_count = len(
        retrieval_service._chunk_text(
            payload.content, payload.chunk_size, payload.overlap
        )
    )

    logger.info(
        "Document added and indexed",
        document_id=document.id,
        company_id=company.id,
        chunks=chunk_count,
    )

    return DocumentResponse(
        id=document.id,
        company_id=document.company_id,
        title=document.title,
        content_type=document.content_type,
        source_url=document.source_url,
        chunk_count=chunk_count,
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List documents for a company",
)
async def list_documents(
    company_name: str,
    db: AsyncSession = Depends(get_db_session),
) -> DocumentListResponse:
    company = await db.execute(select(Company).where(Company.name == company_name))
    company = company.scalar_one_or_none()
    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company '{company_name}' not found",
        )

    result = await db.execute(
        select(Document)
        .where(Document.company_id == company.id)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=doc.id,
                company_id=doc.company_id,
                title=doc.title,
                content_type=doc.content_type,
                source_url=doc.source_url,
                chunk_count=0,  # not fetched for list view
            )
            for doc in documents
        ],
        total=len(documents),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a single document",
)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> DocumentResponse:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return DocumentResponse(
        id=document.id,
        company_id=document.company_id,
        title=document.title,
        content_type=document.content_type,
        source_url=document.source_url,
        chunk_count=0,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> None:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    await db.delete(document)
    await db.commit()
    logger.info("Document deleted", document_id=document_id)
