"""Company management API."""

from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db_session
from ..core.models import Company

logger = structlog.get_logger()
router = APIRouter()


class CompanyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    industry: Optional[str] = Field(default=None, max_length=100)
    website: Optional[str] = Field(default=None, max_length=500)


class CompanyResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    industry: Optional[str]
    website: Optional[str]

    model_config = {"from_attributes": True}


@router.post(
    "/",
    response_model=CompanyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new company",
)
async def create_company(
    payload: CompanyCreate,
    db: AsyncSession = Depends(get_db_session),
) -> CompanyResponse:
    company = Company(
        name=payload.name,
        description=payload.description,
        industry=payload.industry,
        website=payload.website,
    )
    db.add(company)
    await db.commit()
    await db.refresh(company)
    logger.info("Company created", company_id=company.id, name=company.name)
    return CompanyResponse.model_validate(company)


@router.get(
    "/",
    response_model=List[CompanyResponse],
    summary="List all companies",
)
async def list_companies(
    db: AsyncSession = Depends(get_db_session),
) -> List[CompanyResponse]:
    result = await db.execute(select(Company).order_by(Company.name))
    companies = result.scalars().all()
    return [CompanyResponse.model_validate(c) for c in companies]


@router.get(
    "/{company_id}",
    response_model=CompanyResponse,
    summary="Get a company by ID",
)
async def get_company(
    company_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> CompanyResponse:
    company = await db.get(Company, company_id)
    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {company_id} not found",
        )
    return CompanyResponse.model_validate(company)
