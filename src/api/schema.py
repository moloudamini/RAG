"""Schema registration API — lets users define their database tables for NL-to-SQL."""

from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db_session
from ..core.models import SchemaColumn, SchemaTable

logger = structlog.get_logger()
router = APIRouter()


# =========================
# REQUEST / RESPONSE MODELS
# =========================


class ColumnDefinition(BaseModel):
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="SQL data type, e.g. VARCHAR, INTEGER, FLOAT")
    description: Optional[str] = Field(None, description="What this column represents")
    is_primary_key: bool = False
    is_foreign_key: bool = False


class TableCreate(BaseModel):
    table_name: str = Field(..., description="Table name as it exists in the actual DB")
    description: Optional[str] = Field(None, description="What this table contains")
    columns: List[ColumnDefinition] = Field(..., min_length=1)


class ColumnResponse(BaseModel):
    id: int
    name: str
    data_type: str
    description: Optional[str]
    is_primary_key: bool
    is_foreign_key: bool

    model_config = {"from_attributes": True}


class TableResponse(BaseModel):
    id: int
    table_name: str
    description: Optional[str]
    columns: List[ColumnResponse]

    model_config = {"from_attributes": True}


class SchemaListResponse(BaseModel):
    tables: List[TableResponse]
    total: int


# =========================
# ROUTES
# =========================


@router.post(
    "/tables",
    response_model=TableResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a database table schema for NL-to-SQL",
    description=(
        "Register a table schema so the NL-to-SQL engine knows what tables "
        "and columns are available. The table must exist in the actual database "
        "you want to query."
    ),
)
async def register_table(
    payload: TableCreate,
    db: AsyncSession = Depends(get_db_session),
) -> TableResponse:
    existing = await db.execute(
        select(SchemaTable).where(SchemaTable.name == payload.table_name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Table '{payload.table_name}' already registered. Use PUT to update.",
        )

    schema_table = SchemaTable(
        name=payload.table_name,
        description=payload.description,
        schema_json={},
    )
    db.add(schema_table)
    await db.flush()

    columns = []
    for col_def in payload.columns:
        col = SchemaColumn(
            table_id=schema_table.id,
            name=col_def.name,
            data_type=col_def.data_type,
            description=col_def.description,
            is_primary_key=col_def.is_primary_key,
            is_foreign_key=col_def.is_foreign_key,
        )
        db.add(col)
        columns.append(col)

    await db.commit()
    await db.refresh(schema_table)

    logger.info("Table schema registered", table=payload.table_name, columns=len(columns))

    return TableResponse(
        id=schema_table.id,
        table_name=schema_table.name,
        description=schema_table.description,
        columns=[
            ColumnResponse(
                id=col.id,
                name=col.name,
                data_type=col.data_type,
                description=col.description,
                is_primary_key=col.is_primary_key,
                is_foreign_key=col.is_foreign_key,
            )
            for col in columns
        ],
    )


@router.get(
    "/tables",
    response_model=SchemaListResponse,
    summary="List all registered table schemas",
)
async def list_tables(
    db: AsyncSession = Depends(get_db_session),
) -> SchemaListResponse:
    tables_result = await db.execute(select(SchemaTable))
    tables = tables_result.scalars().all()

    response_tables = []
    for t in tables:
        cols_result = await db.execute(
            select(SchemaColumn).where(SchemaColumn.table_id == t.id)
        )
        cols = cols_result.scalars().all()
        response_tables.append(
            TableResponse(
                id=t.id,
                table_name=t.name,
                description=t.description,
                columns=[
                    ColumnResponse(
                        id=c.id,
                        name=c.name,
                        data_type=c.data_type,
                        description=c.description,
                        is_primary_key=c.is_primary_key,
                        is_foreign_key=c.is_foreign_key,
                    )
                    for c in cols
                ],
            )
        )

    return SchemaListResponse(tables=response_tables, total=len(response_tables))


@router.delete(
    "/tables/{table_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a registered table schema",
)
async def delete_table(
    table_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> None:
    table = await db.get(SchemaTable, table_id)
    if table is None:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")
    await db.delete(table)
    await db.commit()
    logger.info("Table schema deleted", table_id=table_id)
