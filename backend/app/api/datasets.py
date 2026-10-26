# -*- coding: utf-8 -*-
"""E5 白名单数据集查询 API。

- ``GET /api/datasets``        各数据集可用维度/指标/过滤字段(LLM 选型用)
- ``POST /api/datasets/query``  {dataset, group_by[], metrics[], filters{}}

权限:view_reports(与报表同口径);行级 org scope 在服务层套用。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import User
from app.infra.db import get_db
from app.services.datasets import describe_datasets, query_dataset
from app.services.permissions import PERM_VIEW_REPORTS, require_permission

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
    dependencies=[
        Depends(require_feature("hr.suite")),
        Depends(require_permission(PERM_VIEW_REPORTS)),
    ],
)


@router.get("")
def list_datasets():
    return describe_datasets()


class QueryIn(BaseModel):
    dataset: str
    group_by: list[str] = Field(default_factory=list, max_length=4)
    metrics: list[str] = Field(default_factory=lambda: ["count"], max_length=4)
    filters: dict = Field(default_factory=dict)


@router.post("/query")
def query(
    body: QueryIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    return query_dataset(db, current, body.model_dump())
