"""LLM Provider 配置 API(管理员专用)。

端点(均仅 admin 可访):
- GET    /api/llm/providers              列表(API key 用 mask 返回)
- POST   /api/llm/providers              新增
- PUT    /api/llm/providers/{id}         更新(api_key 留空则保留原 key)
- DELETE /api/llm/providers/{id}         删除(active 的不能删,先激活其他)
- POST   /api/llm/providers/{id}/activate  激活(同租户其他自动 inactive)
- POST   /api/llm/providers/{id}/test    用当前配置发一条 ping, 验证连通性

数据安全:
- 列表/详情**永不返回明文 api_key**, 只返回 ``api_key_masked``(``sk***xxxx``)
- 编辑时 ``api_key`` 字段留空 → 不变;非空 → 重新加密替换

设计说明:
- ``model`` 字段直接写 LiteLLM 标识符(如 ``openai/gpt-4o-mini``、``deepseek/deepseek-chat``、
  ``azure/<deployment>``);后端不再做厂商识别与前缀拼接,扩展新厂商零改代码
- ``base_url`` 留空 = 走 LiteLLM 默认(对 OpenAI 是公网 endpoint);其他厂商需用户显式填
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import LLMProvider, User
from app.infra.crypto import decrypt, encrypt, mask_secret
from app.infra.db import get_db
from app.integrations.llm.provider import LLMError, call_llm
from app.integrations.llm.registry import ResolvedProvider

router = APIRouter(prefix="/llm", tags=["llm"])


def _require_admin(current: User) -> None:
    if current.role != "admin":
        raise HTTPException(403, "仅管理员可管理 LLM 配置")


class ProviderIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    base_url: str | None = None
    api_key: str | None = Field(None, description="留空 / null 表示保留原 key")
    model: str = Field(..., min_length=1, max_length=128)
    embedding_model: str | None = Field(None, description="可选;知识库语义检索用")
    embedding_base_url: str | None = Field(
        None, description="可选;embedding 服务地址,留空共用 chat base_url"
    )
    embedding_api_key: str | None = Field(
        None, description="可选;embedding 服务 key,留空保留原值/共用 chat key"
    )


class ProviderOut(BaseModel):
    id: str
    name: str
    base_url: str | None
    api_key_masked: str
    model: str
    embedding_model: str | None
    embedding_base_url: str | None
    embedding_api_key_masked: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


def _to_out(p: LLMProvider) -> ProviderOut:
    plain = decrypt(p.api_key_encrypted)
    emb_key = (
        decrypt(p.embedding_api_key_encrypted)
        if p.embedding_api_key_encrypted
        else None
    )
    return ProviderOut(
        id=p.id,
        name=p.name,
        base_url=p.base_url,
        api_key_masked=mask_secret(plain),
        model=p.model,
        embedding_model=p.embedding_model,
        embedding_base_url=p.embedding_base_url,
        embedding_api_key_masked=mask_secret(emb_key) if emb_key else None,
        is_active=p.is_active,
        created_at=p.created_at,
        updated_at=p.updated_at,
    )


@router.get("/providers", response_model=list[ProviderOut])
def list_providers(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> list[ProviderOut]:
    _require_admin(current)
    rows = db.scalars(
        select(LLMProvider)
        .where(LLMProvider.tenant_id == current.tenant_id)
        .order_by(LLMProvider.created_at.desc())
    ).all()
    return [_to_out(p) for p in rows]


@router.post("/providers", response_model=ProviderOut, status_code=status.HTTP_201_CREATED)
def create_provider(
    body: ProviderIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> ProviderOut:
    _require_admin(current)
    if not body.api_key:
        raise HTTPException(400, "首次创建必须提供 api_key")
    p = LLMProvider(
        tenant_id=current.tenant_id,
        name=body.name.strip(),
        base_url=body.base_url or None,
        api_key_encrypted=encrypt(body.api_key),
        model=body.model.strip(),
        embedding_model=body.embedding_model.strip() if body.embedding_model else None,
        embedding_base_url=body.embedding_base_url or None,
        embedding_api_key_encrypted=encrypt(body.embedding_api_key)
        if body.embedding_api_key
        else None,
        is_active=False,
        created_by=current.id,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return _to_out(p)


@router.put("/providers/{provider_id}", response_model=ProviderOut)
def update_provider(
    provider_id: str,
    body: ProviderIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> ProviderOut:
    _require_admin(current)
    p = db.get(LLMProvider, provider_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(404, "Provider 不存在")
    p.name = body.name.strip()
    p.base_url = body.base_url or None
    p.model = body.model.strip()
    # embedding 三字段仅在显式传参时更新 — 未出现在请求体里视为"不动",
    # 这样 provider 编辑表单(不再含 embedding 区)保存时不会误清语义检索配置;
    # 显式传 null 仍可清空(独立 Embedding 设置页走此路径)。
    fs = body.model_fields_set
    if "embedding_model" in fs:
        p.embedding_model = body.embedding_model.strip() if body.embedding_model else None
    if "embedding_base_url" in fs:
        p.embedding_base_url = body.embedding_base_url or None
    if body.embedding_api_key:  # 非空才覆盖,空字符串保留旧 key
        p.embedding_api_key_encrypted = encrypt(body.embedding_api_key)
    if body.api_key:  # 非空才覆盖,空字符串保留旧 key
        p.api_key_encrypted = encrypt(body.api_key)
    p.updated_at = datetime.now(UTC).replace(tzinfo=None)
    db.commit()
    db.refresh(p)
    return _to_out(p)


# ---------- 语义检索(Embedding)独立配置 ----------

_EMBED_DEFAULT_BASE = "https://api.openai.com/v1"


class EmbeddingCfgIn(BaseModel):
    embedding_model: str | None = Field(
        None, description="语义检索模型;置空则关闭向量召回,退化为纯 BM25"
    )
    embedding_base_url: str | None = Field(
        None, description="embedding 服务地址,留空依次兜底 chat base_url/OpenAI"
    )
    embedding_api_key: str | None = Field(
        None, description="embedding 服务 key,留空保留原值;清空请走重置"
    )
    reset: bool = Field(False, description="为 true 时清除全部 embedding 配置")


class EmbeddingCfgOut(BaseModel):
    configured: bool  # 语义检索是否实际生效(有 model + 可用 key)
    provider_id: str | None
    provider_name: str | None
    embedding_model: str | None
    embedding_base_url: str | None
    embedding_api_key_masked: str | None
    effective_base_url: str | None  # 兜底后的实际请求地址
    api_key_source: str | None  # dedicated / chat / none


def _active_provider(db: Session, tenant_id: str) -> LLMProvider:
    p = db.scalars(
        select(LLMProvider)
        .where(LLMProvider.tenant_id == tenant_id, LLMProvider.is_active.is_(True))
        .limit(1)
    ).first()
    if not p:
        raise HTTPException(404, "尚无激活的 LLM Provider — 请先到「LLM 配置」激活一个")
    return p


def _embedding_out(p: LLMProvider) -> EmbeddingCfgOut:
    emb_key = (
        decrypt(p.embedding_api_key_encrypted) if p.embedding_api_key_encrypted else None
    )
    chat_key = decrypt(p.api_key_encrypted) if p.api_key_encrypted else None
    effective_base = (p.embedding_base_url or p.base_url or _EMBED_DEFAULT_BASE).rstrip("/")
    key_source = "dedicated" if emb_key else ("chat" if chat_key else "none")
    return EmbeddingCfgOut(
        configured=bool(p.embedding_model and (emb_key or chat_key)),
        provider_id=p.id,
        provider_name=p.name,
        embedding_model=p.embedding_model,
        embedding_base_url=p.embedding_base_url,
        embedding_api_key_masked=mask_secret(emb_key) if emb_key else None,
        effective_base_url=effective_base,
        api_key_source=key_source,
    )


@router.get("/embedding", response_model=EmbeddingCfgOut)
def get_embedding_cfg(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> EmbeddingCfgOut:
    _require_admin(current)
    return _embedding_out(_active_provider(db, current.tenant_id))


@router.put("/embedding", response_model=EmbeddingCfgOut)
def put_embedding_cfg(
    body: EmbeddingCfgIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> EmbeddingCfgOut:
    _require_admin(current)
    p = _active_provider(db, current.tenant_id)
    if body.reset:
        p.embedding_model = None
        p.embedding_base_url = None
        p.embedding_api_key_encrypted = None
    else:
        p.embedding_model = body.embedding_model.strip() if body.embedding_model else None
        p.embedding_base_url = body.embedding_base_url or None
        if body.embedding_api_key:  # 非空才覆盖,空字符串保留旧 key
            p.embedding_api_key_encrypted = encrypt(body.embedding_api_key)
    p.updated_at = datetime.now(UTC).replace(tzinfo=None)
    db.commit()
    db.refresh(p)
    return _embedding_out(p)


@router.delete("/providers/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_provider(
    provider_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> None:
    _require_admin(current)
    p = db.get(LLMProvider, provider_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(404, "Provider 不存在")
    if p.is_active:
        raise HTTPException(400, "请先激活其他 Provider 后再删除当前激活项")
    db.delete(p)
    db.commit()


@router.post("/providers/{provider_id}/activate", response_model=ProviderOut)
def activate_provider(
    provider_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> ProviderOut:
    _require_admin(current)
    p = db.get(LLMProvider, provider_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(404, "Provider 不存在")
    # 同租户其他全部 inactive
    db.execute(
        update(LLMProvider)
        .where(LLMProvider.tenant_id == current.tenant_id)
        .values(is_active=False)
    )
    p.is_active = True
    p.updated_at = datetime.now(UTC).replace(tzinfo=None)
    db.commit()
    db.refresh(p)
    return _to_out(p)


class TestResult(BaseModel):
    ok: bool
    message: str
    sample: str | None = None


@router.post("/providers/{provider_id}/test", response_model=TestResult)
def test_provider(
    provider_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> TestResult:
    _require_admin(current)
    p = db.get(LLMProvider, provider_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(404, "Provider 不存在")

    api_key = decrypt(p.api_key_encrypted)
    if not api_key:
        return TestResult(ok=False, message="无法解密 API key (可能加密 KEY 已变更)")

    resolved = ResolvedProvider(
        base_url=p.base_url,
        api_key=api_key,
        model=p.model,
        source="test",
        is_mock=False,
    )
    try:
        sample = call_llm(
            "llm.test",
            [
                {"role": "system", "content": "You are a connectivity test bot."},
                {"role": "user", "content": "请用一句话回复:连接成功。"},
            ],
            provider=resolved,
            tenant_id=current.tenant_id,
            user_id=current.id,
            temperature=0.0,
        )
        return TestResult(ok=True, message="连接成功", sample=sample[:200])
    except LLMError as e:
        return TestResult(ok=False, message=str(e))
