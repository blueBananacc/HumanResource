"""组织树管理 API。

当前阶段只做 admin 管理入口,用于补齐 EPIC-01 的组织模型底座:
- GET    /api/org/tree           查询当前租户组织树
- POST   /api/org/nodes          创建组织节点
- PUT    /api/org/nodes/{id}     修改组织节点
- DELETE /api/org/nodes/{id}     删除叶子节点(有子节点/绑定用户则拒绝)
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.auth import require_admin
from app.domain.models import Employee, LifecycleEvent, OrgUnit, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import PERM_VIEW_REPORTS, require_permission

router = APIRouter(prefix="/org", tags=["org"])

class OrgNodeOut(BaseModel):
    id: str
    tenant_id: str
    parent_id: str | None
    name: str
    kind: str
    created_at: datetime
    updated_at: datetime
    children: list["OrgNodeOut"] = Field(default_factory=list)


class CreateOrgNodeIn(BaseModel):
    name: str
    kind: Literal["company", "department", "project"] = "department"
    parent_id: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("名称不能为空")
        return v


class UpdateOrgNodeIn(BaseModel):
    name: str | None = None
    kind: Literal["company", "department", "project"] | None = None
    parent_id: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("名称不能为空")
        return v


def _serialize(node: OrgUnit) -> OrgNodeOut:
    return OrgNodeOut(
        id=node.id,
        tenant_id=node.tenant_id,
        parent_id=node.parent_id,
        name=node.name,
        kind=node.kind,
        created_at=node.created_at,
        updated_at=node.updated_at,
        children=[],
    )


def _get_node_in_tenant(db: Session, *, node_id: str, tenant_id: str) -> OrgUnit:
    node = db.get(OrgUnit, node_id)
    if not node or node.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="组织节点不存在",
        )
    return node


def _ensure_parent_in_tenant(
    db: Session, *, parent_id: str | None, tenant_id: str
) -> OrgUnit | None:
    if parent_id is None:
        return None
    return _get_node_in_tenant(db, node_id=parent_id, tenant_id=tenant_id)


def _ensure_not_descendant(
    db: Session, *, node_id: str, new_parent_id: str | None, tenant_id: str
) -> None:
    if new_parent_id is None:
        return
    cursor = _get_node_in_tenant(db, node_id=new_parent_id, tenant_id=tenant_id)
    while cursor.parent_id is not None:
        if cursor.id == node_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不能把节点挂到自己的子孙节点下面",
            )
        cursor = _get_node_in_tenant(db, node_id=cursor.parent_id, tenant_id=tenant_id)
    if cursor.id == node_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能把节点挂到自己的子孙节点下面",
        )


@router.get("/tree", response_model=list[OrgNodeOut])
def get_org_tree(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
) -> list[OrgNodeOut]:
    rows = db.scalars(
        select(OrgUnit)
        .where(OrgUnit.tenant_id == current.tenant_id)
        .order_by(OrgUnit.created_at.asc())
    ).all()
    nodes = {row.id: _serialize(row) for row in rows}
    roots: list[OrgNodeOut] = []
    for row in rows:
        current_node = nodes[row.id]
        if row.parent_id and row.parent_id in nodes:
            nodes[row.parent_id].children.append(current_node)
        else:
            roots.append(current_node)
    return roots


@router.post("/nodes", response_model=OrgNodeOut, status_code=status.HTTP_201_CREATED)
def create_org_node(
    body: CreateOrgNodeIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_admin),
) -> OrgNodeOut:
    _ensure_parent_in_tenant(db, parent_id=body.parent_id, tenant_id=current.tenant_id)
    node = OrgUnit(
        tenant_id=current.tenant_id,
        parent_id=body.parent_id,
        name=body.name,
        kind=body.kind,
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    write_audit(
        db,
        actor=current,
        entity_type="org_unit",
        entity_id=node.id,
        action="create",
        detail={"name": node.name, "kind": node.kind, "parent_id": node.parent_id},
        request=request,
    )
    db.commit()
    return _serialize(node)


@router.put("/nodes/{node_id}", response_model=OrgNodeOut)
def update_org_node(
    node_id: str,
    body: UpdateOrgNodeIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_admin),
) -> OrgNodeOut:
    node = _get_node_in_tenant(db, node_id=node_id, tenant_id=current.tenant_id)
    before = {"name": node.name, "kind": node.kind, "parent_id": node.parent_id}
    if "parent_id" in body.model_fields_set:
        if body.parent_id == node.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="节点不能挂到自己下面",
            )
        _ensure_parent_in_tenant(db, parent_id=body.parent_id, tenant_id=current.tenant_id)
        _ensure_not_descendant(
            db,
            node_id=node.id,
            new_parent_id=body.parent_id,
            tenant_id=current.tenant_id,
        )
        node.parent_id = body.parent_id
    if body.name is not None:
        node.name = body.name
    if body.kind is not None:
        node.kind = body.kind
    after = {"name": node.name, "kind": node.kind, "parent_id": node.parent_id}
    write_audit(
        db,
        actor=current,
        entity_type="org_unit",
        entity_id=node.id,
        action="update",
        detail={"before": before, "after": after},
        request=request,
    )
    db.commit()
    db.refresh(node)
    return _serialize(node)


@router.delete("/nodes/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_org_node(
    node_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_admin),
) -> None:
    node = _get_node_in_tenant(db, node_id=node_id, tenant_id=current.tenant_id)
    has_children = db.scalars(
        select(OrgUnit.id)
        .where(OrgUnit.tenant_id == current.tenant_id, OrgUnit.parent_id == node.id)
        .limit(1)
    ).first()
    if has_children:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请先删除或迁移子节点",
        )
    has_users = db.scalars(
        select(User.id)
        .where(User.tenant_id == current.tenant_id, User.org_unit_id == node.id)
        .limit(1)
    ).first()
    if has_users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请先迁移归属该组织的用户",
        )
    write_audit(
        db,
        actor=current,
        entity_type="org_unit",
        entity_id=node.id,
        action="delete",
        detail={"name": node.name, "kind": node.kind},
        request=request,
    )
    db.delete(node)
    db.commit()


OrgNodeOut.model_rebuild()


# ===== H2:批量组织调整(助手「按讨论方案执行组织调整」的落点) =====
#
# 对话即方案:agent 把用户描述解析成 changes[] → preview 出规范化计划+
# 警告/错误 → 用户确认 → apply 事务执行。变更类型:
#   create_org_unit   {name, parent?, kind?}
#   rename_org_unit   {org, new_name}
#   move_org_unit     {org, new_parent?}   (null = 升到根)
#   transfer_employee {employee, to_org, new_manager?}
#   set_manager       {employee, manager?} (null = 清掉上级)
# org/employee 取值:id 或精确名字(大小写不敏感);名字歧义报候选。

class OrgChangeIn(BaseModel):
    changes: list[dict] = Field(min_length=1, max_length=200)


def _org_map(db: Session, tenant_id: str) -> dict[str, OrgUnit]:
    return {o.name.lower(): o for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == tenant_id)).all()}


def _emp_map(db: Session, tenant_id: str) -> dict[str, list[Employee]]:
    m: dict[str, list[Employee]] = {}
    for e in db.scalars(
        select(Employee).where(Employee.tenant_id == tenant_id,
                               Employee.status != "terminated")
    ).all():
        m.setdefault(e.name.lower(), []).append(e)
    return m


def _resolve_org(ref, orgs: dict[str, OrgUnit], extra: dict[str, OrgUnit],
                 db: Session, tenant_id: str) -> tuple[OrgUnit | None, str | None]:
    """ref = id 或名字;extra 存本批 create 出来还没入库的新节点。"""
    if ref is None:
        return None, None
    ref = str(ref).strip()
    if ref.lower() in extra:
        return extra[ref.lower()], None
    node = db.get(OrgUnit, ref)
    if node and node.tenant_id == tenant_id:
        return node, None
    hit = orgs.get(ref.lower())
    if hit:
        return hit, None
    return None, f"组织「{ref}」不存在(用 /org/tree 查名字或 id)"


def _resolve_emp(ref, emps: dict[str, list[Employee]], db: Session,
                 tenant_id: str) -> tuple[Employee | None, str | None]:
    ref = str(ref or "").strip()
    if not ref:
        return None, "缺少 employee"
    e = db.get(Employee, ref)
    if e and e.tenant_id == tenant_id:
        return e, None
    hits = emps.get(ref.lower()) or []
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        return None, f"员工「{ref}」有 {len(hits)} 个同名,请改用 employee_id"
    return None, f"员工「{ref}」不存在"


def _plan_change(c: dict, orgs, extra, emps, db, tenant_id) -> tuple[dict, list[str]]:
    """规范化单条变更 → (plan_row, errors)。plan_row.describe 给确认卡用。"""
    t = str(c.get("type") or "")
    err: list[str] = []
    if t == "create_org_unit":
        name = str(c.get("name") or "").strip()
        if not name:
            err.append("create_org_unit 缺 name")
        elif name.lower() in orgs or name.lower() in extra:
            err.append(f"组织「{name}」已存在")
        parent, e1 = _resolve_org(c.get("parent"), orgs, extra, db, tenant_id)
        if c.get("parent") and e1:
            err.append(e1)
        kind = c.get("kind") or "department"
        if kind not in ("company", "department", "project"):
            err.append(f"kind 非法:{kind}")
        return {"type": t, "name": name, "parent_id": parent.id if parent else None,
                "parent_name": parent.name if parent else "(根)",
                "kind": kind,
                "describe": f"新建组织 {name} 挂到 {parent.name if parent else '根节点'}"}, err
    if t == "rename_org_unit":
        node, e1 = _resolve_org(c.get("org"), orgs, extra, db, tenant_id)
        if e1:
            err.append(e1)
        new_name = str(c.get("new_name") or "").strip()
        if not new_name:
            err.append("rename_org_unit 缺 new_name")
        return {"type": t, "org_id": node.id if node else None,
                "old_name": node.name if node else str(c.get("org")),
                "new_name": new_name,
                "describe": f"{node.name if node else c.get('org')} 改名 {new_name}"}, err
    if t == "move_org_unit":
        node, e1 = _resolve_org(c.get("org"), orgs, extra, db, tenant_id)
        if e1:
            err.append(e1)
        parent, e2 = _resolve_org(c.get("new_parent"), orgs, extra, db, tenant_id)
        if c.get("new_parent") and e2:
            err.append(e2)
        if node and parent and node.id == parent.id:
            err.append("不能把节点挂到自己下面")
        elif node and parent:
            cursor = parent
            while cursor is not None:
                if cursor.id == node.id:
                    err.append("不能把节点挂到自己的子孙节点下面")
                    break
                cursor = (db.get(OrgUnit, cursor.parent_id)
                          if cursor.parent_id else None)
        return {"type": t, "org_id": node.id if node else None,
                "org_name": node.name if node else str(c.get("org")),
                "new_parent_id": parent.id if parent else None,
                "new_parent_name": parent.name if parent else "(根)",
                "describe": f"{node.name if node else c.get('org')} 移到 {parent.name if parent else '根节点'} 下"}, err
    if t == "transfer_employee":
        emp, e1 = _resolve_emp(c.get("employee"), emps, db, tenant_id)
        if e1:
            err.append(e1)
        org, e2 = _resolve_org(c.get("to_org"), orgs, extra, db, tenant_id)
        if e2:
            err.append(e2)
        mgr, e3 = (None, None)
        if c.get("new_manager"):
            mgr, e3 = _resolve_emp(c.get("new_manager"), emps, db, tenant_id)
            if e3:
                err.append(e3)
            if mgr and emp and mgr.id == emp.id:
                err.append("上级不能是本人")
        return {"type": t, "employee_id": emp.id if emp else None,
                "employee_name": emp.name if emp else str(c.get("employee")),
                "to_org_id": org.id if org else None,
                "to_org_name": org.name if org else str(c.get("to_org")),
                "new_manager_id": mgr.id if mgr else None,
                "new_manager_name": mgr.name if mgr else None,
                "describe": f"{emp.name if emp else c.get('employee')} 调到 {org.name if org else c.get('to_org')}" +
                            (f",上级改为 {mgr.name}" if mgr else "")}, err
    if t == "set_manager":
        emp, e1 = _resolve_emp(c.get("employee"), emps, db, tenant_id)
        if e1:
            err.append(e1)
        mgr, e2 = (None, None)
        if c.get("manager"):
            mgr, e2 = _resolve_emp(c.get("manager"), emps, db, tenant_id)
            if e2:
                err.append(e2)
            if mgr and emp and mgr.id == emp.id:
                err.append("上级不能是本人")
        return {"type": t, "employee_id": emp.id if emp else None,
                "employee_name": emp.name if emp else str(c.get("employee")),
                "manager_id": mgr.id if mgr else None,
                "manager_name": mgr.name if mgr else "(无)",
                "describe": f"{emp.name if emp else c.get('employee')} 上级设为 {mgr.name if mgr else '空'}"}, err
    return {"type": t or "?", "describe": f"未知变更类型 {t}"}, [f"未知变更类型:{t}"]


@router.post("/preview-changes")
def preview_org_changes(
    body: OrgChangeIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_admin),
):
    """规范化 + 校验一批组织调整,不落库。错误阻止执行,警告仅提示。"""
    orgs = _org_map(db, current.tenant_id)
    emps = _emp_map(db, current.tenant_id)
    extra: dict[str, OrgUnit] = {}
    plan, errors, warnings = [], [], []
    for i, c in enumerate(body.changes):
        row, err = _plan_change(c, orgs, extra, emps, db, current.tenant_id)
        row["idx"] = i
        plan.append(row)
        errors.extend(f"第{i + 1}条:{e}" for e in err)
        if row["type"] == "create_org_unit" and not err:
            extra[row["name"].lower()] = OrgUnit(
                id=f"new:{row['name']}", tenant_id=current.tenant_id,
                name=row["name"], kind=row["kind"])
        if row["type"] == "transfer_employee" and not row.get("new_manager_id"):
            warnings.append(f"第{i + 1}条:{row.get('employee_name')} 未指定新上级,沿用原汇报关系")
        if row["type"] == "move_org_unit":
            emp_cnt = db.scalar(
                select(func.count(Employee.id)).where(
                    Employee.tenant_id == current.tenant_id,
                    Employee.org_unit_id == row.get("org_id"))) or 0
            if emp_cnt:
                warnings.append(f"第{i + 1}条:{row.get('org_name')} 下 {emp_cnt} 名员工随节点移动")
    return {"ok": not errors, "changes": plan, "errors": errors, "warnings": warnings}


@router.post("/apply-changes")
def apply_org_changes(
    body: OrgChangeIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_admin),
):
    """事务执行一批组织调整。任一条校验失败整批 422,不做部分应用。"""
    orgs = _org_map(db, current.tenant_id)
    emps = _emp_map(db, current.tenant_id)
    extra: dict[str, OrgUnit] = {}
    plan, errors = [], []
    for i, c in enumerate(body.changes):
        row, err = _plan_change(c, orgs, extra, emps, db, current.tenant_id)
        plan.append(row)
        errors.extend(f"第{i + 1}条:{e}" for e in err)
        if row["type"] == "create_org_unit" and not err:
            extra[row["name"].lower()] = OrgUnit(
                id=f"new:{row['name']}", tenant_id=current.tenant_id,
                name=row["name"], kind=row["kind"])
    if errors:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=";".join(errors))

    applied = []
    new_nodes: dict[str, OrgUnit] = {}
    for row in plan:
        t = row["type"]
        if t == "create_org_unit":
            parent_id = row["parent_id"]
            if parent_id and parent_id.startswith("new:"):
                parent_id = new_nodes[parent_id].id
            node = OrgUnit(tenant_id=current.tenant_id, name=row["name"],
                           kind=row["kind"], parent_id=parent_id)
            db.add(node)
            db.flush()
            new_nodes[f"new:{row['name']}"] = node
            applied.append(row["describe"] + f"(id={node.id})")
        elif t == "rename_org_unit":
            node = _get_node_in_tenant(db, node_id=row["org_id"],
                                       tenant_id=current.tenant_id)
            node.name = row["new_name"]
            applied.append(row["describe"])
        elif t == "move_org_unit":
            node = _get_node_in_tenant(db, node_id=row["org_id"],
                                       tenant_id=current.tenant_id)
            pid = row["new_parent_id"]
            if pid and pid.startswith("new:"):
                pid = new_nodes[pid].id
            _ensure_not_descendant(db, node_id=node.id, new_parent_id=pid,
                                   tenant_id=current.tenant_id)
            node.parent_id = pid
            applied.append(row["describe"])
        elif t == "transfer_employee":
            emp = db.get(Employee, row["employee_id"])
            to_org = row["to_org_id"]
            if to_org and to_org.startswith("new:"):
                to_org = new_nodes[to_org].id
            frm = emp.org_unit_id
            emp.org_unit_id = to_org
            if row.get("new_manager_id"):
                emp.reports_to_id = row["new_manager_id"]
            db.add(LifecycleEvent(
                tenant_id=current.tenant_id, employee_id=emp.id,
                kind="transfer",
                payload={"from_org_unit_id": frm, "to_org_unit_id": to_org,
                         "via": "assistant_batch"},
                created_by=current.id,
            ))
            applied.append(row["describe"])
        elif t == "set_manager":
            emp = db.get(Employee, row["employee_id"])
            emp.reports_to_id = row["manager_id"]
            applied.append(row["describe"])
    write_audit(db, actor=current, entity_type="org_unit", entity_id="-",
                action="apply_changes",
                detail={"count": len(applied), "changes": applied},
                request=request)
    db.commit()
    return {"applied": len(applied), "results": applied}
