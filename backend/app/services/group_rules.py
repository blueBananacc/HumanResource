"""人才库自动入库规则引擎(北森"自动入库"同款思路)。

规则契约(存 ``CandidateGroup.auto_rule`` JSONB)::

    {
      "mode": "all" | "any",            # all=全部条件满足(AND) any=任一(OR)
      "conditions": [
        {"field": <字段>, "op": <算子>, "value": <值>}
      ]
    }

字段词表(``FIELDS`` — 前端编辑器后端同源):

    name        候选人姓名          contains / not_contains     str
    email       邮箱               contains / not_contains     str
    tag         候选人标签          contains_any / contains_all list[str]
    skill       技能(档案+简历)     contains_any / contains_all list[str]
    stage       管道阶段(任一申请)  in / not_in                list[STAGES]
    status      档案状态            eq                          active|archived
    blacklisted 黑名单             eq                          bool
    source      简历来源(任一)      in                          list[str]
    job         申请过岗位          eq                          job_id str
    text        简历全文/画像       contains / not_contains     str
                (公司/学校/职位/学历等自由文本)

设计约束:
- **确定性可解释**:每条 condition 产出一条命中理由,UI 可逐条展示,
  不做黑盒评分 — 这与决策学习的 BARS 证据链同一思路。
- **只增不删**:规则命中即入组(source=auto);规则改了不踢人 — 手动
  与自动成员并存,删除只能手工。北森/大易同理(避免规则抖动误删池子)。
- **幂等**:(group_id, candidate_id) 唯一约束兜底,重复评估不重复插。
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    Candidate,
    CandidateGroup,
    CandidateGroupMember,
    CandidateSkill,
    JobApplication,
    Resume,
    Skill,
)

logger = logging.getLogger(__name__)

MAX_CONDITIONS = 10
MODES = ("all", "any")
IN_OPS = ("in", "not_in")
LIST_OPS = ("contains_any", "contains_all")
TEXT_OPS = ("contains", "not_contains")

# field -> {"ops": 可用算子, "value": 值类型提示}
FIELDS: dict[str, dict[str, Any]] = {
    "name": {"ops": TEXT_OPS, "value": "str", "label": "姓名"},
    "email": {"ops": TEXT_OPS, "value": "str", "label": "邮箱"},
    "tag": {"ops": LIST_OPS, "value": "str[]", "label": "标签"},
    "skill": {"ops": LIST_OPS, "value": "str[]", "label": "技能"},
    "stage": {"ops": IN_OPS, "value": "stage[]", "label": "管道阶段"},
    "status": {"ops": ("eq",), "value": "status", "label": "档案状态"},
    "blacklisted": {"ops": ("eq",), "value": "bool", "label": "黑名单"},
    "source": {"ops": ("in",), "value": "str[]", "label": "简历来源"},
    "job": {"ops": ("eq",), "value": "job_id", "label": "申请过岗位"},
    "text": {"ops": TEXT_OPS, "value": "str", "label": "简历全文"},
}

STAGES = JobApplication.STAGES
STATUSES = ("active", "archived")


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ---------------- 规则校验 ----------------


def validate_rule(raw: Any) -> dict[str, Any]:
    """校验并规范化规则;非法抛 ValueError(消息可直接回前端)。"""
    if raw is None:
        raise ValueError("auto_rule 不能为空(清除请用 DELETE)")
    if not isinstance(raw, dict):
        raise ValueError("auto_rule 必须是对象 {mode, conditions}")
    mode = str(raw.get("mode") or "all")
    if mode not in MODES:
        raise ValueError("mode 取值 all|any")
    conds = raw.get("conditions")
    if not isinstance(conds, list) or not conds:
        raise ValueError("conditions 至少一条")
    if len(conds) > MAX_CONDITIONS:
        raise ValueError(f"conditions 最多 {MAX_CONDITIONS} 条")

    out: list[dict[str, Any]] = []
    for i, c in enumerate(conds):
        if not isinstance(c, dict):
            raise ValueError(f"第{i + 1}条条件必须是对象")
        field = str(c.get("field") or "")
        spec = FIELDS.get(field)
        if spec is None:
            raise ValueError(f"第{i + 1}条:未知字段 '{field}',可选:{','.join(FIELDS)}")
        op = str(c.get("op") or "")
        if op not in spec["ops"]:
            raise ValueError(
                f"第{i + 1}条:{field} 不支持算子 '{op}',可选:{','.join(spec['ops'])}"
            )
        value = c.get("value")
        value = _norm_value(field, op, value, i)
        out.append({"field": field, "op": op, "value": value})
    return {"mode": mode, "conditions": out}


def _norm_value(field: str, op: str, value: Any, idx: int) -> Any:
    err = f"第{idx + 1}条({field})"
    if field == "blacklisted":
        if not isinstance(value, bool):
            raise ValueError(f"{err}:value 必须是 true/false")
        return value
    if field == "status":
        if value not in STATUSES:
            raise ValueError(f"{err}:取值 {STATUSES}")
        return value
    if field == "stage":
        vals = _str_list(value, err)
        bad = [v for v in vals if v not in STAGES]
        if bad:
            raise ValueError(f"{err}:非法阶段 {bad},可选 {STAGES}")
        return vals
    if field == "job":
        v = str(value or "").strip()
        if not v:
            raise ValueError(f"{err}:job_id 不能为空")
        return v
    if op in LIST_OPS or op in IN_OPS:
        return _str_list(value, err)
    v = str(value or "").strip()
    if not v:
        raise ValueError(f"{err}:value 不能为空")
    return v[:256]


def _str_list(value: Any, err: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{err}:value 必须是数组")
    out = [str(v).strip()[:64] for v in value if str(v or "").strip()]
    if not out:
        raise ValueError(f"{err}:value 数组不能为空")
    return list(dict.fromkeys(out))[:50]


# ---------------- 候选人事实装配 ----------------


def _candidate_facts(db: Session, cand: Candidate) -> dict[str, Any]:
    """把候选人可判定数据装配成一次评估用的事实集。"""
    skills = set(
        db.scalars(
            select(Skill.name)
            .join(CandidateSkill, CandidateSkill.skill_id == Skill.id)
            .where(CandidateSkill.candidate_id == cand.id)
        ).all()
    )
    apps = db.scalars(
        select(JobApplication).where(JobApplication.candidate_id == cand.id)
    ).all()
    resumes = db.scalars(
        select(Resume)
        .where(Resume.candidate_id == cand.id, Resume.parse_status == "done")
        .order_by(Resume.created_at.desc())
        .limit(3)
    ).all()
    # 简历全文:最近 3 份解析完简历的 parsed_text + 结构化 skills
    text_parts: list[str] = []
    for r in resumes:
        pd = r.parsed_data or {}
        skills.update(str(s) for s in (pd.get("skills") or []) if s)
        if r.parsed_text:
            text_parts.append(r.parsed_text[:20000])
        if pd.get("sections"):
            text_parts.append(str(pd["sections"])[:10000])
    return {
        "name": cand.name or "",
        "email": cand.display_email or "",
        "tags": set(cand.tags or []),
        "skills": {s.lower() for s in skills},
        "stages": {a.stage for a in apps},
        "status": cand.status,
        "blacklisted": bool(cand.is_blacklisted),
        "sources": {r.source for r in resumes} | {a.source for a in apps},
        "job_ids": {a.job_id for a in apps},
        "text": "\n".join(text_parts).lower(),
    }


def _eval_condition(cond: dict[str, Any], facts: dict[str, Any]) -> str | None:
    """单条件评估;命中返回中文理由,未命中返回 None。"""
    f, op, v = cond["field"], cond["op"], cond["value"]
    if f == "name":
        hit = v.lower() in facts["name"].lower()
        if op == "contains" and hit:
            return f"姓名含「{v}」"
        if op == "not_contains" and not hit:
            return f"姓名不含「{v}」"
    elif f == "email":
        hit = v.lower() in facts["email"].lower()
        if op == "contains" and hit:
            return f"邮箱含「{v}」"
        if op == "not_contains" and not hit:
            return f"邮箱不含「{v}」"
    elif f == "tag":
        want = set(v)
        if op == "contains_any" and want & facts["tags"]:
            return f"标签命中 {sorted(want & facts['tags'])}"
        if op == "contains_all" and want <= facts["tags"]:
            return f"标签全含 {sorted(want)}"
    elif f == "skill":
        want = {s.lower() for s in v}
        if op == "contains_any" and want & facts["skills"]:
            return f"技能命中 {sorted(want & facts['skills'])}"
        if op == "contains_all" and want <= facts["skills"]:
            return f"技能全含 {sorted(want)}"
    elif f == "stage":
        inter = set(v) & facts["stages"]
        if op == "in" and inter:
            return f"管道阶段命中 {sorted(inter)}"
        if op == "not_in" and not inter:
            return f"管道阶段不在 {sorted(v)}"
    elif f == "status":
        if facts["status"] == v:
            return f"档案状态={v}"
    elif f == "blacklisted":
        if facts["blacklisted"] == v:
            return f"黑名单={v}"
    elif f == "source":
        inter = set(v) & facts["sources"]
        if inter:
            return f"来源命中 {sorted(inter)}"
    elif f == "job":
        if v in facts["job_ids"]:
            return "申请过该岗位"
    elif f == "text":
        hit = v.lower() in facts["text"]
        if op == "contains" and hit:
            return f"简历全文含「{v}」"
        if op == "not_contains" and not hit:
            return f"简历全文不含「{v}」"
    return None


def evaluate(
    db: Session, cand: Candidate, rule: dict[str, Any]
) -> tuple[bool, list[str]]:
    """评估单候选人。返回 (是否命中, 逐条命中理由)。"""
    facts = _candidate_facts(db, cand)
    mode = rule.get("mode") or "all"
    hits: list[str] = []
    for cond in rule.get("conditions") or []:
        reason = _eval_condition(cond, facts)
        if reason:
            hits.append(reason)
        elif mode == "all":
            return False, hits
    if mode == "any":
        return bool(hits), hits
    return True, hits


# ---------------- 入库执行 ----------------


def _auto_groups(db: Session, tenant_id: str) -> list[CandidateGroup]:
    return db.scalars(
        select(CandidateGroup).where(
            CandidateGroup.tenant_id == tenant_id,
            CandidateGroup.auto_rule.is_not(None),
        )
    ).all()


def enroll_candidate(
    db: Session, cand: Candidate, *, groups: list[CandidateGroup] | None = None
) -> list[str]:
    """对单个候选人跑全部自动组,命中即入组(幂等)。返回新入组的 group_id。"""
    if groups is None:
        groups = _auto_groups(db, cand.tenant_id)
    if not groups:
        return []
    existing = set(
        db.scalars(
            select(CandidateGroupMember.group_id).where(
                CandidateGroupMember.candidate_id == cand.id,
                CandidateGroupMember.group_id.in_([g.id for g in groups]),
            )
        ).all()
    )
    added: list[str] = []
    for g in groups:
        if g.id in existing:
            continue
        try:
            matched, _reasons = evaluate(db, cand, g.auto_rule or {})
        except Exception:  # noqa: BLE001 — 单组坏规则不拖垮候选人主链路
            logger.exception("auto_rule 评估失败 group=%s", g.id)
            continue
        if matched:
            db.add(
                CandidateGroupMember(
                    group_id=g.id, candidate_id=cand.id, source="auto",
                    added_by=None,
                )
            )
            added.append(g.id)
    return added


def rescan_group(
    db: Session, group: CandidateGroup, *, limit: int = 2000
) -> dict[str, Any]:
    """手动补扫:对租户全部候选人重放本组规则。返回 {scanned,added,details}。"""
    rule = group.auto_rule
    if not rule:
        raise ValueError("该分组没有自动入库规则")
    cands = db.scalars(
        select(Candidate)
        .where(Candidate.tenant_id == group.tenant_id)
        .order_by(Candidate.created_at.desc())
        .limit(limit)
    ).all()
    existing = set(
        db.scalars(
            select(CandidateGroupMember.candidate_id).where(
                CandidateGroupMember.group_id == group.id
            )
        ).all()
    )
    added: list[dict[str, str]] = []
    now = _utcnow()
    for c in cands:
        if c.id in existing:
            continue
        matched, reasons = evaluate(db, c, rule)
        if not matched:
            continue
        db.add(
            CandidateGroupMember(
                group_id=group.id, candidate_id=c.id, source="auto",
                added_by=None,
            )
        )
        c.last_active_at = now
        added.append(
            {"candidate_id": c.id, "name": c.name, "reasons": ";".join(reasons[:3])}
        )
    if added:
        group.updated_at = now
    return {"scanned": len(cands), "added": added}


def preview_rule(
    db: Session, tenant_id: str, rule: dict[str, Any], *, limit: int = 20
) -> dict[str, Any]:
    """规则试运行:不落库,返回命中样本与总数(供编辑器预览)。"""
    cands = db.scalars(
        select(Candidate)
        .where(Candidate.tenant_id == tenant_id)
        .order_by(Candidate.created_at.desc())
        .limit(500)
    ).all()
    matched: list[dict[str, Any]] = []
    for c in cands:
        ok, reasons = evaluate(db, c, rule)
        if ok:
            matched.append(
                {
                    "candidate_id": c.id,
                    "name": c.name,
                    "reasons": reasons[:3],
                }
            )
    return {"total": len(matched), "scanned": len(cands), "sample": matched[:limit]}
