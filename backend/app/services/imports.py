"""AI 导入向导引擎(T8)。

四步:上传解析 → AI 列映射(启发式兜底)→ 校验 → 执行+错误文件。

- ``IMPORT_DATASETS``:数据集注册表,声明字段 spec/校验/写库逻辑
- ``parse_file``:csv(stdlib)/xlsx(openpyxl)统一解析为 文本列→值 行集
- ``suggest_mapping``:启发式(normalize+别名+difflib);``ai_suggest`` 在其上
  叠加 LLM 结果,服务端校验防幻觉列名与重复占用,LLM 不可用自动降级
- ``check_rows``:apply_mapping → 逐行校验 → errors/will_create;可选 AI 归因
- ``commit``:数据集专属写库(沿用旧 /import 去重规则),另建缺失组织
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import get_close_matches
from typing import Any, Callable

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.domain.models import Candidate, Employee, ImportJob, OrgUnit
from app.services.candidates import _hash as _contact_hash

logger = logging.getLogger(__name__)

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 2000


class UploadError(Exception):
    """文件无法解析;message 面向用户。"""


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def _clean(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = str(v).strip()
    return s or None


def _parse_csv(raw: bytes) -> tuple[list[str], list[dict]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise UploadError("CSV 编码必须是 UTF-8(Excel 另存为 UTF-8 CSV)") from e
    reader = csv.DictReader(io.StringIO(text))
    columns = [(c or "").strip() for c in (reader.fieldnames or [])]
    columns = [c for c in columns if c and not c.startswith("Unnamed:")]
    rows = []
    for rec in reader:
        row = {k.strip(): _clean(v) for k, v in rec.items() if k and k.strip() in columns}
        if any(v is not None for v in row.values()):
            rows.append(row)
    return columns, rows


def _parse_xlsx(raw: bytes) -> tuple[list[str], list[dict]]:
    try:
        import openpyxl
    except ImportError as e:
        raise UploadError("服务器未安装 openpyxl,暂不支持 xlsx,请转 CSV") from e
    try:
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        ws = wb.worksheets[0]
        it = ws.iter_rows(values_only=True)
        header = None
        for row in it:
            if any(v is not None and str(v).strip() for v in row):
                header = [str(v).strip() if v is not None else "" for v in row]
                break
        if not header:
            raise UploadError("Excel 第一个 sheet 为空")
        columns = [c for c in header if c and not c.startswith("Unnamed:")]
        keep = [i for i, c in enumerate(header) if c in columns]
        rows = []
        for row in it:
            rec = {header[i]: _clean(row[i] if i < len(row) else None) for i in keep}
            if any(v is not None for v in rec.values()):
                rows.append(rec)
        wb.close()
        return columns, rows
    except UploadError:
        raise
    except Exception as e:  # noqa: BLE001
        raise UploadError(f"Excel 解析失败: {e}") from e


def parse_file(file_name: str, raw: bytes) -> tuple[list[str], list[dict]]:
    """返回 (列名, 行集);行值为 str|None。"""
    if len(raw) > MAX_BYTES:
        raise UploadError("文件超过 10MB")
    name = (file_name or "").lower()
    if name.endswith(".csv"):
        columns, rows = _parse_csv(raw)
    elif name.endswith((".xlsx", ".xls")):
        columns, rows = _parse_xlsx(raw)
    else:
        raise UploadError("只支持 .csv / .xlsx / .xls")
    if not columns or not rows:
        raise UploadError("文件没有可导入的数据行")
    if len(rows) > MAX_ROWS:
        rows = rows[:MAX_ROWS]
    return columns, rows


def sample_values(rows: list[dict], column: str, count: int = 3) -> list[str]:
    out = []
    for r in rows:
        v = r.get(column)
        if v is not None:
            out.append(str(v))
        if len(out) >= count:
            break
    return out


# ---------------------------------------------------------------------------
# 字段与数据集注册表
# ---------------------------------------------------------------------------

@dataclass
class Field:
    key: str
    label: str
    required: bool = False
    kind: str = "text"  # text/email/tel/date/number/choice/suggest
    aliases: tuple = ()
    choices: tuple = ()          # kind=choice 的合法值
    choice_map: dict = field(default_factory=dict)  # 中文别名→choice 值
    sample: str = ""
    suggest: str = ""            # kind=suggest 的选项来源 key


@dataclass
class DatasetSpec:
    key: str
    label: str
    subtitle: str
    entity_type: str              # audit/automation 实体名
    permission: str
    fields: list[Field]
    validate_row: Callable[[dict, dict], dict]   # (row, ctx) → {field: msg}
    commit: Callable[..., dict]                   # (db, tenant_id, rows) → summary

    def spec_json(self, suggest_options: dict[str, list[str]]) -> dict:
        return {
            "key": self.key, "label": self.label, "subtitle": self.subtitle,
            "fields": [
                {
                    "key": f.key, "label": f.label, "required": f.required,
                    "kind": f.kind, "aliases": list(f.aliases),
                    "choices": list(f.choices), "sample": f.sample,
                    "options": suggest_options.get(f.suggest, []) if f.suggest else [],
                }
                for f in self.fields
            ],
        }


_EMP_TYPES = ("full_time", "part_time", "intern", "contractor")
_EMP_TYPE_CN = {"全职": "full_time", "兼职": "part_time", "实习": "intern",
               "实习生": "intern", "外包": "contractor", "劳务": "contractor"}

_EMP_FIELDS = [
    Field("name", "姓名", required=True,
          aliases=("姓名", "员工姓名", "name", "employee name", "full name", "名称"),
          sample="张三"),
    Field("employee_no", "工号",
          aliases=("工号", "员工编号", "employee no", "emp id", "badge", "编号"),
          sample="EMP0012"),
    Field("work_email", "工作邮箱", kind="email",
          aliases=("工作邮箱", "邮箱", "email", "e-mail", "mail", "电子邮箱"),
          sample="zhangsan@acme.co"),
    Field("phone", "手机", kind="tel",
          aliases=("手机", "电话", "手机号", "联系电话", "phone", "mobile", "tel"),
          sample="13800138000"),
    Field("designation", "职位",
          aliases=("职位", "岗位", "职务", "designation", "title", "job title", "position"),
          sample="后端工程师"),
    Field("org_name", "部门", kind="suggest", suggest="org_units",
          aliases=("部门", "组织", "部门名称", "org", "department", "dept", "org_name",
                   "所属部门"),
          sample="研发部"),
    Field("employment_type", "雇佣类型", kind="choice", choices=_EMP_TYPES,
          choice_map=_EMP_TYPE_CN,
          aliases=("雇佣类型", "员工类型", "employment type", "emp type"),
          sample="full_time"),
    Field("hire_date", "入职日期", kind="date",
          aliases=("入职日期", "入职时间", "hire date", "doj", "date of joining",
                   "start date", "入职日"),
          sample="2024-06-12"),
]

_CAND_FIELDS = [
    Field("name", "姓名", required=True,
          aliases=("姓名", "候选人", "name", "candidate", "full name"),
          sample="李四"),
    Field("email", "邮箱", kind="email",
          aliases=("邮箱", "email", "e-mail", "mail", "电子邮箱", "联系邮箱"),
          sample="lisi@mail.com"),
    Field("phone", "手机", kind="tel",
          aliases=("手机", "电话", "手机号", "phone", "mobile", "tel"),
          sample="13900139000"),
    Field("tags", "标签",
          aliases=("标签", "tags", "tag", "技能标签", "标记"),
          sample="python;后端"),
    Field("org_name", "部门", kind="suggest", suggest="org_units",
          aliases=("部门", "组织", "org", "department", "dept", "org_name"),
          sample="研发部"),
]

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DATE_FMTS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y年%m月%d日", "%Y.%m.%d", "%m/%d/%Y")
_EXCEL_EPOCH = date(1899, 12, 30)


def parse_date_value(v: str) -> date | None:
    """多格式日期解析;Excel 序列天数也接受。"""
    s = (v or "").strip()
    if not s:
        return None
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}.*", s):
        return parse_date_value(s[:10])
    if re.fullmatch(r"\d{5}(\.0)?", s):
        n = int(float(s))
        if 20000 <= n <= 60000:  # 1954..2064 之间的 Excel 序列日期
            return _EXCEL_EPOCH + timedelta(days=n)
    return None


def _base_validate(row: dict, spec: DatasetSpec, ctx: dict) -> dict:
    """通用校验:必填 + kind(email/date/number/choice 含中文别名)。"""
    errs: dict[str, str] = {}
    for f in spec.fields:
        v = row.get(f.key)
        if v is None or str(v).strip() == "":
            if f.required:
                errs[f.key] = f"{f.label} 必填"
            continue
        if f.kind == "email" and not _EMAIL_RE.match(str(v).strip()):
            errs[f.key] = f"{f.label} 不是合法邮箱: {v}"
        elif f.kind == "date" and parse_date_value(str(v)) is None:
            errs[f.key] = f"{f.label} 无法解析为日期: {v}"
        elif f.kind == "number":
            try:
                float(str(v).replace(",", ""))
            except ValueError:
                errs[f.key] = f"{f.label} 不是数字: {v}"
        elif f.kind == "choice":
            s = str(v).strip().lower()
            if s not in f.choices and f.choice_map.get(s) is None \
                    and f.choice_map.get(str(v).strip()) is None:
                errs[f.key] = (
                    f"{f.label} 非法: {v}(可选: "
                    f"{','.join(f.choices)} 或中文别名)"
                )
    return errs


def _normalize_choice(row: dict, spec: DatasetSpec) -> None:
    for f in spec.fields:
        if f.kind == "choice" and row.get(f.key):
            s = str(row[f.key]).strip()
            row[f.key] = f.choice_map.get(s, f.choice_map.get(s.lower(), s.lower()))


def _ctx(db: Session, tenant_id: str) -> dict:
    orgs = db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == tenant_id)
    ).all()
    emp_nos = set(
        db.scalars(
            select(Employee.employee_no).where(Employee.tenant_id == tenant_id)
        ).all()
    )
    return {
        "db": db, "tenant_id": tenant_id,
        "org_by_name": {o.name: o for o in orgs},
        "org_names": sorted(o.name for o in orgs),
        "emp_nos": {n for n in emp_nos if n},
        "cand_hashes": None,  # 惰性:check_rows 按需 IN 查询
    }


def _emp_validate(row: dict, ctx: dict) -> dict:
    return {}


def _cand_validate(row: dict, ctx: dict) -> dict:
    return {}


def _commit_employees(db: Session, tenant_id: str, rows: list[dict]) -> dict:
    from app.api.employees import next_employee_no

    created = 0
    for row in rows:
        emp_no = row.get("employee_no") or next_employee_no(db, tenant_id)
        org_id = ctx_org_id(db, tenant_id, row.get("org_name"))
        hire = parse_date_value(row["hire_date"]) if row.get("hire_date") else None
        db.add(Employee(
            tenant_id=tenant_id,
            employee_no=emp_no,
            name=(row["name"] or "")[:128],
            org_unit_id=org_id,
            designation=(row.get("designation") or None) and row["designation"][:128],
            work_email=row.get("work_email") or None,
            phone=row.get("phone") or None,
            hire_date=hire,
            employment_type=(row.get("employment_type") or "full_time"),
            status="active",
        ))
        created += 1
    return {"created": created}


def _commit_candidates(db: Session, tenant_id: str, rows: list[dict]) -> dict:
    created = 0
    for row in rows:
        email, phone = row.get("email") or None, row.get("phone") or None
        tags = [t.strip() for t in re.split(r"[;,、;]", row.get("tags") or "")
                if t.strip()][:20]
        db.add(Candidate(
            tenant_id=tenant_id,
            name=(row["name"] or "")[:128],
            email_hash=_contact_hash(email, tenant_id),
            phone_hash=_contact_hash(phone, tenant_id),
            display_email=email,
            display_phone=phone,
            tags=tags or None,
            org_unit_id=ctx_org_id(db, tenant_id, row.get("org_name")),
            last_active_at=datetime.utcnow(),
        ))
        created += 1
    return {"created": created}


def ctx_org_id(db: Session, tenant_id: str, name: str | None) -> str | None:
    """组织名→id;不存在则创建(导入向导 will_create 预告过的)。"""
    if not name:
        return None
    org = db.scalar(
        select(OrgUnit).where(
            OrgUnit.tenant_id == tenant_id, OrgUnit.name == name.strip()
        ).limit(1)
    )
    if org is None:
        org = OrgUnit(tenant_id=tenant_id, name=name.strip()[:128], kind="department")
        db.add(org)
        db.flush()
    return org.id


IMPORT_DATASETS: dict[str, DatasetSpec] = {
    "employees": DatasetSpec(
        key="employees", label="员工", subtitle="姓名/工号/部门/入职信息",
        entity_type="employee", permission="manage_employees",
        fields=_EMP_FIELDS, validate_row=_emp_validate, commit=_commit_employees,
    ),
    "candidates": DatasetSpec(
        key="candidates", label="候选人", subtitle="姓名/联系方式/标签",
        entity_type="candidate", permission="write_resumes",
        fields=_CAND_FIELDS, validate_row=_cand_validate, commit=_commit_candidates,
    ),
}


# ---------------------------------------------------------------------------
# 映射:启发式 + AI
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿]", "", str(s).lower())


def heuristic_suggest(columns: list[str], fields: list[Field]) -> dict[str, str]:
    """{field_key: column};未命中为 ''。别名精确命中 → difflib 模糊。"""
    by_norm = {_norm(c): c for c in columns}
    mapping, taken = {}, set()
    for f in fields:
        names = [f.key, f.label, *f.aliases]
        found = next(
            (by_norm[_norm(n)] for n in names if _norm(n) in by_norm), ""
        )
        if not found:
            close = get_close_matches(
                _norm(f.key),
                [n for n in by_norm if by_norm[n] not in taken],
                n=1, cutoff=0.7,
            )
            found = by_norm[close[0]] if close else ""
        if found in taken:
            found = ""
        if found:
            taken.add(found)
        mapping[f.key] = found
    return mapping


def apply_mapping(rows: list[dict], mapping: dict, fields: list[Field]) -> list[dict]:
    """列→字段键的行集;__row 从 2 起对齐表格行号。"""
    keys = [f.key for f in fields]
    out = []
    for i, rec in enumerate(rows, start=2):
        row = {"__row": i}
        for k in keys:
            col = mapping.get(k) or ""
            row[k] = rec.get(col) if col else None
        out.append(row)
    return out


def ai_suggest(
    db: Session, tenant_id: str, spec: DatasetSpec,
    columns: list[str], rows: list[dict],
) -> tuple[dict[str, str], dict, list[str]]:
    """返回 (mapping, meta, warnings)。LLM 为主、启发式兜底;全程不因 LLM
    失败而报错——meta.source 标记每字段来源。"""
    heuristic = heuristic_suggest(columns, spec.fields)
    meta = {
        k: {"confidence": 0.55 if v else 0.0, "source": "heuristic", "reason": ""}
        for k, v in heuristic.items()
    }
    warnings: list[str] = []
    ai_map: dict[str, str] = {}

    from app.infra.license.state import is_feature_enabled
    from app.integrations.llm.provider import LLMError, call_llm_json

    if is_feature_enabled(db, "assistant.agent"):
        try:
            from app.services.prompting import (
                PROMPT_REGISTRY,
                render_prompt,
                render_vars,
            )
            fields_json = json.dumps(
                [{ "key": f.key, "label": f.label, "required": f.required,
                   "kind": f.kind, "aliases": list(f.aliases), "sample": f.sample}
                 for f in spec.fields],
                ensure_ascii=False,
            )
            cols_json = json.dumps(
                [{"column": c, "samples": sample_values(rows, c)} for c in columns],
                ensure_ascii=False,
            )
            vars_ = {"fields": fields_json, "columns": cols_json,
                     "dataset": spec.label}
            prompt = render_prompt(
                db, tenant_id, "import.column_map", module="imports", **vars_
            ) or render_vars(
                PROMPT_REGISTRY["import.column_map"].content, vars_
            )
            data = call_llm_json(
                "import.column_map",
                [{"role": "system", "content": prompt}],
                db=db, tenant_id=tenant_id,
                prompt_key="import.column_map",
            )
            raw_map = data.get("mapping") or {}
            conf = data.get("confidence") or {}
            warnings = [str(w)[:200] for w in (data.get("warnings") or [])][:5]
            taken = set()
            # 按置信度排序去重列占用;校验列名必须真实存在(防幻觉)
            order = sorted(
                raw_map.items(),
                key=lambda kv: float(conf.get(kv[0], 0) or 0), reverse=True,
            )
            for fkey, col in order:
                if fkey not in {f.key for f in spec.fields}:
                    continue
                col = str(col or "").strip()
                if col and col in columns and col not in taken:
                    ai_map[fkey] = col
                    taken.add(col)
                    meta[fkey] = {
                        "confidence": float(conf.get(fkey, 0.8) or 0.8),
                        "source": "ai",
                        "reason": str(data.get("reasons", {}).get(fkey, ""))[:200]
                        if isinstance(data.get("reasons"), dict) else "",
                    }
        except (LLMError, Exception) as e:  # noqa: BLE001 — 降级不报错
            logger.warning("AI 列映射失败,降级启发式: %s", e)
            warnings.append("AI 映射不可用,已用启发式建议")
    # 合并:AI 结果优先,启发式补空缺(不占用已被 AI 使用的列)
    mapping = dict(ai_map)
    used = set(ai_map.values())
    for k, col in heuristic.items():
        if k not in mapping and col and col not in used:
            mapping[k] = col
            used.add(col)
        elif k not in mapping:
            mapping[k] = ""
    for f in spec.fields:
        mapping.setdefault(f.key, "")
    return mapping, meta, warnings


# ---------------------------------------------------------------------------
# 校验 / 执行
# ---------------------------------------------------------------------------

def check_rows(
    db: Session, spec: DatasetSpec, rows: list[dict], ctx: dict
) -> dict:
    """返回 {valid_rows, errors, will_create, valid, error_count}。"""
    valid_rows: list[dict] = []
    errors: list[dict] = []
    seen_nos: set[str] = set()
    seen_hashes: set[str] = set()

    # 候选人去重:先把文件里的 hash 收集起来,一条 IN 查询命中库中已有
    if spec.key == "candidates":
        needed = set()
        for row in rows:
            needed |= {
                h for h in (
                    _contact_hash(row.get("email"), ctx["tenant_id"]),
                    _contact_hash(row.get("phone"), ctx["tenant_id"]),
                ) if h
            }
        ctx["cand_hashes"] = (
            set(
                db.scalars(
                    select(Candidate.email_hash).where(
                        Candidate.tenant_id == ctx["tenant_id"],
                        Candidate.email_hash.in_(needed),
                    )
                ).all()
            )
            | set(
                db.scalars(
                    select(Candidate.phone_hash).where(
                        Candidate.tenant_id == ctx["tenant_id"],
                        Candidate.phone_hash.in_(needed),
                    )
                ).all()
            )
            if needed
            else set()
        )

    for row in rows:
        field_errs = _base_validate(row, spec, ctx)
        field_errs.update(spec.validate_row(row, ctx) or {})
        n = int(row["__row"])

        if spec.key == "employees":
            emp_no = (row.get("employee_no") or "").strip() or None
            if emp_no:
                if emp_no in seen_nos:
                    field_errs["employee_no"] = f"文件内工号重复: {emp_no}"
                elif emp_no in ctx["emp_nos"]:
                    field_errs["employee_no"] = f"工号已存在: {emp_no}"
                else:
                    seen_nos.add(emp_no)
        elif spec.key == "candidates":
            e_hash = _contact_hash(row.get("email"), ctx["tenant_id"])
            p_hash = _contact_hash(row.get("phone"), ctx["tenant_id"])
            hs = {h for h in (e_hash, p_hash) if h}
            if hs & seen_hashes:
                field_errs["email"] = f"文件内重复(同邮箱/手机): {row.get('name')}"
            elif hs & ctx["cand_hashes"]:
                field_errs["email"] = f"库中已存在同邮箱/手机候选人: {row.get('name')}"
            else:
                seen_hashes |= hs

        if field_errs:
            errors.append({
                "row": n,
                "data": {f.key: row.get(f.key) for f in spec.fields},
                "messages": list(field_errs.values()),
                "by_field": field_errs,
            })
        else:
            valid_rows.append(row)

    org_names = {
        (r.get("org_name") or "").strip() for r in valid_rows if r.get("org_name")
    }
    will_create = {
        "orgs": sorted(n for n in org_names if n not in ctx["org_by_name"])
    }
    return {
        "valid_rows": valid_rows, "errors": errors, "will_create": will_create,
        "valid": len(valid_rows), "error_count": len(errors),
    }


def ai_check_note(
    db: Session, tenant_id: str, spec: DatasetSpec, check: dict
) -> str:
    """错误归因摘要(一句中文);LLM 不可用/无错误时返回 ''。"""
    if not check["errors"]:
        return ""
    from app.infra.license.state import is_feature_enabled
    if not is_feature_enabled(db, "assistant.agent"):
        return ""
    try:
        from app.integrations.llm.provider import LLMError, call_llm
        from app.services.prompting import (
            PROMPT_REGISTRY,
            render_prompt,
            render_vars,
        )
        agg: dict[str, dict] = {}
        for e in check["errors"][:200]:
            for fkey, msg in e["by_field"].items():
                a = agg.setdefault(fkey, {"count": 0, "sample": ""})
                a["count"] += 1
                if not a["sample"]:
                    a["sample"] = msg
        vars_ = {"dataset": spec.label,
                 "errors": json.dumps(agg, ensure_ascii=False)}
        prompt = render_prompt(
            db, tenant_id, "import.check_note", module="imports", **vars_
        ) or render_vars(PROMPT_REGISTRY["import.check_note"].content, vars_)
        text = call_llm(
            "import.check_note",
            [{"role": "system", "content": prompt}],
            db=db, tenant_id=tenant_id,
            prompt_key="import.check_note",
        ).strip()
        return text[:500]
    except Exception as e:  # noqa: BLE001
        logger.warning("AI 归因摘要失败: %s", e)
        return ""


def run_import(db: Session, job: ImportJob, spec: DatasetSpec) -> dict:
    """执行:重新解析文件 → 映射 → 校验 → 写 valid 行。"""
    columns, rows = parse_file(job.file_name, job.file_bytes)
    mapped = apply_mapping(rows, job.mapping or {}, spec.fields)
    ctx = _ctx(db, job.tenant_id)
    for row in mapped:
        _normalize_choice(row, spec)
    check = check_rows(db, spec, mapped, ctx)
    summary = spec.commit(db, job.tenant_id, check["valid_rows"])
    summary["skipped"] = 0
    summary["errors"] = check["error_count"]
    summary["will_create"] = check["will_create"]
    return summary


def error_csv(job: ImportJob, spec: DatasetSpec) -> str | None:
    """错误工作簿:字段列 + 错误说明列;check 之后才有。"""
    if not job.check_result or not job.check_result.get("errors"):
        return None
    buf = io.StringIO()
    w = csv.writer(buf)
    keys = ["行号"] + [f.label for f in spec.fields] + ["错误说明"]
    w.writerow(keys)
    for e in job.check_result["errors"]:
        w.writerow(
            [e["row"]]
            + [e["data"].get(f.key) or "" for f in spec.fields]
            + [";".join(e["messages"])]
        )
    return buf.getvalue()
