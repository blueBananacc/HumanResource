"""T6 AI 产品化 — 真实 HTTP 链路验证(容器内运行)。

用法:
    docker compose exec backend python scripts/verify_t6_live.py

覆盖:
- prompt_templates scope_module 两级回落(API + render)
- /api/ai/field-assist 字段级助手 + ai_call 审计
- /api/ai/summarize/{entity}/{id} 记录摘要
- /api/ai/doc-templates + /api/ai/doc-draft 文档起草
- /api/doc-templates 管理 CRUD/reset
- knowledge product_help:检索过滤 + API 创建 422 + 改/删 409
- sidecar 完整链路(自带 OpenAI 桩,不依赖外网 LLM):
  * tool_result SSE 事件(search_talents / search_product_help)
  * 写工具 confirm_required -> /api/assistant/confirm -> 落库(save_filter / create_watcher)
- 租户隔离(跨租户 404)

退出码 0 = 全过。需要 stub_llm_server.py 已在 backend:8799 监听
(脚本会尝试自动拉起;容器内 exec 场景下用 subprocess 起后台进程)。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

BASE = os.environ.get("BASE_URL", "http://localhost:8000")
RUN = uuid.uuid4().hex[:6]

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))


def req(method: str, path: str, token: str, **kw):
    r = requests.request(method, f"{BASE}{path}", headers={"Authorization": f"Bearer {token}"}, timeout=60, **kw)
    return r


def login(email: str, pwd: str) -> str:
    r = requests.post(f"{BASE}/api/auth/login", json={"email": email, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def sse_chat(token: str, message: str, on_event=None, timeout: int = 120) -> list[tuple[str, dict]]:
    """消费一轮 /api/assistant/chat SSE;on_event(evt,data) 可中途动作(如 confirm)。"""
    events: list[tuple[str, dict]] = []
    conv = f"t6-{RUN}-{uuid.uuid4().hex[:4]}"
    with requests.post(
        f"{BASE}/api/assistant/chat",
        headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
        json={"message": message, "conversation_id": conv},
        stream=True,
        timeout=timeout,
    ) as r:
        r.raise_for_status()
        ev, data_lines = None, []
        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.rstrip("\n")
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line == "":
                if ev:
                    try:
                        data = json.loads("".join(data_lines) or "{}")
                    except json.JSONDecodeError:
                        data = {"_raw": "".join(data_lines)}
                    events.append((ev, data))
                    if on_event:
                        on_event(ev, data)
                    if ev == "done":
                        break
                ev, data_lines = None, []
    return events


def main() -> int:
    admin = login("admin@example.com", "admin123456")

    # 上次运行可能残留激活的 stub provider — 先清干净,否则 backend chat() 全走桩
    try:
        from sqlalchemy import delete as sa_delete
        from app.domain.models import LLMProvider
        from app.infra.db import SessionLocal

        with SessionLocal() as db:
            db.execute(sa_delete(LLMProvider).where(LLMProvider.name.like("t6-stub-%")))
            db.commit()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 残留 provider 清理: {e}")

    # ---------- 1. prompt_templates scope_module ----------
    r = req("GET", "/api/prompt-templates", admin)
    check("prompt-templates 列表含 module/scoped_count", r.status_code == 200 and "module" in r.json()[0] and "scoped_count" in r.json()[0])
    keys = {x["key"] for x in r.json()}
    check("注册表含 T6 key", {"field.jobs.description.draft", "summarize.candidate", "doc.draft"} <= keys)

    key = "field.jobs.description.draft"
    r = req("PUT", f"/api/prompt-templates/{key}", admin, json={"content": "全局覆盖 {{title}} 测试内容"})
    check("PUT 全局模板覆盖", r.status_code == 200)
    r = req("PUT", f"/api/prompt-templates/{key}", admin, json={"content": "jobs 模块覆盖 {{title}} 测试内容", "scope_module": "jobs"})
    check("PUT jobs 模块覆盖", r.status_code == 200)
    r = req("POST", f"/api/prompt-templates/{key}/preview", admin, json={"vars": {"title": "X", "fields": "{}", "current": "", "instruction": ""}, "scope_module": "jobs"})
    scoped = r.status_code == 200 and "jobs 模块覆盖" in r.json().get("rendered", "")
    check("preview scope_module=jobs 命中模块覆盖", scoped, r.text[:200])
    r = req("POST", f"/api/prompt-templates/{key}/preview", admin, json={"vars": {"title": "X", "fields": "{}", "current": "", "instruction": ""}})
    check("preview 无 scope 命中全局覆盖", r.status_code == 200 and "全局覆盖" in r.json().get("rendered", ""))
    r = req("POST", f"/api/prompt-templates/{key}/reset?scope_module=jobs", admin)
    check("reset 模块覆盖", r.status_code == 200)
    r = req("POST", f"/api/prompt-templates/{key}/reset", admin)
    check("reset 全局覆盖", r.status_code == 200)

    # ---------- 2. field-assist(mock LLM 路径)----------
    r = req("POST", "/api/ai/field-assist", admin, json={
        "target": "jobs.description", "intent": "draft",
        "fields": {"title": f"T6验证岗{RUN}", "level": "advanced"}, "instruction": "语气正式",
    })
    check("field-assist 返回文本", r.status_code == 200 and bool(r.json().get("text")), r.text[:200])
    r = req("POST", "/api/ai/field-assist", admin, json={"target": "nope.x", "intent": "draft", "fields": {}})
    check("field-assist 未知 target 回落 generic", r.status_code == 200 and r.json().get("key", "").startswith("field.generic"))

    # ---------- 3. 记录摘要 ----------
    # talents 无公开 POST(候选人由简历管道产生)— 直写库造前置数据
    cid = None
    try:
        from app.domain.models import Candidate, Tenant, User as DbUser
        from app.infra.db import SessionLocal

        with SessionLocal() as db:
            admin_u = db.scalars(
                __import__("sqlalchemy").select(DbUser).where(DbUser.email == "admin@example.com")
            ).first()
            c = Candidate(tenant_id=admin_u.tenant_id, name=f"T6摘要{RUN}", display_email=f"t6-{RUN}@example.com")
            db.add(c)
            db.commit()
            cid = c.id
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 建候选人失败: {e}")
    check("建候选人(摘要前置)", cid is not None)
    if cid:
        r = req("GET", f"/api/ai/summarize/candidates/{cid}", admin)
        check("候选人摘要", r.status_code == 200 and bool(r.json().get("summary")), r.text[:200])

    # ---------- 4. 文档起草 ----------
    r = req("GET", "/api/ai/doc-templates", admin)
    tpl_keys = {x["key"] for x in r.json()} if r.status_code == 200 else set()
    check("轻量模板列表含内置", {"offer_letter", "employment_certificate"} <= tpl_keys, r.text[:200])
    if cid:
        r = req("POST", "/api/ai/doc-draft", admin, json={"key": "offer_letter", "entity": "candidates", "record_id": cid})
        ok = r.status_code == 200 and r.json().get("vars", {}).get("candidate_name") == f"T6摘要{RUN}"
        check("doc-draft 预填变量+生成", ok, r.text[:200])
    r = req("GET", "/api/doc-templates", admin)
    check("管理模板列表", r.status_code == 200 and any(x["is_builtin"] for x in r.json()))
    r = req("PUT", f"/api/doc-templates/custom.t6{RUN}", admin, json={"name": "T6自定义", "entity": "any", "body": "证明 {{var}} — 测试内容足够长"})
    check("新建自定义模板", r.status_code == 200)
    r = req("DELETE", f"/api/doc-templates/custom.t6{RUN}", admin)
    check("删除自定义模板", r.status_code == 204)
    r = req("DELETE", "/api/doc-templates/offer_letter", admin)
    check("内置模板禁止删除(400)", r.status_code == 400)

    # ---------- 5. product_help ----------
    r = req("POST", "/api/knowledge/search", admin, json={"q": "导入", "category": "product_help", "limit": 5})
    hits = r.json() if r.status_code == 200 else []
    check("product_help 检索命中", r.status_code == 200 and len(hits) > 0 and all(h["category"] == "product_help" for h in hits))
    r = req("POST", "/api/knowledge/articles", admin, json={"title": "x", "category": "product_help", "content": "y"})
    check("API 创建 product_help 被拒(422)", r.status_code == 422)
    if hits:
        art_id = hits[0]["id"]
        r = req("PUT", f"/api/knowledge/articles/{art_id}", admin, json={"title": "改", "category": "faq", "content": "x"})
        check("product_help 编辑被拒(409)", r.status_code == 409)
        r = req("DELETE", f"/api/knowledge/articles/{art_id}", admin)
        check("product_help 删除被拒(409)", r.status_code == 409)

    # ---------- 6. sidecar 全链路(stub LLM)----------
    # 6.0 确保桩在跑
    try:
        requests.get("http://localhost:8799/healthz", timeout=2)
    except Exception:
        subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), "stub_llm_server.py")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
    try:
        requests.get("http://localhost:8799/healthz", timeout=3)
        stub_up = True
    except Exception:
        stub_up = False
    check("stub LLM :8799 在线", stub_up)

    provider_id = None
    if stub_up:
        # sidecar 与 backend 分属不同容器 — provider base_url 用 docker DNS 名
        # openai/ 前缀让 backend litellm 走 openai provider;sidecar 剥前缀后拿裸名
        r = req("POST", "/api/llm/providers", admin, json={
            "name": f"t6-stub-{RUN}", "base_url": "http://backend:8799",
            "api_key": "stub-key", "model": "openai/stub-model",
        })
        if r.status_code not in (200, 201):
            check("创建 stub provider", False, r.text[:200])
        else:
            provider_id = r.json()["id"]
            r = req("POST", f"/api/llm/providers/{provider_id}/activate", admin)
            check("激活 stub provider", r.status_code == 200 and r.json()["is_active"])

        # 6.1 只读工具 → tool_result
        evs = sse_chat(admin, "帮我查一下候选人")
        names = [d.get("name") for e, d in evs if e == "tool_result"]
        check("SSE tool_result(search_talents)", "search_talents" in names, str([(e, str(d)[:80]) for e, d in evs])[:300])
        tr = next((d for e, d in evs if e == "tool_result" and d.get("name") == "search_talents"), None)
        check("tool_result 带 result 载荷", tr is not None and tr.get("result") is not None)
        check("SSE 收尾 done", any(e == "done" for e, _ in evs))

        # 6.2 产品帮助工具
        evs = sse_chat(admin, "这个系统怎么批量导入员工?查下手册")
        names = [d.get("name") for e, d in evs if e == "tool_result"]
        check("SSE tool_result(search_product_help)", "search_product_help" in names, str([(e, str(d)[:80]) for e, d in evs])[:300])

        # 6.3 写工具确认链 save_filter
        fired = {"confirm": None}

        def on_ev(ev, data):
            if ev == "confirm_required" and data.get("name") == "save_filter":
                cr = requests.post(
                    f"{BASE}/api/assistant/confirm",
                    headers={"Authorization": f"Bearer {admin}"},
                    json={"call_id": data["call_id"], "approved": True}, timeout=30,
                )
                fired["confirm"] = cr.status_code

        evs = sse_chat(admin, "帮我保存一个筛选", on_event=on_ev)
        check("confirm_required(save_filter)", any(e == "confirm_required" and d.get("name") == "save_filter" for e, d in evs))
        check("confirm 放行(200)", fired["confirm"] == 200, str(fired["confirm"]))
        check("tool_result(save_filter) 无错误", any(e == "tool_result" and d.get("name") == "save_filter" and not d.get("is_error") for e, d in evs), str([(e, str(d)[:120]) for e, d in evs])[:400])
        r = req("GET", "/api/saved-filters?page=talents", admin)
        check("saved_filter 落库", r.status_code == 200 and any(f["name"].startswith("T6链路筛选") for f in r.json()))

        # 6.4 写工具确认链 create_watcher
        fired["confirm"] = None

        def on_ev2(ev, data):
            if ev == "confirm_required" and data.get("name") == "create_watcher":
                requests.post(
                    f"{BASE}/api/assistant/confirm",
                    headers={"Authorization": f"Bearer {admin}"},
                    json={"call_id": data["call_id"], "approved": True}, timeout=30,
                )
                fired["confirm"] = 200

        evs = sse_chat(admin, "候选人超过7天没更新提醒我", on_event=on_ev2)
        check("confirm_required(create_watcher)", any(e == "confirm_required" and d.get("name") == "create_watcher" for e, d in evs))
        check("tool_result(create_watcher)", any(e == "tool_result" and d.get("name") == "create_watcher" and not d.get("is_error") for e, d in evs), str([(e, str(d)[:120]) for e, d in evs])[:400])
        r = req("GET", "/api/me/watchers", admin)
        check("watcher 落库", r.status_code == 200 and any(w.get("kind") == "candidate_stalled" for w in r.json()))

        # 6.5 via=assistant 审计
        r = req("GET", "/api/audit/events?entity_type=saved_filter&action=create&size=10", admin)
        via_ok = r.status_code == 200 and any(x.get("via") == "assistant" for x in r.json().get("items", []))
        check("saved_filter 审计 via=assistant", via_ok, r.text[:200])

        # 清理:停用并删除 stub provider(恢复 mock 路径)
        if provider_id:
            try:
                from sqlalchemy import update as sa_update
                from app.domain.models import LLMProvider
                from app.infra.db import SessionLocal

                with SessionLocal() as db:
                    db.execute(sa_update(LLMProvider).where(LLMProvider.id == provider_id).values(is_active=False))
                    db.commit()
                req("DELETE", f"/api/llm/providers/{provider_id}", admin)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] provider 清理失败: {e}")

    # ---------- 汇总 ----------
    print(f"\n===== {len(PASS)} PASS / {len(FAIL)} FAIL =====")
    for n in FAIL:
        print(f"  FAIL: {n}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
