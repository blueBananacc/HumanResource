"""AI 助手真实使用模拟 —— 真实 LLM 驱动,不是假数据。

容器内执行:  python /app/scripts/seed_assistant_demo.py
- admin / employee 两个视角各跑一串真实对话(SSE),写工具自动过确认门
- 再补几个后端直连 AI 调用(题库生成/AI摘要/纪要提炼)充实模型调用日志
产物:助手会话历史(assistant/data)+ model_call_logs + 真实业务副作用
"""
import json, os, sys, time
sys.path.insert(0, "/app")
import httpx

BASE = os.environ.get("SEED_BASE", "http://localhost:8000")
ADMIN_PW = os.environ.get("SEED_ADMIN_PASSWORD", "admin123456")
EMP_PW = os.environ.get("SEED_EMP_PASSWORD", "emp123456")
CLIENT = httpx.Client(timeout=httpx.Timeout(300, connect=10))

def login(email, pw):
    r = CLIENT.post(f"{BASE}/api/auth/login",
                    json={"email": email, "password": pw})
    r.raise_for_status()
    return r.json()["access_token"]

def confirm(token, call_id, approved=True):
    r = CLIENT.post(f"{BASE}/api/assistant/confirm",
                    json={"call_id": call_id, "approved": approved},
                    headers={"Authorization": f"Bearer {token}"})
    return r.status_code

def chat(token, message, approve=True, tag=""):
    """发一条消息,流式读完;遇 confirm_required 自动放行。返回(文本, 工具列表)"""
    tools, confirms, text_parts, conv_id, err = [], [], [], None, None
    ev, data = None, []
    t0 = time.time()
    with CLIENT.stream(
        "POST", f"{BASE}/api/assistant/chat",
        json={"message": message},
        headers={"Authorization": f"Bearer {token}"},
    ) as resp:
        for line in resp.iter_lines():
            if not line:
                if ev and data:
                    payload = json.loads("".join(data)) if data != ["{}"] else {}
                    if ev == "text_delta":
                        text_parts.append(payload.get("delta", ""))
                    elif ev == "tool_start":
                        tools.append(payload.get("name", "?"))
                    elif ev == "confirm_required":
                        cid = payload.get("call_id")
                        sc = confirm(token, cid, approve)
                        confirms.append((payload.get("name"), approve, sc))
                    elif ev == "conversation":
                        conv_id = payload.get("conversation_id") or payload.get("id")
                    elif ev == "error":
                        err = payload.get("message")
                    ev, data = None, []
                continue
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
    dt = int(time.time() - t0)
    text = "".join(text_parts)
    print(f"\n{'='*64}\n[{tag}] {message}\n  -> {dt}s | tools={tools} confirms={confirms}")
    if err:
        print(f"  !! error: {err}")
    print(f"  回复: {text[:160]}{'...' if len(text) > 160 else ''}")
    return {"msg": message, "tools": tools, "confirms": confirms,
            "seconds": dt, "reply_head": text[:300], "conversation_id": conv_id,
            "error": err}

# ------------------------------------------------------------------ 剧本
ADMIN_SCRIPT = [
    "我现在有几条待审批?都是什么类型的?",
    "生成 2026-10 的薪酬底表",
    "看看员工「张伟」的人才画像,给我个摘要",
    "新项目「移动端 App 重构」需要 TypeScript 和 React Native 强的人,帮我从公司内部找找",
    "帮我统计一下各部门的人数分布",
    "最近有哪些考勤异常?列出来看看",
    "帮我发一条全员公告:本周五 15:00 在大会议室开全员大会,同步 Q4 规划,要求全员参加",
    "给我建一个项目「AI 招聘助手二期」,我是负责人,需要 LLM 应用和系统设计能力",
    "我下个月想休两天年假,帮我看看我的假期余额够不够",
]
EMP_SCRIPT = [
    "我还有几天年假?",
    "我最近的请假申请审批到哪一步了?",
    "帮我申请 11 月 6 号一天年假,家里有事",
    "我这个月工资条发了吗?到手多少?",
    "公司补充医疗险怎么给家属加?",
]

results = []
print("== admin 视角 ==")
adm = login("admin@example.com", ADMIN_PW)
for i, m in enumerate(ADMIN_SCRIPT):
    results.append(chat(adm, m, tag=f"admin#{i+1}"))
    time.sleep(1)

print("\n\n== employee 视角 ==")
emp = login("emp-uie2e@example.com", EMP_PW)
for i, m in enumerate(EMP_SCRIPT):
    results.append(chat(emp, m, tag=f"emp#{i+1}"))
    time.sleep(1)

# ------------------------------------------------------------------ 后端直连 AI(充实模型调用日志)
print("\n\n== 后端直连 AI 调用 ==")
H = {"Authorization": f"Bearer {adm}"}

def call(method, path, **kw):
    try:
        r = CLIENT.request(method, f"{BASE}{path}", headers=H, timeout=180, **kw)
        print(f"  {method} {path} -> {r.status_code}")
        return r
    except Exception as e:
        print(f"  {method} {path} -> ERR {e}")

# 题库 AI 生成(真 LLM,落库 + 调用日志)
call("POST", "/api/question-library/generate",
     json={"category": "前端工程", "kind": "tech", "difficulty": "advanced", "count": 3})
call("POST", "/api/question-library/generate",
     json={"category": "系统设计", "kind": "scenario", "difficulty": "intermediate", "count": 2})

# 实体 AI 摘要(员工详情/候选人详情的 SummaryCard)
from sqlalchemy import select
from app.infra.db import SessionLocal
from app.domain.models import Employee, Candidate, User
db = SessionLocal()
admin_u = db.scalars(select(User).where(User.email == "admin@example.com")).first()
T = str(admin_u.tenant_id)   # 跟随 admin 租户
zhangwei = db.scalars(select(Employee).where(Employee.tenant_id == T, Employee.name == "张伟")).first()
wangfang = db.scalars(select(Employee).where(Employee.tenant_id == T, Employee.name == "王芳")).first()
cand = db.scalars(select(Candidate).where(Candidate.tenant_id == T)).first()
for e in (zhangwei, wangfang):
    if e: call("GET", f"/api/ai/summarize/employees/{e.id}")
if cand: call("GET", f"/api/ai/summarize/candidates/{cand.id}")

# 洞察纪要上传(真 LLM 提炼技能建议 → 待确认队列)
if wangfang:
    call("POST", f"/api/employees/{wangfang.id}/insights",
         json={"kind": "meeting", "title": "审批引擎性能评审纪要",
               "body": ("王芳主导审批引擎 P99 优化复盘:通过实例行锁+步骤快照把裁决延迟从 800ms 降到 220ms;"
                        "她提出把条件求值缓存到策略层,已验证可行。主管点评:可以开始带一名新人。"),
               "visibility": "manager_only", "extract_skills": True})
if zhangwei:
    call("POST", f"/api/employees/{zhangwei.id}/insights",
         json={"kind": "project_review", "title": "移动端重构里程碑复盘",
               "body": ("张伟完成 RN 壳层与原生模块桥接,首屏时间 1.8s→0.9s;"
                        "他写的桥接层类型定义规范被组内采纳为标准。"),
               "visibility": "manager_only", "extract_skills": True})
db.close()

print("\n\n== 汇总 ==")
ok = sum(1 for r in results if not r["error"])
print(f"对话 {ok}/{len(results)} 成功 | 写工具确认 "
      f"{sum(len(r['confirms']) for r in results)} 次")
with open("/tmp/assistant_demo_log.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("transcript -> /tmp/assistant_demo_log.json")
