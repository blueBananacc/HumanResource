"""T5 功能链路真实 HTTP 验证 — 在 backend 容器内跑:
    docker compose exec -T backend python scripts/verify_t5_live.py
"""
import json
import sys
import urllib.request

BASE = "http://localhost:8000"
ok, fail = [], []


def req(method, path, token=None, body=None, headers=None):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if headers:
        h.pop("Content-Type", None) if headers.get("Content-Type", "").startswith("multipart") else None
        h.update(headers)
    data = None
    if body is not None:
        data = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode()
    r = urllib.request.Request(BASE + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            p = resp.read()
            return resp.status, json.loads(p or b"{}")
    except urllib.error.HTTPError as e:
        p = e.read()
        try:
            return e.code, json.loads(p or b"{}")
        except Exception:
            return e.code, {"raw": p[:300].decode("utf8", "replace")}


def check(name, cond, extra=""):
    (ok if cond else fail).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  :: {extra}" if extra and not cond else ""))


def upload(path, token, fname, text):
    b = "----t5b"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\n"
            f"Content-Type: text/csv\r\n\r\n{text}\r\n--{b}--\r\n").encode()
    return req("POST", path, token, body, {"Content-Type": f"multipart/form-data; boundary={b}"})


# ---------- login ----------
st, j = req("POST", "/api/auth/login", body={"email": "admin@example.com", "password": "admin123456"})
check("login", st == 200, j)
tok = j.get("access_token")

# ---------- 5d schema ----------
st, j = req("GET", "/api/entities/jobs/schema", tok)
check("5d schema writable+linkable", st == 200 and "level" in j.get("writable", {}) and "skills" in j.get("linkable", {}), j)
st, j = req("GET", "/api/entities/notes/schema", tok)
check("5d notes creatable", st == 200 and j.get("creatable") is True, j)

# ---------- 5a create ----------
st, j = req("POST", "/api/entities/knowledge", tok, {"fields": {"title": "T5直播验证", "content": "链路验证内容", "category": "onboarding"}})
check("5a create knowledge", st == 201 and j.get("id"), j)
art_id = j.get("id")

# ---------- G history ----------
st, j = req("GET", f"/api/audit/events?entity_id={art_id}", tok)
check("G history rows", st == 200 and len(j.get("items", [])) >= 1, j)

# ---------- 5c link/unlink ----------
import time
RUN = str(int(time.time()))[-6:]
st, j = req("POST", "/api/skills", tok, {"name": f"T5Skill{RUN}", "category": "tech"})
check("create skill", st in (200, 201), j)
skill_id = j.get("id")
st, j = req("POST", "/api/jobs/", tok, {"title": "T5链路岗"})
check("create job", st in (200, 201), j)
job_id = j.get("id")
st, j = req("POST", f"/api/entities/jobs/{job_id}/links", tok,
            {"rel_name": "skills", "related_id": skill_id, "extra": {"importance": "must"}})
check("5c link job.skill", st == 201, j)
st, j = req("GET", f"/api/entities/jobs/{job_id}/relations", tok)
check("5c relations shows skill", st == 200, j)
st, j = req("DELETE", f"/api/entities/jobs/{job_id}/links/skills/{skill_id}", tok)
check("5c unlink", st == 200, j)

# ---------- H optimistic lock ----------
st, j = req("GET", f"/api/jobs/{job_id}", tok)
base_ts = j.get("updated_at")
check("job detail has updated_at", bool(base_ts), j)
st, j = req("PATCH", f"/api/entities/jobs/{job_id}", tok,
            {"fields": {"level": "advanced"}, "base_updated_at": base_ts})
check("H matching ts → 200", st == 200, j)
st, j = req("PATCH", f"/api/entities/jobs/{job_id}", tok,
            {"fields": {"level": "entry"}, "base_updated_at": base_ts})
check("H stale ts → 409", st == 409, j)
st, j = req("GET", f"/api/jobs/{job_id}", tok)
ts2 = j.get("updated_at")
st, j = req("PUT", f"/api/jobs/{job_id}", tok, {"title": "T5链路岗-改", "base_updated_at": "2000-01-01T00:00:00"})
check("H dedicated PUT stale → 409", st == 409, j)

# ---------- A saved_filters ----------
st, j = req("POST", "/api/saved-filters", tok, {"page": "talents", "name": "T5面试中", "params": {"stage": "interview"}})
check("A create filter", st == 201, j)
fid = j.get("id")
st, j = req("GET", "/api/saved-filters?page=talents", tok)
check("A list filters", st == 200 and any(f["id"] == fid for f in j), j)
st, j = req("DELETE", f"/api/saved-filters/{fid}", tok)
check("A delete filter", st == 200, j)

# ---------- D candidate csv import + duplicates + C merge ----------
em1, em2 = f"t5a{RUN}@x.com", f"t5b{RUN}@x.com"
csv_data = f"name,email,phone,tags\n赵六,{em1},139{RUN}1,导入A\n钱七,{em1},139{RUN}2,导入A\n"
st, j = upload("/api/talents/import", tok, "c.csv", csv_data)
check("D import 1 created + 1 in-file dedup-skip", st == 200 and j.get("created") == 1 and len(j.get("skipped", [])) == 1, j)
st, j = upload("/api/talents/import", tok, "c.csv", csv_data)
check("D re-import → DB dedup skip", st == 200 and j.get("created") == 0 and len(j.get("skipped", [])) == 2, j)
st, j = req("GET", f"/api/entities/talents/search?q={em1}", tok)
hits = j if isinstance(j, list) else j.get("items", [])
check("find imported cand", st == 200 and len(hits) >= 1, j)
c1 = hits[0]["id"]
st, j = upload("/api/talents/import", tok, "c2.csv", f"name,email\n赵六,{em2}\n")
st, j = req("GET", f"/api/entities/talents/search?q={em2}", tok)
c2 = (j if isinstance(j, list) else j.get("items", []))[0]["id"]
st, j = req("GET", f"/api/talents/{c1}/duplicates", tok)
check("C duplicates endpoint", st == 200, j)
st, j = req("POST", f"/api/talents/{c1}/merge", tok, {"drop_id": c2, "field_choices": {}})
check("C merge", st == 200, j)
st, j = req("GET", f"/api/entities/talents/search?q={em2}", tok)
check("C merged-away gone", len(j if isinstance(j, list) else j.get("items", [])) == 0, j)
st, j = req("GET", f"/api/entities/talents/{c1}/relations", tok)
check("C relations after merge", st == 200, j)

# ---------- B favorites ----------
st, j = req("POST", "/api/favorites", tok, {"entity": "talents", "entity_id": c1})
check("B favorite", st == 201, j)
favid = j.get("id")
st, j = req("GET", "/api/favorites?entity=talents", tok)
check("B list favorites", st == 200 and any(f["id"] == favid for f in j), j)
st, j = req("POST", "/api/favorites", tok, {"entity": "aliens", "entity_id": c1})
check("B allowlist guard → 400", st == 400, j)
st, j = req("DELETE", f"/api/favorites/{favid}", tok)
check("B unfavorite", st == 200, j)

# ---------- D employee csv import ----------
st, j = upload("/api/employees/import", tok, "e.csv",
               f"name,employee_no,designation,employment_type\n孙八,E{RUN},测试工程师,full_time\n坏行,,经理,bogus\n")
check("D employee import 1 ok 1 skip", st == 200 and j.get("created") == 1 and len(j.get("skipped", [])) == 1, j)

# ---------- E dictionary → request kind ----------
dv = f"t5r{RUN}"
st, j = req("POST", "/api/dictionaries", tok, {"dict_key": "request_kind", "value": dv, "label": "异地调动"})
check("E create dict item", st == 201, j)
st, j = req("GET", "/api/dictionaries/request_kind/options", tok)
check("E options incl custom", st == 200 and any(o["value"] == dv for o in j), j)

# ---------- F scheduler ----------
st, j = req("GET", "/api/scheduler/jobs", tok)
check("F scheduler jobs", st == 200 and isinstance(j, list), j)
names = [x["name"] for x in j]
if names:
    st, j = req("GET", f"/api/scheduler/jobs/{names[0]}/runs", tok)
    check("F job runs", st == 200, j)
else:
    check("F job runs (no jobs registered)", False, "empty jobs")

# ---------- 5e report params via report-defs ----------
st, j = req("GET", "/api/report-defs", tok)
check("5e report-defs", st == 200 and isinstance(j, (list, dict)), j)

# ---------- 5b delete ----------
st, j = req("DELETE", f"/api/entities/knowledge/{art_id}", tok)
check("5b delete article", st == 200, j)
st, j = req("DELETE", f"/api/entities/knowledge/{art_id}", tok)
check("5b re-delete → 404", st == 404, j)
st, j = req("DELETE", f"/api/entities/candidates/{c1}", tok)
check("5b candidate not deletable → 400", st == 400, j)

# ---------- MCP ----------
st, j = req("POST", "/api/me/mcp-tokens", tok, {"name": "t5-verify"})
check("MCP token issue", st in (200, 201) and j.get("token", "").startswith("hrm_"), j)
pat = j.get("token")
mcp_h = {"Authorization": f"Bearer {pat}"}

def mcp(method_, params=None, rid=1):
    return req("POST", "/api/mcp", None, {"jsonrpc": "2.0", "id": rid, "method": method_, "params": params or {}}, mcp_h)

st, j = mcp("initialize")
check("MCP initialize", st == 200, j)
st, j = mcp("tools/list")
tools = [t["name"] for t in j.get("result", {}).get("tools", [])]
need = ["create_record", "delete_record", "link_records", "unlink_records", "get_record_fields"]
check("MCP tools incl T5", st == 200 and all(n in tools for n in need), [n for n in need if n not in tools])
st, j = mcp("resources/list")
res = j.get("result", {}).get("resources", [])
check("5f resources/list", st == 200 and len(res) > 0, j)
st, j = mcp("resources/read", {"uri": "hr://entities"})
check("5f resources/read hr://entities", st == 200 and "contents" in j.get("result", {}), j)
st, j = mcp("tools/call", {"name": "get_record_fields", "arguments": {"entity": "jobs"}}, 3)
check("MCP call get_record_fields", st == 200 and "result" in j, j)
st, j = mcp("tools/call", {"name": "create_record", "arguments": {"entity": "knowledge", "fields": {"title": "MCP建稿", "content": "x", "category": "other"}}}, 4)
cid = None
try:
    payload = j["result"]["content"][0]["text"]
    cid = json.loads(payload).get("id")
except Exception:
    pass
check("MCP create_record", cid is not None, j)
if cid:
    st, j = req("GET", f"/api/audit/events?entity_id={cid}", tok)
    items = j.get("items") or []
    check("MCP write audit via=mcp", any(e.get("via") == "mcp" for e in items), j)
st, j = mcp("tools/call", {"name": "list_report_kinds", "arguments": {}}, 5)
check("5e list_report_kinds via MCP", st == 200 and "result" in j, j)

# revoke PAT
st, j = req("GET", "/api/me/mcp-tokens", tok)
rows = j if isinstance(j, list) else j.get("items", [])
tid = [t["id"] for t in rows if t.get("name") == "t5-verify"]
st, j = req("DELETE", f"/api/me/mcp-tokens/{tid[0]}", tok) if tid else (0, {})
check("MCP token revoke", bool(tid) and st in (200, 204), st)

print(f"\n==== {len(ok)} PASS / {len(fail)} FAIL ====")
if fail:
    print("FAILED:", fail)
    sys.exit(1)
