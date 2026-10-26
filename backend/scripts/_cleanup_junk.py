"""一次性清理:硬删除测试残留员工及其登录账号(含所有依赖数据)。"""
import sys
sys.path.insert(0, "/app")
from app.infra.db import SessionLocal
from sqlalchemy import text

T = "f30ffc6f-eff5-4bbf-9247-d0b1aee68abe"
KEEP = ["卢品超", "UI员工", "UI经理"]

db = SessionLocal()
junk = [r[0] for r in db.execute(text(
    "SELECT id FROM employees WHERE tenant_id=:t AND NOT (name = ANY(:keep))"
), {"t": T, "keep": KEEP}).fetchall()]
users = [r[0] for r in db.execute(text(
    "SELECT user_id FROM employees WHERE id = ANY(:j) AND user_id IS NOT NULL"
), {"j": junk}).fetchall()]
print("junk employees:", len(junk), "users:", len(users))

def run(sql, **kw):
    n = db.execute(text(sql), kw).rowcount
    if n: print(f"  {n:4d} <- {sql.split()[1][:44]}")
    return n

# 1) 解引用
run("UPDATE employees SET reports_to_id=NULL WHERE reports_to_id = ANY(:j)", j=junk)
run("UPDATE users SET delegate_user_id=NULL WHERE delegate_user_id = ANY(:u)", u=users)
for col, tbl in [("approver_id","leave_applications"),("approver_id","overtime_requests"),
                 ("approver_id","attendance_requests"),("approver_id","expense_claims"),
                 ("excused_by_id","attendance_flags"),("assignee_user_id","hr_cases"),
                 ("reviewer_id","appraisals"),("requested_by","job_requisitions"),
                 ("approver_id","job_requisitions"),("owner_user_id","job_applications"),
                 ("author_user_id","insight_notes"),("created_by","automation_rules"),
                 ("created_by","import_jobs"),("interviewer_user_id","interview_feedbacks"),
                 ("created_by","interview_scorecards")]:
    run(f"UPDATE {tbl} SET {col}=NULL WHERE {col} = ANY(:u)", u=users)

# 2) 审批实例(request_id 多态无FK):先收请求id再删 steps→instances
for tbl in ["leave_applications","overtime_requests","attendance_requests","expense_claims","employee_requests"]:
    req_ids = [r[0] for r in db.execute(text(f"SELECT id FROM {tbl} WHERE employee_id = ANY(:j)"), {"j": junk}).fetchall()]
    if req_ids:
        insts = [r[0] for r in db.execute(text("SELECT id FROM approval_instances WHERE request_id = ANY(:r)"), {"r": req_ids}).fetchall()]
        if insts:
            run("DELETE FROM approval_steps WHERE instance_id = ANY(:i)", i=insts)
            run("DELETE FROM approval_instances WHERE id = ANY(:i)", i=insts)

# 3) 员工子表(先孙后子)
run("DELETE FROM expense_items WHERE claim_id IN (SELECT id FROM expense_claims WHERE employee_id = ANY(:j))", j=junk)
for tbl in ["appraisals","attendance_flags","attendance_requests","attendances","employee_checkins",
            "employee_requests","employee_skills","expense_claims","goals","insight_notes",
            "leave_allocations","leave_applications","leave_ledger_entries","lifecycle_events",
            "overtime_requests","project_members","salary_slips","shift_assignments"]:
    run(f"DELETE FROM {tbl} WHERE employee_id = ANY(:j)", j=junk)
run("DELETE FROM hr_cases WHERE requester_employee_id = ANY(:j)", j=junk)
run("UPDATE job_offers SET employee_id=NULL WHERE employee_id = ANY(:j)", j=junk)
run("DELETE FROM employees WHERE id = ANY(:j)", j=junk)

# 4) 用户子表
for tbl in ["ai_watchers","announcement_reads","favorites","notifications","saved_filters",
            "mcp_tokens","idempotency_keys","report_subscriptions"]:
    run(f"DELETE FROM {tbl} WHERE user_id = ANY(:u)", u=users)
run("DELETE FROM users WHERE id = ANY(:u)", u=users)

db.commit()
print("done")
