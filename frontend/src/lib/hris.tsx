/**
 * HRIS 模块共享:后端 schema 对应的前端类型 + 状态文案/配色。
 * 类型与 backend/app/api/* Out 模型一一对应。
 */

export type Employee = {
  id: string;
  employee_no: string;
  name: string;
  org_unit_id: string | null;
  org_unit_name: string | null;
  designation: string | null;
  reports_to_id: string | null;
  reports_to_name: string | null;
  user_id: string | null;
  work_email: string | null;
  phone: string | null;
  hire_date: string | null;
  employment_type: string;
  status: string;
  personal: Record<string, unknown> | null;
  source_candidate_id: string | null;
  has_account: boolean;
  created_at: string;
  updated_at: string;
};

export const EMPLOYMENT_TYPE_LABEL: Record<string, string> = {
  full_time: "全职",
  part_time: "兼职",
  intern: "实习",
  contractor: "外包",
};

export type LeaveType = {
  id: string;
  name: string;
  annual_days: number;
  is_paid: boolean;
  allocated_days: number;
  used_days: number;
};

export type LeaveAllocation = {
  id: string;
  employee_id: string;
  employee_name: string;
  leave_type_id: string;
  leave_type_name: string;
  year: number;
  days: number;
  note: string | null;
  created_at: string;
};

export type LeaveApplication = {
  id: string;
  employee_id: string;
  employee_name: string;
  org_unit_name: string | null;
  leave_type_id: string;
  leave_type_name: string;
  start_date: string;
  end_date: string;
  days: number;
  reason: string;
  status: string;
  approver_id: string | null;
  approver_email: string | null;
  decided_at: string | null;
  decision_note: string | null;
  created_at: string;
};

export type LeaveBalance = {
  allocation_id: string;
  employee_id: string;
  employee_name: string;
  leave_type_id: string;
  leave_type_name: string;
  year: number;
  allocated: number;
  used: number;
  pending: number;
  balance: number;
};

export type LeaveLedgerEntry = {
  id: string;
  employee_id: string;
  employee_name: string;
  leave_type_name: string;
  year: number;
  delta_days: number;
  ref_type: string;
  ref_id: string;
  note: string | null;
  created_at: string;
};

export type Attendance = {
  id: string;
  employee_id: string;
  employee_name: string;
  org_unit_name: string | null;
  day: string;
  status: string;
  first_in: string | null;
  last_out: string | null;
  work_hours: number | null;
  source: string;
};

export type Checkin = {
  id: string;
  employee_id: string;
  employee_name: string;
  ts: string;
  kind: string;
  method: string;
  note: string | null;
};

export type LifecycleTemplate = {
  id: string;
  kind: string;
  name: string;
  tasks: string[];
  is_default: boolean;
};

export type LifecycleTask = {
  id: string;
  title: string;
  done: boolean;
  done_at: string | null;
  sort: number;
};

export type LifecycleEvent = {
  id: string;
  employee_id: string;
  employee_name: string;
  kind: string;
  status: string;
  effective_date: string | null;
  payload: Record<string, unknown> | null;
  note: string | null;
  tasks: LifecycleTask[];
  created_at: string;
  completed_at: string | null;
};

export type Requisition = {
  id: string;
  title: string;
  org_unit_id: string | null;
  org_unit_name: string | null;
  headcount: number;
  level: string;
  reason: string;
  status: string;
  requested_by: string | null;
  requested_by_email: string | null;
  approver_id: string | null;
  approver_email: string | null;
  decided_at: string | null;
  decision_note: string | null;
  job_id: string | null;
  job_title: string | null;
  created_at: string;
};

export type Offer = {
  id: string;
  candidate_id: string;
  candidate_name: string;
  job_id: string | null;
  job_title: string | null;
  position: string | null;
  salary: Record<string, unknown> | null;
  start_date: string | null;
  letter: string | null;
  status: string;
  sent_at: string | null;
  responded_at: string | null;
  employee_id: string | null;
  created_at: string;
};

export type Goal = {
  id: string;
  employee_id: string;
  employee_name: string;
  title: string;
  description: string;
  due_date: string | null;
  progress: number;
  status: string;
};

export type AppraisalCycle = {
  id: string;
  name: string;
  period_start: string;
  period_end: string;
  status: string;
  appraisal_count: number;
};

export type Appraisal = {
  id: string;
  cycle_id: string;
  cycle_name: string;
  employee_id: string;
  employee_name: string;
  org_unit_name: string | null;
  status: string;
  self_review: string | null;
  self_rating: number | null;
  manager_review: string | null;
  manager_rating: number | null;
  updated_at: string;
};

export type ExpenseType = {
  id: string;
  name: string;
};

export type ExpenseItem = {
  id: string;
  type_id: string | null;
  type_name: string | null;
  occurred_on: string;
  amount: number;
  note: string | null;
};

export type ExpenseClaim = {
  id: string;
  employee_id: string;
  employee_name: string;
  org_unit_name: string | null;
  title: string;
  total: number;
  status: string;
  items: ExpenseItem[];
  approver_id: string | null;
  decided_at: string | null;
  decision_note: string | null;
  created_at: string;
};

export type SlipItem = {
  label: string;
  kind: string;
  amount: number;
};

export type SalarySlip = {
  id: string;
  employee_id: string;
  employee_name: string;
  org_unit_name: string | null;
  period: string;
  gross: number;
  deductions: number;
  net: number;
  currency: string;
  items: SlipItem[] | null;
  status: string;
  issued_at: string | null;
  note: string | null;
  created_at: string;
};

export type OrgTreeNode = {
  id: string;
  name: string;
  kind?: string;
  children?: OrgTreeNode[];
};

export type OrgChartNode = {
  id: string;
  name: string;
  designation: string | null;
  org_unit_name: string | null;
  status: string;
  children: OrgChartNode[];
};

// ── /api/me/* (ESS) 类型 ─────────────────────────────────────

export type MeProfile = {
  id: string;
  employee_no: string;
  name: string;
  org_unit_name: string | null;
  designation: string | null;
  reports_to_name: string | null;
  work_email: string | null;
  phone: string | null;
  hire_date: string | null;
  employment_type: string;
  status: string;
  personal: Record<string, unknown> | null;
};

export type MeLeaveType = { id: string; name: string; is_paid: boolean };

export type MeBalance = {
  leave_type_id: string;
  leave_type_name: string;
  allocated: number;
  used: number;
  pending: number;
  balance: number;
};

export type MeApplication = {
  id: string;
  leave_type_id: string;
  leave_type_name: string;
  start_date: string;
  end_date: string;
  days: number;
  reason: string;
  status: string;
  decided_at: string | null;
  decision_note: string | null;
  created_at: string;
};

export type MeCheckin = {
  id: string;
  ts: string;
  kind: string;
  method: string;
  note: string | null;
};

export type MeAttendance = {
  day: string;
  status: string;
  first_in: string | null;
  last_out: string | null;
  work_hours: number | null;
};

export type MeSlip = {
  id: string;
  period: string;
  gross: number;
  deductions: number;
  net: number;
  currency: string;
  items: SlipItem[] | null;
  issued_at: string | null;
  note: string | null;
};

export type MeClaimItem = {
  id: string;
  type_id: string | null;
  type_name: string | null;
  occurred_on: string;
  amount: number;
  note: string | null;
};

export type MeClaim = {
  id: string;
  title: string;
  total: number;
  status: string;
  items: MeClaimItem[];
  decided_at: string | null;
  decision_note: string | null;
  created_at: string;
};

export type MeGoal = {
  id: string;
  title: string;
  description: string;
  due_date: string | null;
  progress: number;
  status: string;
};

export type MeAppraisal = {
  id: string;
  cycle_id: string;
  cycle_name: string;
  status: string;
  self_review: string | null;
  self_rating: number | null;
  manager_review: string | null;
  manager_rating: number | null;
};

export type MeEvent = {
  id: string;
  kind: string;
  status: string;
  effective_date: string | null;
  note: string | null;
  tasks: { id: string; title: string; done: boolean; done_at: string | null; sort: number }[];
};

// ── 状态文案/配色 ────────────────────────────────────────────

const STATUS_LABEL: Record<string, string> = {
  // 通用单据
  draft: "草稿",
  open: "待审批",
  pending: "待审批",
  approved: "已批准",
  rejected: "已驳回",
  cancelled: "已取消",
  paid: "已支付",
  issued: "已发放",
  // 员工在职
  active: "在职",
  leave: "休假中",
  suspended: "停职",
  terminated: "已离职",
  // 考勤
  present: "出勤",
  absent: "缺勤",
  remote: "远程",
  holiday: "节假日",
  // 打卡
  in: "上班",
  out: "下班",
  // 生命周期
  in_progress: "进行中",
  done: "已完成",
  // 招聘
  sent: "已发出",
  accepted: "已接受",
  declined: "已谢绝",
  withdrawn: "已撤回",
  converted: "已转岗位",
  // 考核
  self_submitted: "待上级评",
  reviewed: "已完成",
  completed: "已完成",
  closed: "已关闭",
  dropped: "已放弃",
};

const GREEN = "text-[var(--color-success)] bg-[var(--color-success-soft)]";
const RED = "text-[var(--color-danger)] bg-[var(--color-danger-soft)]";
const GRAY = "text-[var(--color-text-tertiary)] bg-[var(--color-bg-subtle)]";
const AMBER = "text-[var(--color-warning-text)] bg-[var(--color-warning-soft)]";
const BLUE = "text-[var(--color-info)] bg-[var(--color-info-soft)]";

const STATUS_TONE: Record<string, string> = {
  approved: GREEN,
  accepted: GREEN,
  issued: GREEN,
  paid: GREEN,
  present: GREEN,
  done: GREEN,
  completed: GREEN,
  reviewed: GREEN,
  active: GREEN,
  in: GREEN,
  rejected: RED,
  absent: RED,
  declined: RED,
  terminated: RED,
  cancelled: GRAY,
  withdrawn: GRAY,
  suspended: GRAY,
  closed: GRAY,
  dropped: GRAY,
  leave: BLUE,
  remote: BLUE,
  sent: BLUE,
  converted: BLUE,
  pending: AMBER,
  self_submitted: AMBER,
};

export const LIFECYCLE_KIND_LABEL: Record<string, string> = {
  onboarding: "入职",
  transfer: "调动",
  promotion: "晋升",
  separation: "离职",
};

export function statusLabel(s: string): string {
  return STATUS_LABEL[s] ?? s;
}

export function statusTone(s: string): string {
  return STATUS_TONE[s] ?? AMBER;
}

/** 小状态胶囊 — 所有 HRIS 页面共用。``label`` 可覆盖文案(如目标 active → 进行中)。 */
export function StatusPill({ status, label }: { status: string; label?: string }) {
  return (
    <span
      className={`px-2 py-0.5 rounded-md text-[11px] font-medium font-body whitespace-nowrap ${statusTone(status)}`}
    >
      {label ?? statusLabel(status)}
    </span>
  );
}

export function fmtMoney(n: number, currency = "CNY") {
  return `${n.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${currency}`;
}

export function fmtTime(iso: string | null) {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}
