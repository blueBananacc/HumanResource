export type AppRole =
  | "admin"
  | "hr"
  | "interviewer"
  | "hiring_manager"
  | "viewer"
  | "employee";

export const ROLE_LABEL: Record<AppRole, string> = {
  admin: "管理员",
  hr: "HR",
  interviewer: "面试官",
  hiring_manager: "用人经理",
  viewer: "只读",
  employee: "员工",
};

export const ROLE_HINT: Record<AppRole, string> = {
  admin: "完全权限",
  hr: "简历 / 面试 / 报告 / 员工管理",
  interviewer: "参与面试与查看相关结果",
  hiring_manager: "查看岗位与候选人结论 / 审批下属",
  viewer: "仅查看",
  employee: "员工自助(我的请假/考勤/工资条)",
};

export function formatRoleLabel(role: AppRole | string | null | undefined) {
  if (!role) return "";
  return ROLE_LABEL[role as AppRole] ?? role;
}

/**
 * 权限项 —— 与后端 :mod:`app.services.permissions` 的 ``PERM_*`` 常量 1:1 对应。
 * 改动时记得两端同步。
 */
export const PERM = {
  MANAGE_ORG: "manage_org",
  MANAGE_TEAM: "manage_team",
  MANAGE_SETTINGS: "manage_settings",
  WRITE_JOBS: "write_jobs",
  DELETE_JOBS: "delete_jobs",
  WRITE_RESUMES: "write_resumes",
  DELETE_RESUMES: "delete_resumes",
  WRITE_INTERVIEW: "write_interview",
  OVERRIDE_SCORE: "override_score",
  VIEW_REPORTS: "view_reports",
  EXPORT_DATA: "export_data",
  MANAGE_EMPLOYEES: "manage_employees",
  MANAGE_LEAVE: "manage_leave",
  APPROVE_LEAVE: "approve_leave",
  MANAGE_ATTENDANCE: "manage_attendance",
  MANAGE_LIFECYCLE: "manage_lifecycle",
  WRITE_REQUISITIONS: "write_requisitions",
  APPROVE_REQUISITIONS: "approve_requisitions",
  MANAGE_OFFERS: "manage_offers",
  MANAGE_PERFORMANCE: "manage_performance",
  MANAGE_EXPENSES: "manage_expenses",
  MANAGE_PAYROLL: "manage_payroll",
} as const;

export type AppPermission = (typeof PERM)[keyof typeof PERM];
