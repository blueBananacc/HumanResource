import { useState } from "react";
import { Navigate, Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";
import { RightRail } from "./RightRail";
import { AiAssistant } from "./AiAssistant";
import { useAuth } from "@/lib/auth";

const AI_DOCK_KEY = "hronfire.ai.docked";

const TITLE_BY_PATH: Record<string, string> = {
  "/": "概览",
  "/talents": "人才库",
  "/jobs": "岗位",
  "/interviews": "面试",
  "/question-library": "题库",
  "/reports": "报告",
  "/settings/llm": "LLM 配置",
  "/settings/email": "邮箱拉取",
  "/settings/smtp": "SMTP 发件",
  "/settings/license": "License 设置",
  "/settings/team": "团队与组织",
  "/settings/sso": "SSO 接入",
  "/settings/audit": "审计中心",
  "/employees": "员工列表",
  "/org-chart": "组织架构图",
  "/lifecycle": "生命周期",
  "/leave": "请假审批",
  "/leave/balances": "假期额度",
  "/leave/types": "假期类型",
  "/leave/policies": "假期政策",
  "/attendance": "考勤记录",
  "/attendance/checkins": "打卡流水",
  "/attendance/requests": "补卡审批",
  "/attendance/calendar": "节假日历",
  "/overtime": "加班审批",
  "/requisitions": "招聘需求审批",
  "/offers": "Offer 管理",
  "/performance": "目标管理",
  "/performance/cycles": "绩效考核",
  "/expenses": "报销审批",
  "/expenses/types": "报销类型",
  "/payroll": "工资条",
  "/announcements": "公司公告",
  "/requests": "员工申请",
  "/cases": "HR 工单",
  "/knowledge": "知识库",
  "/salary-bands": "薪酬带宽",
  "/campaigns": "人才再激活",
  "/me": "我的工作台",
  "/me/leave": "我的请假",
  "/me/attendance": "我的考勤",
  "/me/payslips": "我的工资条",
  "/me/expenses": "我的报销",
  "/me/goals": "我的目标",
  "/me/requests": "我的申请",
  "/me/cases": "我的工单",
  "/me/profile": "我的资料",
};

/**
 * 受保护的应用骨架。
 * - 未登录 → 跳 /login(并记录原路径以便登录后回跳)
 * - 加载中 → 全屏占位提示
 * - 已登录 → sidebar + topbar + <Outlet />
 */
export function AppShell() {
  const { user, loading } = useAuth();
  const loc = useLocation();
  const [aiOpen, setAiOpen] = useState(false);
  const [aiDocked, setAiDocked] = useState(() => localStorage.getItem(AI_DOCK_KEY) === "1");
  const changeDock = (v: boolean) => {
    localStorage.setItem(AI_DOCK_KEY, v ? "1" : "0");
    setAiDocked(v);
  };

  if (loading) {
    return (
      <div className="h-full grid place-items-center text-sm text-[var(--color-text-tertiary)] font-body">
        加载中…
      </div>
    );
  }
  if (!user) {
    return <Navigate to="/login" state={{ from: loc.pathname }} replace />;
  }
  // 员工自助账号只能待在 /me 区域 — 其他页面后端也会 403,前端直接挡
  if (user.role === "employee" && !loc.pathname.startsWith("/me")) {
    return <Navigate to="/me" replace />;
  }
  // 非员工角色若绑定了员工档案(用人经理等)也可进 /me 自助区 ——
  // 后端 /api/me 同样只对绑定账号开放;未绑定的访问回首页。
  if (
    user.role !== "employee" &&
    loc.pathname.startsWith("/me") &&
    !user.employee_id
  ) {
    return <Navigate to="/" replace />;
  }

  // 面试详情/报告也归到"面试"标签下,不为每个 id 写一行
  let title = TITLE_BY_PATH[loc.pathname] ?? "HROnfire";
  if (loc.pathname.startsWith("/interviews/")) {
    title = loc.pathname.endsWith("/report") ? "面试报告" : "面试进行中";
  } else if (loc.pathname.startsWith("/jobs/")) {
    // /jobs/:id → 岗位详情;/jobs/:id/matches → 岗位匹配候选人
    title = loc.pathname.endsWith("/matches") ? "岗位匹配" : "岗位详情";
  } else if (loc.pathname.startsWith("/employees/")) {
    title = "员工详情";
  } else if (/^\/talents\/[^/]+\/resumes\/[^/]+$/.test(loc.pathname)) {
    title = "简历详情";
  } else if (loc.pathname.startsWith("/talents/")) {
    title = "候选人档案";
  } else if (/^\/question-library\/(sets|categories)\//.test(loc.pathname)) {
    title = "题集详情";
  }
  return (
    <div className="flex flex-col h-full">
      <Topbar title={title} />
      <div className="flex-1 flex min-h-0">
        <Sidebar />
        <div className="flex-1 overflow-auto min-w-0">
          <Outlet />
        </div>
        <RightRail aiOpen={aiOpen} onToggleAi={() => setAiOpen((v) => !v)} />
        <AiAssistant
          open={aiOpen}
          docked={aiDocked}
          onClose={() => setAiOpen(false)}
          onDockChange={changeDock}
        />
      </div>
    </div>
  );
}
