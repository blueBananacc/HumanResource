import { createBrowserRouter, Navigate } from "react-router-dom";
import { Login } from "@/pages/Login";
import { Dashboard } from "@/pages/Dashboard";
import { LicenseSettings } from "@/pages/LicenseSettings";
import { LLMSettings } from "@/pages/LLMSettings";
import { EmailSettings } from "@/pages/EmailSettings";
import { Jobs } from "@/pages/Jobs";
import { JobDetail } from "@/pages/JobDetail";
import { JobMatches } from "@/pages/JobMatches";
import { ScoringRules } from "@/pages/ScoringRules";
import { Pipeline } from "@/pages/Pipeline";
import { ResumeDetail } from "@/pages/ResumeDetail";
import { Interviews } from "@/pages/Interviews";
import { Schedule } from "@/pages/Schedule";
import { InterviewReport } from "@/pages/InterviewReport";
import { Reports } from "@/pages/Reports";
import { TeamSettings } from "@/pages/TeamSettings";
import { AuditCenter } from "@/pages/AuditCenter";
import { SsoCallback } from "@/pages/SsoCallback";
import { SsoSettings } from "@/pages/SsoSettings";
import { McpTokens } from "@/pages/McpTokens";
import { SkillCenter } from "@/pages/SkillCenter";
import { TalentMap } from "@/pages/TalentMap";
import { Dictionaries } from "@/pages/Dictionaries";
import { Schedulers } from "@/pages/Schedulers";
import { IdempotencyKeys } from "@/pages/IdempotencyKeys";
import { ModelCalls } from "@/pages/ModelCalls";
import { Automations } from "@/pages/Automations";
import { ImportWizard } from "@/pages/ImportWizard";
import { TalentPool } from "@/pages/TalentPool";
import { TalentDetail } from "@/pages/TalentDetail";
import { LegacyQuestionSetRedirect, QuestionSetDetail } from "@/pages/QuestionSetDetail";
import { QuestionLibrary } from "@/pages/QuestionLibrary";
import { CategorySetDetail } from "@/pages/CategorySetDetail";
import { Analytics } from "@/pages/Analytics";
import { InterviewQuality } from "@/pages/InterviewQuality";
import { CandidateInvite } from "@/pages/CandidateInvite";
import { CandidateSession } from "@/pages/CandidateSession";
import { CandidateDone } from "@/pages/CandidateDone";
import { Surveys } from "@/pages/Surveys";
import { SurveyFill } from "@/pages/SurveyFill";
import { SmtpSettings } from "@/pages/SmtpSettings";
import { PromptSettings } from "@/pages/PromptSettings";
import { EmailTemplates } from "@/pages/EmailTemplates";
import { VoiceSettings } from "@/pages/VoiceSettings";
import { AppShell } from "@/components/layout/AppShell";
import { useAuth } from "@/lib/auth";
import type { ReactNode } from "react";
// HRIS 模块(hr.suite)
import { Employees } from "@/pages/hris/Employees";
import { EmployeeDetail } from "@/pages/hris/EmployeeDetail";
import { OrgChart } from "@/pages/hris/OrgChart";
import { Lifecycle } from "@/pages/hris/Lifecycle";
import { LeaveAdmin } from "@/pages/hris/LeaveAdmin";
import { LeavePolicies } from "@/pages/hris/LeavePolicies";
import { OvertimeRequests } from "@/pages/hris/OvertimeRequests";
import { LeaveBalances } from "@/pages/hris/LeaveBalances";
import { LeaveTypes } from "@/pages/hris/LeaveTypes";
import { AttendanceAdmin } from "@/pages/hris/AttendanceAdmin";
import { Checkins } from "@/pages/hris/Checkins";
import { AttendanceRequests } from "@/pages/hris/AttendanceRequests";
import { AttendanceFlagsPage } from "@/pages/hris/AttendanceFlags";
import { WorkCalendarPage } from "@/pages/hris/WorkCalendarPage";
import { ShiftsPage } from "@/pages/hris/Shifts";
import { Requisitions } from "@/pages/hris/Requisitions";
import { Offers } from "@/pages/hris/Offers";
import { Performance } from "@/pages/hris/Performance";
import { AppraisalCycles } from "@/pages/hris/AppraisalCycles";
import { Expenses } from "@/pages/hris/Expenses";
import { ExpenseTypes } from "@/pages/hris/ExpenseTypes";
import { Payroll } from "@/pages/hris/Payroll";
import { Announcements } from "@/pages/hris/Announcements";
import { EmployeeRequests } from "@/pages/hris/EmployeeRequests";
import { Approvals } from "@/pages/hris/Approvals";
import { Projects } from "@/pages/hris/Projects";
import { HrCases } from "@/pages/hris/HrCases";
import { Knowledge } from "@/pages/hris/Knowledge";
import { SalaryBands } from "@/pages/hris/SalaryBands";
import { Campaigns } from "@/pages/hris/Campaigns";
// 员工自助(ESS)
import { MeDashboard } from "@/pages/me/MeDashboard";
import { MyLeave } from "@/pages/me/MyLeave";
import { MyAttendance } from "@/pages/me/MyAttendance";
import { MyPayslips } from "@/pages/me/MyPayslips";
import { MyExpenses } from "@/pages/me/MyExpenses";
import { MyGoals } from "@/pages/me/MyGoals";
import { MyProfile } from "@/pages/me/MyProfile";
import { MyRequests } from "@/pages/me/MyRequests";
import { MyCases } from "@/pages/me/MyCases";

/** employee 角色进 / 时去自助工作台;其他角色看招聘 Dashboard */
function IndexByRole() {
  const { user } = useAuth();
  if (user?.role === "employee") return <Navigate to="/me" replace />;
  return <Dashboard />;
}

/** 员工自助路由守卫:employee 账号,或绑定了员工档案的其他角色(用人经理等)可进 */
function MeGuard({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  if (user && user.role !== "employee" && !user.employee_id) {
    return <Navigate to="/" replace />;
  }
  return <>{children}</>;
}

export const router = createBrowserRouter([
  { path: "/login", element: <Login /> },
  { path: "/login/sso-callback", element: <SsoCallback /> },
  // 候选人侧公开路由 — 无登录态,极简 layout,不挂 AppShell
  { path: "/i/:token", element: <CandidateInvite /> },
  { path: "/i/:token/session", element: <CandidateSession /> },
  { path: "/i/:token/done", element: <CandidateDone /> },
  { path: "/s/:token", element: <SurveyFill /> },
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <IndexByRole /> },
      // ── HRIS 组织 ──
      { path: "employees", element: <Employees /> },
      { path: "employees/:id", element: <EmployeeDetail /> },
      { path: "org-chart", element: <OrgChart /> },
      { path: "lifecycle", element: <Lifecycle /> },
      // ── HRIS 假勤 ──
      { path: "leave", element: <LeaveAdmin /> },
      { path: "leave/balances", element: <LeaveBalances /> },
      { path: "leave/types", element: <LeaveTypes /> },
      { path: "leave/policies", element: <LeavePolicies /> },
      { path: "attendance", element: <AttendanceAdmin /> },
      { path: "attendance/checkins", element: <Checkins /> },
      { path: "attendance/requests", element: <AttendanceRequests /> },
      { path: "attendance/flags", element: <AttendanceFlagsPage /> },
      { path: "attendance/calendar", element: <WorkCalendarPage /> },
      { path: "attendance/shifts", element: <ShiftsPage /> },
      { path: "overtime", element: <OvertimeRequests /> },
      // ── HRIS 招聘扩展 ──
      { path: "requisitions", element: <Requisitions /> },
      { path: "offers", element: <Offers /> },
      // ── HRIS 绩效/费用/薪酬 ──
      { path: "performance", element: <Performance /> },
      { path: "performance/cycles", element: <AppraisalCycles /> },
      { path: "expenses", element: <Expenses /> },
      { path: "expenses/types", element: <ExpenseTypes /> },
      { path: "payroll", element: <Payroll /> },
      { path: "salary-bands", element: <SalaryBands /> },
      // ── HR 服务台 / 协作 ──
      { path: "announcements", element: <Announcements /> },
      { path: "approvals", element: <Approvals /> },
      { path: "projects", element: <Projects /> },
      { path: "requests", element: <EmployeeRequests /> },
      { path: "cases", element: <HrCases /> },
      { path: "knowledge", element: <Knowledge /> },
      { path: "campaigns", element: <Campaigns /> },
      // ── 员工自助(ESS) ──
      { path: "me", element: <MeGuard><MeDashboard /></MeGuard> },
      { path: "me/leave", element: <MeGuard><MyLeave /></MeGuard> },
      { path: "me/attendance", element: <MeGuard><MyAttendance /></MeGuard> },
      { path: "me/payslips", element: <MeGuard><MyPayslips /></MeGuard> },
      { path: "me/expenses", element: <MeGuard><MyExpenses /></MeGuard> },
      { path: "me/goals", element: <MeGuard><MyGoals /></MeGuard> },
      { path: "me/requests", element: <MeGuard><MyRequests /></MeGuard> },
      { path: "me/cases", element: <MeGuard><MyCases /></MeGuard> },
      { path: "me/profile", element: <MeGuard><MyProfile /></MeGuard> },
      // ── 招聘 ──
      { path: "talents", element: <TalentPool /> },
      { path: "talents/:id", element: <TalentDetail /> },
      { path: "talents/:id/resumes/:resumeId", element: <ResumeDetail /> },
      { path: "jobs", element: <Jobs /> },
      { path: "jobs/:id", element: <JobDetail /> },
      { path: "jobs/:id/matches", element: <JobMatches /> },
      { path: "skills", element: <SkillCenter /> },
      { path: "talent-map", element: <TalentMap /> },
      { path: "scoring-rules", element: <ScoringRules /> },
      { path: "pipeline", element: <Pipeline /> },
      { path: "interviews", element: <Interviews /> },
      { path: "interviews/schedule", element: <Schedule /> },
      { path: "surveys", element: <Surveys /> },
      { path: "interviews/:id/report", element: <InterviewReport /> },
      { path: "question-library", element: <QuestionLibrary /> },
      { path: "question-library/sets/:id", element: <QuestionSetDetail /> },
      { path: "question-library/categories/:name", element: <CategorySetDetail /> },
      // 「面试题集」已并入题库,旧地址重定向
      { path: "question-sets", element: <Navigate to="/question-library?view=sets" replace /> },
      { path: "question-sets/:id", element: <LegacyQuestionSetRedirect /> },
      { path: "reports", element: <Reports /> },
      { path: "reports/interview-quality", element: <InterviewQuality /> },
      { path: "analytics", element: <Analytics /> },
      { path: "settings/llm", element: <LLMSettings /> },
      { path: "settings/email", element: <EmailSettings /> },
      { path: "settings/smtp", element: <SmtpSettings /> },
      { path: "settings/voice", element: <VoiceSettings /> },
      { path: "settings/prompts", element: <PromptSettings /> },
      { path: "settings/email-templates", element: <EmailTemplates /> },
      { path: "settings/license", element: <LicenseSettings /> },
      { path: "settings/team", element: <TeamSettings /> },
      { path: "settings/sso", element: <SsoSettings /> },
      { path: "settings/audit", element: <AuditCenter /> },
      { path: "settings/mcp", element: <McpTokens /> },
      { path: "settings/dictionaries", element: <Dictionaries /> },
      { path: "settings/schedulers", element: <Schedulers /> },
      { path: "settings/idempotency", element: <IdempotencyKeys /> },
      { path: "settings/model-calls", element: <ModelCalls /> },
      { path: "settings/automations", element: <Automations /> },
      { path: "settings/imports", element: <ImportWizard /> },
    ],
  },
  { path: "*", element: <Navigate to="/" replace /> },
]);
