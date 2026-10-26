import {
  Activity,
  Banknote,
  Briefcase,
  CalendarDays,
  ClipboardList,
  Clock,
  FileSpreadsheet,
  Fingerprint,
  Goal,
  Home,
  BookOpen,
  KeyRound,
  ListChecks,
  LayoutDashboard,
  LifeBuoy,
  Mail,
  Mic,
  Network,
  Plug,
  Receipt,
  Send,
  Settings,
  ShieldCheck,
  Shuffle,
  Sparkles,
  TrendingUp,
  Users,
  Video,
  Wallet,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { fetchJSON } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM, type AppPermission } from "@/lib/roles";

/*
 * 侧栏导航与顶栏搜索共用的页面定义。层级规则(参照 bob 的"深色一级栏 + 浅色二级面板"):
 *
 * 1. 最多三层:一级模块(NavModule)→ 分组(NavGroup,只是标题,不可点)→ 页面(NavItem)。
 *    不做第四层;再细的划分放到页面内的 tab 里。
 * 2. 一级模块要么是直达页面(`to`),要么带 `groups`(点击后打开二级面板)。二者互斥。
 * 3. 带 groups 的模块:点击 → 跳到第一个可用页面 + 展开二级面板;已在该模块内再点 → 收起/展开面板。
 * 4. 权限只标在页面上;模块/分组里没有可见页面时整体隐藏。
 * 5. `soon` 表示规划中的页面:在面板里置灰显示"规划"标签,不可点、不进搜索。
 * 6. 一级模块分 main(上方)与 bottom(栏底,放设置等低频入口)两区。
 * 7. 同一模块内的页面要么全带图标,要么全不带(条目多、需要扫视时才加,如设置)。
 * 8. 详情页(/talents/:id 等)不进面板,用模块的 `alsoActive` 前缀保持所属模块高亮。
 */

export type NavItem = {
  to: string;
  label: string;
  icon?: LucideIcon;
  /** 搜索用的同义词,不展示 */
  keywords?: string;
  /** 仅 admin 可见(相当于 require_role('admin'))。 */
  adminOnly?: boolean;
  /** 需要该权限位才显示。优先级低于 adminOnly。 */
  permission?: AppPermission;
  /** 需要 license 功能位(如 'hr.suite');未激活时整项隐藏。 */
  feature?: string;
  /** 规划中:置灰、不可点、不进搜索 */
  soon?: boolean;
  /** 自定义激活判断;缺省为 pathname 等于或以 `to` 的路径为前缀 */
  activeWhen?: (pathname: string, search: URLSearchParams) => boolean;
};

export type NavGroup = { label?: string; items: NavItem[] };

export type NavModule = {
  key: string;
  label: string;
  icon: LucideIcon;
  /** 直达页面(无二级面板) */
  to?: string;
  keywords?: string;
  /** 二级面板内容 */
  groups?: NavGroup[];
  /** 额外归属的路径前缀(详情页等不在面板里的页面,也让本模块保持激活) */
  alsoActive?: string[];
  /** 模块级 license 功能位;未激活时整个模块隐藏 */
  feature?: string;
};

const pathOf = (to: string) => to.split("?")[0];

export const mainModules: NavModule[] = [
  { key: "home", label: "概览", icon: LayoutDashboard, to: "/", keywords: "首页 dashboard home" },
  {
    key: "org",
    label: "组织",
    icon: Network,
    feature: "hr.suite",
    alsoActive: ["/employees/", "/org-chart", "/lifecycle"],
    groups: [
      {
        label: "员工",
        items: [
          { to: "/employees", label: "员工列表", permission: PERM.MANAGE_EMPLOYEES, keywords: "员工 employee 花名册" },
          { to: "/org-chart", label: "组织架构图", keywords: "汇报线 架构 org" },
          { to: "/lifecycle", label: "生命周期", permission: PERM.MANAGE_LIFECYCLE, keywords: "入职 离职 调动 晋升 onboarding" },
        ],
      },
    ],
  },
  {
    key: "talent",
    label: "人才",
    icon: Users,
    alsoActive: ["/talents/"],
    groups: [
      {
        items: [
          { to: "/talents", label: "人才库", keywords: "候选人 简历 上传 talent resume" },
          { to: "/campaigns", label: "人才再激活", permission: PERM.VIEW_REPORTS, feature: "hr.suite", keywords: "外联 触达 campaign 批量邮件" },
        ],
      },
      {
        label: "规划中",
        items: [
          { to: "/talents/tags", label: "人才标签", soon: true },
          { to: "/talents/dedupe", label: "重复候选人合并", soon: true },
        ],
      },
    ],
  },
  {
    key: "recruit",
    label: "招聘",
    icon: Briefcase,
    alsoActive: ["/jobs/", "/question-library/"],
    groups: [
      {
        label: "岗位",
        items: [
          { to: "/jobs", label: "岗位列表", keywords: "职位 job 匹配" },
          { to: "/scoring-rules", label: "评分规则", keywords: "评分规则 scoring 维度 匹配评估" },
          { to: "/pipeline", label: "招聘管道", permission: PERM.VIEW_REPORTS, feature: "hr.suite", keywords: "看板 kanban 阶段 pipeline 申请" },
          { to: "/requisitions", label: "招聘需求审批", permission: PERM.WRITE_REQUISITIONS, feature: "hr.suite", keywords: "编制 headcount 需求 requisition" },
          { to: "/offers", label: "Offer 管理", permission: PERM.MANAGE_OFFERS, feature: "hr.suite", keywords: "offer 录用 通知书" },
        ],
      },
      {
        label: "题库",
        items: [
          {
            to: "/question-library",
            label: "题目",
            keywords: "题库 面试题 出题 生成",
            activeWhen: (p, s) => p === "/question-library" && s.get("view") !== "sets",
          },
          {
            to: "/question-library?view=sets",
            label: "题集",
            keywords: "题库 题集 分类",
            activeWhen: (p, s) =>
              (p === "/question-library" && s.get("view") === "sets") ||
              /^\/question-library\/(sets|categories)\//.test(p),
          },
        ],
      },
    ],
  },
  {
    key: "interview",
    label: "面试",
    icon: Video,
    alsoActive: ["/interviews/"],
    groups: [
      {
        items: [
          { to: "/interviews", label: "面试安排", permission: PERM.VIEW_REPORTS, keywords: "interview 推荐面试" },
          { to: "/reports", label: "面试报告", permission: PERM.VIEW_REPORTS, keywords: "报告 report" },
          { to: "/reports/interview-quality", label: "质量洞察", permission: PERM.VIEW_REPORTS, keywords: "质量 洞察 quality 维度 遗漏" },
          { to: "/surveys", label: "问卷调查", icon: ClipboardList, permission: PERM.VIEW_REPORTS, keywords: "问卷 survey 满意度 反馈" },
        ],
      },
      {
        label: "排期",
        items: [{ to: "/interviews/schedule", label: "面试安排", icon: CalendarDays, permission: PERM.VIEW_REPORTS, keywords: "会议室 预约 排期 schedule" }],
      },
    ],
  },
  {
    key: "timeoff",
    label: "假勤",
    icon: CalendarDays,
    feature: "hr.suite",
    groups: [
      {
        label: "请假",
        items: [
          { to: "/leave", label: "请假审批", permission: PERM.APPROVE_LEAVE, keywords: "请假 审批 leave" },
          { to: "/leave/balances", label: "假期额度", permission: PERM.MANAGE_LEAVE, keywords: "额度 余额 年假" },
          { to: "/leave/types", label: "假期类型", permission: PERM.MANAGE_LEAVE, keywords: "类型 设置" },
          { to: "/leave/policies", label: "假期政策", permission: PERM.MANAGE_LEAVE, keywords: "政策 发放 结转 policy" },
        ],
      },
      {
        label: "考勤",
        items: [
          { to: "/attendance", label: "考勤记录", permission: PERM.VIEW_REPORTS, keywords: "出勤 考勤 attendance" },
          { to: "/attendance/checkins", label: "打卡流水", permission: PERM.VIEW_REPORTS, keywords: "打卡 签到 checkin" },
          { to: "/attendance/requests", label: "补卡审批", permission: PERM.APPROVE_LEAVE, keywords: "补卡 漏打卡 申诉" },
          { to: "/attendance/flags", label: "考勤异常", permission: PERM.APPROVE_LEAVE, keywords: "迟到 早退 异常 工时 汇总" },
          { to: "/overtime", label: "加班审批", permission: PERM.APPROVE_LEAVE, keywords: "加班 调休 overtime" },
          { to: "/attendance/calendar", label: "节假日历", permission: PERM.MANAGE_ATTENDANCE, keywords: "节假日 工作日历 调休" },
          { to: "/attendance/shifts", label: "排班", permission: PERM.MANAGE_ATTENDANCE, keywords: "排班 班次 替班 schedule shift" },
        ],
      },
    ],
  },
  {
    key: "perf",
    label: "绩效",
    icon: Goal,
    feature: "hr.suite",
    groups: [
      {
        items: [
          { to: "/performance", label: "目标", permission: PERM.VIEW_REPORTS, keywords: "目标 goal okr" },
          { to: "/performance/cycles", label: "考核", permission: PERM.MANAGE_PERFORMANCE, keywords: "绩效考核 appraisal 评分" },
        ],
      },
    ],
  },
  {
    key: "expense",
    label: "费用",
    icon: Receipt,
    feature: "hr.suite",
    groups: [
      {
        items: [
          { to: "/expenses", label: "报销审批", permission: PERM.VIEW_REPORTS, keywords: "报销 费用 expense" },
          { to: "/expenses/types", label: "报销类型", permission: PERM.MANAGE_EXPENSES, keywords: "报销类型" },
        ],
      },
    ],
  },
  {
    key: "payroll",
    label: "薪酬",
    icon: Banknote,
    feature: "hr.suite",
    groups: [
      {
        items: [
          { to: "/payroll", label: "工资条", permission: PERM.MANAGE_PAYROLL, keywords: "工资 薪资 工资条 salary" },
          { to: "/salary-bands", label: "薪酬带宽", permission: PERM.MANAGE_PAYROLL, keywords: "定薪 带宽 区间 salary band" },
        ],
      },
    ],
  },
  {
    key: "service",
    label: "员工服务",
    icon: LifeBuoy,
    feature: "hr.suite",
    groups: [
      {
        items: [
          { to: "/approvals", label: "审批中心", keywords: "审批 收件箱 待办 approval inbox" },
          { to: "/announcements", label: "公告管理", permission: PERM.VIEW_REPORTS, keywords: "公告 通知 announcement news" },
          { to: "/requests", label: "员工申请", permission: PERM.MANAGE_EMPLOYEES, keywords: "申请 审批 培训 福利 request" },
          { to: "/cases", label: "HR 工单", permission: PERM.MANAGE_EMPLOYEES, keywords: "工单 诉求 ticket case 服务台" },
          { to: "/knowledge", label: "知识库", permission: PERM.VIEW_REPORTS, keywords: "知识 制度 faq knowledge" },
        ],
      },
    ],
  },
  {
    key: "analytics",
    label: "分析",
    icon: TrendingUp,
    groups: [
      {
        items: [
          { to: "/analytics", label: "招聘数据", permission: PERM.VIEW_REPORTS, keywords: "统计 图表 analytics 数据分析" },
        ],
      },
      {
        label: "规划中",
        items: [{ to: "/analytics/custom", label: "自定义报表", soon: true, permission: PERM.VIEW_REPORTS }],
      },
    ],
  },
];

// 设置类页面默认需要 manage_settings(只有 admin 持有);adminOnly 是更严格的硬限 admin。
export const bottomModules: NavModule[] = [
  {
    key: "settings",
    label: "设置",
    icon: Settings,
    groups: [
      {
        label: "AI 与语音",
        items: [
          { to: "/settings/llm", label: "LLM 配置", icon: Plug, permission: PERM.MANAGE_SETTINGS, keywords: "大模型 模型 key ai" },
          { to: "/settings/voice", label: "语音 STT/TTS", icon: Mic, permission: PERM.MANAGE_SETTINGS, keywords: "语音识别 合成" },
          { to: "/settings/prompts", label: "AI 提示词", icon: Sparkles, permission: PERM.MANAGE_SETTINGS, keywords: "prompt 提示词 模板" },
          { to: "/settings/mcp", label: "MCP 接入", icon: Plug, feature: "assistant.agent", keywords: "mcp token 令牌 外部 ai" },
        ],
      },
      {
        label: "邮件",
        items: [
          { to: "/settings/email", label: "邮箱拉取", icon: Mail, permission: PERM.MANAGE_SETTINGS, keywords: "imap 收件 简历邮箱" },
          { to: "/settings/smtp", label: "SMTP 发件", icon: Send, permission: PERM.MANAGE_SETTINGS, keywords: "邮件 发信 smtp" },
          { to: "/settings/email-templates", label: "邮件模板", icon: Mail, permission: PERM.MANAGE_SETTINGS, keywords: "模板 email template 邮件正文" },
        ],
      },
      {
        label: "组织与安全",
        items: [
          { to: "/settings/team", label: "团队", icon: Users, permission: PERM.MANAGE_TEAM, keywords: "成员 组织 账号" },
          { to: "/settings/sso", label: "SSO 接入", icon: Shuffle, adminOnly: true, keywords: "单点登录" },
          { to: "/settings/audit", label: "审计中心", icon: ShieldCheck, adminOnly: true, keywords: "日志 audit" },
          { to: "/settings/dictionaries", label: "数据字典", icon: BookOpen, permission: PERM.MANAGE_SETTINGS, keywords: "字典 枚举 下拉 dictionary" },
          { to: "/settings/automations", label: "自动化规则", icon: Zap, permission: PERM.MANAGE_SETTINGS, keywords: "自动化 规则 触发 automation 通知" },
          { to: "/settings/imports", label: "数据导入", icon: FileSpreadsheet, permission: PERM.MANAGE_EMPLOYEES, keywords: "导入 csv excel 批量 AI import" },
          { to: "/settings/schedulers", label: "后台任务", icon: ListChecks, permission: PERM.MANAGE_SETTINGS, keywords: "任务 调度 scheduler 运维" },
          { to: "/settings/idempotency", label: "幂等键", icon: Fingerprint, permission: PERM.MANAGE_SETTINGS, keywords: "幂等 键 重试 idempotency 回放" },
          { to: "/settings/model-calls", label: "模型调用", icon: Activity, permission: PERM.MANAGE_SETTINGS, keywords: "LLM 调用 留痕 模型 model call AI 观测" },
        ],
      },
      {
        label: "授权",
        items: [
          { to: "/settings/license", label: "License 设置", icon: KeyRound, permission: PERM.MANAGE_SETTINGS, keywords: "授权 许可证 试用" },
        ],
      },
    ],
  },
];

/**
 * 员工自助(ESS)导航 — role=employee 登录后整个侧栏换成这套,
 * 对应后端 /api/me/*。直达式,不用二级面板。
 */
export const essModules: NavModule[] = [
  { key: "me-home", label: "我的工作台", icon: Home, to: "/me", keywords: "员工 自助 ess" },
  { key: "me-leave", label: "请假", icon: CalendarDays, to: "/me/leave", keywords: "请假 年假 休假" },
  { key: "me-attend", label: "考勤", icon: Clock, to: "/me/attendance", keywords: "打卡 考勤 出勤" },
  { key: "me-payslips", label: "工资条", icon: Wallet, to: "/me/payslips", keywords: "工资 薪资 工资条" },
  { key: "me-expenses", label: "报销", icon: Receipt, to: "/me/expenses", keywords: "报销 费用" },
  { key: "me-requests", label: "申请", icon: ClipboardList, to: "/me/requests", keywords: "申请 诉求 request" },
  { key: "me-cases", label: "工单", icon: LifeBuoy, to: "/me/cases", keywords: "工单 服务台 case ticket" },
  { key: "me-goals", label: "目标", icon: Goal, to: "/me/goals", keywords: "目标 绩效 okr" },
  { key: "me-profile", label: "我的资料", icon: Users, to: "/me/profile", keywords: "档案 资料 个人信息" },
];

/**
 * 绑定了员工档案的非 employee 角色(用人经理/HR 等)追加的「我的」模块:
 * 二级面板放整套 ESS 页面 —— 他们可以审批下属,也需要给自己提请假/报销。
 */
const boundEssModule: NavModule = {
  key: "me",
  label: "我的",
  icon: Home,
  feature: "hr.suite",
  groups: [
    {
      items: [
        { to: "/me", label: "我的工作台", keywords: "员工 自助 ess" },
        { to: "/me/leave", label: "我的请假", keywords: "请假 年假 休假" },
        { to: "/me/attendance", label: "我的考勤", keywords: "打卡 考勤 出勤" },
        { to: "/me/payslips", label: "我的工资条", keywords: "工资 薪资 工资条" },
        { to: "/me/expenses", label: "我的报销", keywords: "报销 费用" },
        { to: "/me/requests", label: "我的申请", keywords: "申请 诉求 request" },
        { to: "/me/cases", label: "我的工单", keywords: "工单 服务台 case" },
        { to: "/me/goals", label: "我的目标", keywords: "目标 绩效 okr" },
        { to: "/me/profile", label: "我的资料", keywords: "档案 资料 个人信息" },
      ],
    },
  ],
};

export function isItemActive(it: NavItem, pathname: string, search: URLSearchParams) {
  if (it.soon) return false;
  if (it.activeWhen) return it.activeWhen(pathname, search);
  const p = pathOf(it.to);
  return pathname === p || (p !== "/" && pathname.startsWith(p + "/"));
}

export function moduleItems(m: NavModule): NavItem[] {
  return m.groups?.flatMap((g) => g.items) ?? [];
}

/** 模块的落地页:直达页面,或第一个可用(非规划)页面。 */
export function moduleEntry(m: NavModule): string | undefined {
  return m.to ?? moduleItems(m).find((it) => !it.soon)?.to;
}

export function isModuleActive(m: NavModule, pathname: string, search: URLSearchParams) {
  if (m.to) return m.to === "/" ? pathname === "/" : pathname === m.to || pathname.startsWith(m.to + "/");
  return (
    moduleItems(m).some((it) => isItemActive(it, pathname, search)) ||
    (m.alsoActive ?? []).some((pre) => pathname.startsWith(pre))
  );
}

/** 搜索用的扁平页面条目 */
export type FlatPage = { to: string; label: string; icon: LucideIcon; keywords?: string; module: string };

export function useVisibleNav() {
  const { user, hasPermission } = useAuth();
  const isAdmin = user?.role === "admin";
  const isEmployee = user?.role === "employee";
  // license 功能位:hris 模块挂在 hr.suite 上
  const { data: lic, isError: licError } = useQuery<{
    features: Record<string, boolean>;
  }>({
    queryKey: ["license-status"],
    queryFn: () => fetchJSON("/api/license/status"),
    staleTime: 60_000,
  });
  // 未加载完成时隐藏功能模块,避免社区版闪现不可用的 HRIS 导航;
  // 接口报错则 fail-open 保持原有行为(后端仍会 402 兜底)。
  const featureOn = (f?: string) =>
    !f || (lic ? (lic.features?.[f] ?? false) : licError);
  const allowed = (it: NavItem) => {
    if (!featureOn(it.feature)) return false;
    if (it.adminOnly) return isAdmin;
    if (it.permission) return hasPermission(it.permission);
    return true;
  };
  const filter = (mods: NavModule[]): NavModule[] =>
    mods.flatMap((m) => {
      if (!featureOn(m.feature)) return [];
      if (m.to) return [m];
      const groups = (m.groups ?? [])
        .map((g) => ({ ...g, items: g.items.filter(allowed) }))
        .filter((g) => g.items.length > 0);
      // 只剩规划项的模块不展示(没有可去的页面)
      if (!groups.some((g) => g.items.some((it) => !it.soon))) return [];
      return [{ ...m, groups }];
    });
  // 员工账号:整组换 ESS 导航;非员工但绑定了员工档案的账号
  // (用人经理等)在正常导航尾部追加「我的」入口。
  const main = isEmployee
    ? essModules
    : [...filter(mainModules), ...(user?.employee_id ? filter([boundEssModule]) : [])];
  const bottom = isEmployee ? [] : filter(bottomModules);
  const pages: FlatPage[] = [...main, ...bottom].flatMap((m) =>
    m.to
      ? [{ to: m.to, label: m.label, icon: m.icon, keywords: m.keywords, module: m.label }]
      : moduleItems(m)
          .filter((it) => !it.soon)
          .map((it) => ({
            to: it.to,
            label: it.label,
            icon: it.icon ?? m.icon,
            keywords: `${m.label} ${it.keywords ?? ""}`,
            module: m.label,
          })),
  );
  return { main, bottom, pages };
}
