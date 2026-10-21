import {
  LayoutDashboard,
  Briefcase,
  Video,
  ChartBar,
  KeyRound,
  Mail,
  Mic,
  Plug,
  Send,
  ShieldCheck,
  Shuffle,
  BookOpen,
  TrendingUp,
  Users,
  type LucideIcon,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { PERM, type AppPermission } from "@/lib/roles";

// 侧栏导航与顶栏搜索共用的页面定义。

export type NavDef = {
  to: string;
  label: string;
  icon: LucideIcon;
  /** 搜索用的同义词,不展示 */
  keywords?: string;
  /** 仅 admin 可见(快捷:相当于 require_role('admin'))。 */
  adminOnly?: boolean;
  /** 需要至少一个该权限位才显示。优先级低于 adminOnly。 */
  permission?: AppPermission;
};

export const mainNav: NavDef[] = [
  { to: "/", label: "概览", icon: LayoutDashboard, keywords: "首页 dashboard home" },
  { to: "/talents", label: "人才库", icon: Users, keywords: "候选人 简历 上传 talent resume" },
  { to: "/jobs", label: "岗位", icon: Briefcase, keywords: "职位 job" },
  { to: "/question-library", label: "题库", icon: BookOpen, keywords: "题目 题集 面试题 出题 生成" },
  { to: "/interviews", label: "面试", icon: Video, permission: PERM.VIEW_REPORTS, keywords: "interview" },
  { to: "/reports", label: "报告", icon: ChartBar, permission: PERM.VIEW_REPORTS, keywords: "面试报告 report" },
  { to: "/analytics", label: "数据分析", icon: TrendingUp, permission: PERM.VIEW_REPORTS, keywords: "统计 图表 analytics" },
];

// 系统设置类菜单 —— 默认需要 manage_settings(只有 admin 持有)。
// adminOnly 单独标的项是更严格的"硬限 admin"(SSO / 审计)。
export const sysNav: NavDef[] = [
  { to: "/settings/llm", label: "LLM 配置", icon: Plug, permission: PERM.MANAGE_SETTINGS, keywords: "大模型 模型 key ai" },
  { to: "/settings/email", label: "邮箱拉取", icon: Mail, permission: PERM.MANAGE_SETTINGS, keywords: "imap 收件 简历邮箱" },
  { to: "/settings/smtp", label: "SMTP 发件", icon: Send, permission: PERM.MANAGE_SETTINGS, keywords: "邮件 发信 smtp" },
  { to: "/settings/voice", label: "语音 STT/TTS", icon: Mic, permission: PERM.MANAGE_SETTINGS, keywords: "语音识别 合成" },
  { to: "/settings/license", label: "License 设置", icon: KeyRound, permission: PERM.MANAGE_SETTINGS, keywords: "授权 许可证 试用" },
  { to: "/settings/team", label: "团队", icon: Users, permission: PERM.MANAGE_TEAM, keywords: "成员 组织 账号" },
  { to: "/settings/sso", label: "SSO 接入", icon: Shuffle, adminOnly: true, keywords: "单点登录" },
  { to: "/settings/audit", label: "审计中心", icon: ShieldCheck, adminOnly: true, keywords: "日志 audit" },
];

export function useVisibleNav() {
  const { user, hasPermission } = useAuth();
  const isAdmin = user?.role === "admin";
  const filter = (items: NavDef[]) =>
    items.filter((it) => {
      if (it.adminOnly) return isAdmin;
      if (it.permission) return hasPermission(it.permission);
      return true;
    });
  return { main: filter(mainNav), sys: filter(sysNav) };
}
