/** AI 技能中心 — 对标 HireEasy「技能卡一键调用」+ MCP 开放生态。
 * 技能卡点选即用:注入助手预填问题(`hr:assistant-ask`),不绕过确认门。
 * 开放生态区:MCP 对外工具暴露就绪度评分 + 接入渠道一览。
 */
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  ClipboardList,
  FileSearch,
  FileText,
  Mail,
  Plug,
  Radar,
  Sparkles,
  UserRoundSearch,
  Users,
  type LucideIcon,
} from "lucide-react";
import { fetchJSON } from "@/lib/api";
import { SkillGraphPanel } from "@/components/skills/SkillGraphPanel";

type Skill = {
  icon: LucideIcon;
  title: string;
  desc: string;
  prompt: string;
  tag: string;
};

const SKILLS: Skill[] = [
  {
    icon: UserRoundSearch,
    title: "一句话搜人",
    desc: "用自然语言描述人选画像,从人才库召回排序",
    prompt: "帮我找有 5 年以上 B 端 SaaS 销售经验的人选",
    tag: "检索",
  },
  {
    icon: Radar,
    title: "批量匹配打分",
    desc: "对岗位候选批量跑 BARS 逐维评分,产出对比",
    prompt: "给最近一周入库的简历全部做岗位匹配评分",
    tag: "评分",
  },
  {
    icon: ClipboardList,
    title: "生成面试题包",
    desc: "结合 JD 与候选人简历,一键出题并导出 Word",
    prompt: "为下一场面试生成一份面试题包并导出",
    tag: "出题",
  },
  {
    icon: FileText,
    title: "招聘周报",
    desc: "KPI 漏斗 + Top 候选人 + 下一步行动,三件套成稿",
    prompt: "生成本周招聘进度汇报 PPT",
    tag: "汇报",
  },
  {
    icon: Users,
    title: "候选人报告",
    desc: "按候选人生成多维人才分析报告(含评分依据)",
    prompt: "为当前岗位的 Top1 候选人生成人才分析报告",
    tag: "汇报",
  },
  {
    icon: FileSearch,
    title: "知识库问答",
    desc: "带引用溯源的公司制度/岗位知识检索",
    prompt: "我们的年假制度是怎么规定的?给出引用出处",
    tag: "知识",
  },
  {
    icon: Sparkles,
    title: "约面提醒编排",
    desc: "起一个招聘任务链,JD→搜人→打分→约面→汇报",
    prompt: "为新岗位起一个完整招聘任务链",
    tag: "编排",
  },
  {
    icon: FileSearch,
    title: "数据问答",
    desc: "对话式 BI——问数字、漏斗、转化,AI 查真实指标回答",
    prompt: "本月招聘漏斗转化情况怎么样?",
    tag: "BI",
  },
  {
    icon: ClipboardList,
    title: "JD 合规审查",
    desc: "六维体检岗位描述:歧视性用语/模糊措辞/法律风险",
    prompt: "帮我审查一下当前在招岗位的 JD 有没有合规问题",
    tag: "合规",
  },
];

function ask(prompt: string) {
  window.dispatchEvent(new CustomEvent("hr:assistant-ask", { detail: { text: prompt } }));
}

export function SkillCenter() {
  const { data: tokens } = useQuery<{ id: string; revoked_at: string | null; last_used_at: string | null }[]>({
    queryKey: ["mcp-tokens"],
    queryFn: () => fetchJSON("/api/me/mcp-tokens"),
    staleTime: 60_000,
  });
  const activeTokens = tokens?.filter((t) => !t.revoked_at) ?? [];
  const usedTokens = activeTokens.filter((t) => t.last_used_at);

  // MCP 生态就绪度评分(0-100):服务端点可达(40)+ 已签发令牌(30)+ 令牌被外部调用过(30)
  const ecoScore = 40 + (activeTokens.length > 0 ? 30 : 0) + (usedTokens.length > 0 ? 30 : 0);
  const ecoLabel = ecoScore >= 100 ? "生态活跃" : ecoScore >= 70 ? "已可接入" : "待签发令牌";

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <header className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)] flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-[var(--color-accent)]" />
          AI 技能中心
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          点选技能即用 — 自动唤起 AI 助手并预填任务;写操作仍需你在助手内确认。
        </p>
      </header>

      <div className="grid grid-cols-3 gap-4">
        {SKILLS.map((s) => (
          <button
            key={s.title}
            type="button"
            onClick={() => ask(s.prompt)}
            className="group text-left rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-bg-card)] p-4 shadow-[var(--shadow-card)] hover:border-[var(--color-accent)] hover:shadow-md transition-all flex flex-col gap-2"
          >
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-[var(--color-accent-soft)] grid place-items-center group-hover:bg-[var(--color-orange-tint)] transition-colors">
                <s.icon className="w-4 h-4 text-[var(--color-accent)]" />
              </div>
              <span className="font-heading font-medium text-[14px]">{s.title}</span>
              <span className="ml-auto rounded bg-[var(--color-bg-subtle)] px-1.5 py-px text-[10px] text-[var(--color-text-tertiary)] font-body">
                {s.tag}
              </span>
            </div>
            <p className="text-[12px] text-[var(--color-text-secondary)] font-body">{s.desc}</p>
            <p className="text-[11px] text-[var(--color-text-tertiary)] font-body italic">
              「{s.prompt}」
            </p>
          </button>
        ))}
      </div>

      <section className="rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-bg-card)] shadow-[var(--shadow-card)] overflow-hidden">
        <div className="flex items-center gap-3 px-5 pt-4 pb-3 border-b border-[var(--color-border-row)]">
          <Plug className="w-5 h-5 text-[var(--color-accent)]" />
          <div className="flex-1">
            <div className="font-heading font-semibold text-[15px]">开放生态 · MCP</div>
            <div className="text-[11px] text-[var(--color-text-tertiary)] font-body">
              外部 AI 客户端(Claude/Cursor/自研 agent)通过 MCP 协议调用本工作台工具集
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-lg font-semibold text-[var(--color-text-primary)]">{ecoScore}</span>
            <span
              className={`rounded-full px-2 py-0.5 text-[11px] font-body ${
                ecoScore >= 100
                  ? "bg-emerald-50 text-emerald-600"
                  : ecoScore >= 70
                    ? "bg-amber-50 text-amber-600"
                    : "bg-slate-100 text-slate-500"
              }`}
            >
              {ecoLabel}
            </span>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3 px-5 py-4">
          <EcoItem
            ok
            label="MCP 服务端点"
            detail={`${location.origin}/api/mcp`}
            hint="已就绪"
          />
          <EcoItem
            ok={activeTokens.length > 0}
            label="访问令牌"
            detail={`${activeTokens.length} 个生效中`}
            hint={activeTokens.length > 0 ? "已签发" : "未签发"}
            link="/settings/mcp"
          />
          <EcoItem
            ok={usedTokens.length > 0}
            label="外部调用"
            detail={usedTokens.length > 0 ? `${usedTokens.length} 个令牌有调用记录` : "暂无外部调用"}
            hint={usedTokens.length > 0 ? "活跃" : "等待首次调用"}
          />
        </div>
        <div className="px-5 pb-4 flex items-center gap-2">
          <Mail className="w-3.5 h-3.5 text-[var(--color-text-tertiary)]" />
          <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
            接入渠道:邮箱 IMAP 简历抓取(在
          </span>
          <Link to="/settings/email" className="text-[11px] text-[var(--color-accent)] hover:underline font-body">
            设置 → 邮箱拉取
          </Link>
          <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">中配置)</span>
        </div>
      </section>

      {/* 技能图谱:词典层级+别名+引用热度 */}
      <SkillGraphPanel />
    </main>
  );
}

function EcoItem({ ok, label, detail, hint, link }: { ok: boolean; label: string; detail: string; hint: string; link?: string }) {
  const inner = (
    <div className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-bg-canvas)] p-3 flex flex-col gap-1">
      <div className="flex items-center gap-1.5">
        <span className={`w-1.5 h-1.5 rounded-full ${ok ? "bg-emerald-500" : "bg-slate-300"}`} />
        <span className="text-[12px] font-medium font-body">{label}</span>
        <span className={`ml-auto text-[10px] font-body ${ok ? "text-emerald-600" : "text-[var(--color-text-tertiary)]"}`}>
          {hint}
        </span>
      </div>
      <span className="text-[11px] text-[var(--color-text-secondary)] font-mono truncate">{detail}</span>
    </div>
  );
  return link ? <Link to={link}>{inner}</Link> : inner;
}
