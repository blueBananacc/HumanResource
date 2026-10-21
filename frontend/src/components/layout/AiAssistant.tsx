import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import {
  ArrowUp,
  FileText,
  History,
  Mail,
  MessageSquare,
  PanelRight,
  PictureInPicture2,
  Plus,
  Sparkles,
  Target,
  TrendingUp,
  UserSearch,
  X,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

// 前端交互骨架:后端 agent 接口未实现,发送后给出占位回复。

/** HiBubble 气泡形品牌头像(规范 §6:小尺寸用气泡标识)。 */
export function AiAvatar({ size = 32 }: { size?: number }) {
  return (
    <span
      className="relative grid place-items-center bg-[var(--color-accent)] text-white shrink-0 rounded-[45%_45%_45%_14%]"
      style={{ width: size, height: size }}
    >
      <Sparkles style={{ width: size * 0.5, height: size * 0.5 }} />
      <span
        className="absolute -top-0.5 -right-0.5 rounded-full bg-[var(--color-orange)] ring-2 ring-white"
        style={{ width: size * 0.28, height: size * 0.28 }}
      />
    </span>
  );
}

type Capability = { label: string; icon: LucideIcon; prompts: string[] };
const CAPABILITIES: Capability[] = [
  { label: "人才筛选", icon: UserSearch, prompts: ["找出 3 年以上 Java 经验的候选人", "最近一周新增了哪些候选人？"] },
  { label: "面试出题", icon: Sparkles, prompts: ["为前端工程师岗位生成 10 道面试题", "出几道考察沟通能力的行为面试题"] },
  { label: "岗位匹配", icon: Target, prompts: ["哪些候选人最适合产品经理岗位？", "解释一下这位候选人的匹配分"] },
  { label: "报告解读", icon: FileText, prompts: ["总结最近一场面试的结论", "对比两位候选人的面试表现"] },
  { label: "招聘分析", icon: TrendingUp, prompts: ["本月招聘漏斗转化情况如何？", "哪个岗位招聘周期最长？"] },
  { label: "撰写邮件", icon: Mail, prompts: ["写一封面试邀请邮件", "写一封委婉的拒信"] },
];
const DEFAULT_PROMPTS = [
  "最近一周新增了哪些候选人？",
  "为前端工程师岗位生成 10 道面试题",
  "本月招聘漏斗转化情况如何？",
  "写一封面试邀请邮件",
];

type Msg = { role: "user" | "assistant"; text: string };
type Mode = "agent" | "chat";

const PANEL_W = 400;
const RAIL_W = 48;
const SNAP_PX = 40;
const POS_KEY = "hronfire.ai.pos";

/** 文档根上的 zoom(字体大小设置);指针坐标需换算回 CSS 像素。 */
const zoom = () => parseFloat(document.documentElement.style.zoom || "1") || 1;
const viewport = () => ({ w: window.innerWidth / zoom(), h: window.innerHeight / zoom() });
const panelH = () => Math.min(680, viewport().h - 80);

function clampPos(x: number, y: number) {
  const { w, h } = viewport();
  return {
    x: Math.min(Math.max(8, x), w - RAIL_W - PANEL_W - 8),
    y: Math.min(Math.max(64, y), h - panelH() - 8),
  };
}

function initialPos() {
  try {
    const saved = JSON.parse(localStorage.getItem(POS_KEY) ?? "null");
    if (saved && typeof saved.x === "number") return clampPos(saved.x, saved.y);
  } catch {
    /* ignore */
  }
  const { w, h } = viewport();
  return clampPos(w - RAIL_W - PANEL_W - 16, h - panelH() - 16);
}

export function AiAssistant({
  open,
  docked,
  onClose,
  onDockChange,
}: {
  open: boolean;
  docked: boolean;
  onClose: () => void;
  onDockChange: (docked: boolean) => void;
}) {
  const [pos, setPos] = useState(initialPos);
  const [dragging, setDragging] = useState(false);
  const [snapHint, setSnapHint] = useState(false);
  const drag = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null);
  // window 监听器里读最新值
  const live = useRef({ pos, snapHint, onDockChange });
  live.current = { pos, snapHint, onDockChange };

  useEffect(() => {
    const onResize = () => setPos((p) => clampPos(p.x, p.y));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // 拖动期间监听 window:贴边→浮窗切换会重新挂载标题栏,元素级 pointer capture 会丢
  useEffect(() => {
    if (!dragging) return;
    const move = (e: PointerEvent) => {
      const d = drag.current;
      if (!d) return;
      const z = zoom();
      const rawX = d.ox + (e.clientX - d.sx) / z;
      setPos(clampPos(rawX, d.oy + (e.clientY - d.sy) / z));
      // 拖到右侧边缘附近 → 提示松手即贴边
      setSnapHint(rawX + PANEL_W >= viewport().w - RAIL_W - SNAP_PX);
    };
    const up = () => {
      drag.current = null;
      setDragging(false);
      const { pos: p, snapHint: snap, onDockChange: dock } = live.current;
      setSnapHint(false);
      if (snap) dock(true);
      else localStorage.setItem(POS_KEY, JSON.stringify(p));
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, [dragging]);

  const onPointerDown = (e: ReactPointerEvent) => {
    if (e.button !== 0 || (e.target as HTMLElement).closest("button")) return;
    e.preventDefault();
    let origin = pos;
    if (docked) {
      // 贴边状态下拖动标题栏 → 脱离成浮窗,窗口跟随指针
      const z = zoom();
      origin = clampPos(e.clientX / z - PANEL_W / 2, e.clientY / z - 20);
      setPos(origin);
      onDockChange(false);
    }
    drag.current = { sx: e.clientX, sy: e.clientY, ox: origin.x, oy: origin.y };
    setDragging(true);
  };

  const panel = (
    <ChatPanel
      docked={docked}
      onClose={onClose}
      onToggleDock={() => onDockChange(!docked)}
      onDragStart={onPointerDown}
      dragging={dragging}
    />
  );

  // 两种形态共用同一棵树,切换时不重置对话。贴边:作为布局列挤压内容区;浮动:fixed。
  return (
    <>
      {snapHint && (
        <div
          className="fixed top-14 bottom-0 z-30 bg-[var(--color-accent-soft)] border-2 border-dashed border-[var(--color-accent)] rounded-l-xl pointer-events-none"
          style={{ right: 0, width: PANEL_W }}
          aria-hidden
        />
      )}
      <div
        className={cn(
          "bg-white overflow-hidden",
          !open && "hidden",
          docked
            ? "h-full shrink-0 border-l border-[var(--color-border-subtle)]"
            : "fixed z-40 rounded-xl border border-[var(--color-border-subtle)]",
          !docked &&
            (dragging
              ? "shadow-[0_16px_48px_rgba(58,58,55,0.22)] select-none"
              : "shadow-[0_10px_40px_rgba(58,58,55,0.16)]"),
        )}
        style={docked ? { width: PANEL_W } : { left: pos.x, top: pos.y, width: PANEL_W, height: panelH() }}
      >
        {panel}
      </div>
    </>
  );
}

function ChatPanel({
  docked,
  onClose,
  onToggleDock,
  onDragStart,
  dragging,
}: {
  docked: boolean;
  onClose: () => void;
  onToggleDock: () => void;
  onDragStart: (e: ReactPointerEvent) => void;
  dragging: boolean;
}) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("agent");
  const [cap, setCap] = useState<Capability | null>(null);
  const [pending, setPending] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, pending]);

  const send = (text = input) => {
    const t = text.trim();
    if (!t || pending) return;
    setMessages((m) => [...m, { role: "user", text: t }]);
    setInput("");
    setPending(true);
    window.setTimeout(() => {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: `AI 助手的后端能力还在建设中，暂时无法处理「${t}」。接入后我会以${mode === "agent" ? " Agent 模式直接帮你执行操作" : "对话模式回答问题"}。`,
        },
      ]);
      setPending(false);
    }, 600);
  };

  const prompts = cap?.prompts ?? DEFAULT_PROMPTS;

  return (
    <div className="flex flex-col h-full font-body">
      {/* 标题栏:拖动手柄 */}
      <div
        onPointerDown={onDragStart}
        className={cn(
          "h-12 shrink-0 flex items-center gap-2 pl-4 pr-2 border-b border-[var(--color-border-row)] bg-[var(--color-bg-canvas)] touch-none",
          dragging ? "cursor-grabbing" : "cursor-grab",
        )}
      >
        <AiAvatar size={22} />
        <span className="font-heading font-semibold text-[14px] text-[var(--color-text-primary)] flex-1 truncate">
          HROnfire AI 助手
        </span>
        <HeaderBtn icon={Plus} label="新对话" onClick={() => { setMessages([]); setCap(null); }} />
        <HeaderBtn icon={History} label="历史会话（即将上线）" disabled />
        <HeaderBtn
          icon={docked ? PictureInPicture2 : PanelRight}
          label={docked ? "浮动窗口" : "贴靠右侧"}
          onClick={onToggleDock}
        />
        <HeaderBtn icon={X} label="关闭" onClick={onClose} />
      </div>

      {/* 内容区 */}
      <div ref={listRef} className="flex-1 overflow-y-auto px-4 py-5">
        {messages.length === 0 ? (
          <div className="flex flex-col gap-5">
            <div className="flex items-center gap-4 pt-4 pb-2">
              <AiAvatar size={60} />
              <div className="flex flex-col">
                <span className="font-heading text-[20px] font-semibold text-[var(--color-text-primary)] leading-snug">
                  你好，欢迎使用
                </span>
                <span className="font-heading text-[20px] font-bold text-[var(--color-accent)] leading-snug">
                  AI 助手
                </span>
              </div>
            </div>

            <div className="flex flex-col gap-2.5">
              <span className="text-[12px] font-bold text-[var(--color-text-tertiary)] tracking-[0.08em]">我可以帮你</span>
              <div className="grid grid-cols-3 gap-2">
                {CAPABILITIES.map((c) => {
                  const active = cap?.label === c.label;
                  return (
                    <button
                      key={c.label}
                      type="button"
                      onClick={() => setCap(active ? null : c)}
                      className={cn(
                        "flex items-center gap-2 rounded-xl border px-2.5 py-2.5 text-left transition-all duration-200 ease-[var(--ease-bob)] active:scale-[0.98]",
                        active
                          ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]"
                          : "border-[var(--color-border-subtle)] bg-white hover:border-[var(--color-accent)] hover:shadow-[var(--shadow-card)]",
                      )}
                    >
                      <span className="w-7 h-7 rounded-lg bg-[var(--color-accent-soft)] text-[var(--color-accent)] grid place-items-center shrink-0">
                        <c.icon className="w-4 h-4" />
                      </span>
                      <span className={cn("text-[12px] font-bold truncate", active ? "text-[var(--color-accent)]" : "text-[var(--color-text-primary)]")}>
                        {c.label}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {prompts.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => send(p)}
                  className="text-[12px] text-[var(--color-text-secondary)] bg-[var(--color-bg-muted)] hover:bg-[var(--color-accent-soft)] hover:text-[var(--color-accent)] rounded-full px-3 py-1.5 transition-colors text-left"
                >
                  <span className="text-[var(--color-orange)] font-bold mr-1">#</span>
                  {p}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="self-end max-w-[85%] bg-[var(--color-accent)] text-white text-[13px] leading-relaxed rounded-2xl rounded-br-sm px-3.5 py-2 whitespace-pre-wrap">
                  {m.text}
                </div>
              ) : (
                <div key={i} className="flex gap-2 max-w-[92%]">
                  <AiAvatar size={24} />
                  <div className="bg-[var(--color-bg-muted)] text-[var(--color-text-primary)] text-[13px] leading-relaxed rounded-2xl rounded-tl-sm px-3.5 py-2 whitespace-pre-wrap">
                    {m.text}
                  </div>
                </div>
              ),
            )}
            {pending && (
              <div className="flex gap-2 items-center">
                <AiAvatar size={24} />
                <span className="flex gap-1 px-3.5 py-3 bg-[var(--color-bg-muted)] rounded-2xl rounded-tl-sm">
                  {[0, 150, 300].map((d) => (
                    <span key={d} className="w-1.5 h-1.5 rounded-full bg-[var(--color-text-tertiary)] animate-bounce" style={{ animationDelay: `${d}ms` }} />
                  ))}
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 输入区 */}
      <div className="shrink-0 px-4 pb-3 pt-1">
        <div
          className="rounded-xl border border-[var(--color-border-subtle)] bg-white focus-within:border-[var(--color-accent)] focus-within:shadow-[0_0_0_3px_var(--color-accent-soft)] transition-all"
          onClick={() => inputRef.current?.focus()}
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send();
              }
            }}
            rows={2}
            placeholder="有什么可以帮你？Enter 发送，Shift+Enter 换行"
            className="w-full resize-none bg-transparent px-3 pt-2.5 text-[13px] text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] outline-none focus-visible:outline-none"
          />
          <div className="flex items-center justify-between px-2 pb-2">
            <div className="flex items-center rounded-full bg-[var(--color-bg-muted)] p-0.5">
              {([
                ["agent", "Agent", Sparkles],
                ["chat", "Chat", MessageSquare],
              ] as const).map(([v, label, Icon]) => (
                <button
                  key={v}
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setMode(v); }}
                  title={v === "agent" ? "Agent：帮你直接执行操作" : "Chat：只回答问题"}
                  className={cn(
                    "flex items-center gap-1 rounded-full px-2.5 py-1 text-[12px] font-bold transition-all duration-200",
                    mode === v
                      ? "bg-white text-[var(--color-accent)] shadow-[var(--shadow-card)]"
                      : "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]",
                  )}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); send(); }}
              disabled={!input.trim() || pending}
              aria-label="发送"
              className="w-8 h-8 rounded-full grid place-items-center bg-[var(--color-accent)] text-white hover:bg-[var(--color-accent-hover)] active:scale-[0.94] transition-all disabled:opacity-40 disabled:pointer-events-none"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          </div>
        </div>
        <p className="mt-2 text-center text-[11px] text-[var(--color-text-tertiary)]">
          内容由 AI 生成，仅供参考，请结合实际判断
        </p>
      </div>
    </div>
  );
}

function HeaderBtn({
  icon: Icon,
  label,
  onClick,
  disabled,
}: {
  icon: LucideIcon;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      className="w-8 h-8 grid place-items-center rounded-full text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-muted)] hover:text-[var(--color-text-primary)] transition-colors disabled:opacity-40 disabled:pointer-events-none"
    >
      <Icon className="w-4 h-4" />
    </button>
  );
}
