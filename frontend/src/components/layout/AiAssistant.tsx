import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import {
  ArrowUp,
  Check,
  ChevronLeft,
  FileText,
  History,
  Loader2,
  Mail,
  PanelRight,
  PictureInPicture2,
  Plus,
  Sparkles,
  Square,
  Target,
  Trash2,
  TrendingUp,
  UserSearch,
  Wrench,
  X,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchJSON, getToken } from "@/lib/api";

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
  { label: "人才筛选", icon: UserSearch, prompts: ["现在人才库里有哪些候选人？", "最近新增了哪些候选人？"] },
  { label: "面试出题", icon: Sparkles, prompts: ["为前端工程师岗位生成 5 道面试题", "出几道考察沟通能力的行为面试题"] },
  { label: "岗位匹配", icon: Target, prompts: ["哪些候选人最适合产品经理岗位？", "看看有没有匹配分超过 90 的推荐面试"] },
  { label: "报告解读", icon: FileText, prompts: ["总结最近一场面试的结论", "有哪些已完成的面试报告？"] },
  { label: "招聘分析", icon: TrendingUp, prompts: ["现在有哪些进行中的面试？", "有哪些 open 的岗位？"] },
  { label: "发起面试", icon: Mail, prompts: ["帮高分候选人发起一场面试"] },
];
const DEFAULT_PROMPTS = [
  "现在人才库里有哪些候选人？",
  "有哪些 open 的岗位？",
  "看看有没有匹配分超过 90 的推荐面试",
  "总结最近一场面试的结论",
];

type ToolStep = { callId: string; label: string; status: "running" | "done" | "error" };
type ConfirmReq = {
  callId: string;
  label: string;
  name: string;
  args: Record<string, unknown>;
  decided?: "approved" | "rejected" | "timeout";
};
type Msg = {
  role: "user" | "assistant";
  text: string;
  tools?: ToolStep[];
  confirms?: ConfirmReq[];
};
type ConvMeta = { id: string; title: string; updated_at: string; message_count?: number };

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

/** 读 SSE 流,逐帧回调 {event, data}。 */
async function readSse(
  body: ReadableStream<Uint8Array>,
  onFrame: (event: string, data: string) => void,
) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) onFrame(event, data);
    }
  }
}

/** 把 sidecar 存的 pi 消息映射回 UI 气泡(user / assistant 文本)。 */
function storedToMsgs(messages: unknown[]): Msg[] {
  const out: Msg[] = [];
  for (const raw of messages) {
    const m = raw as { role?: string; content?: unknown };
    const text = Array.isArray(m.content)
      ? (m.content as Array<{ type?: string; text?: string }>)
          .filter((c) => c.type === "text" && c.text)
          .map((c) => c.text)
          .join("")
      : typeof m.content === "string"
        ? m.content
        : "";
    if (!text) continue;
    if (m.role === "user") out.push({ role: "user", text });
    else if (m.role === "assistant") out.push({ role: "assistant", text });
  }
  return out;
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
  const [cap, setCap] = useState<Capability | null>(null);
  const [pending, setPending] = useState(false);
  const [convId, setConvId] = useState<string | null>(null);
  const [view, setView] = useState<"chat" | "history">("chat");
  const [convs, setConvs] = useState<ConvMeta[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, pending]);

  useEffect(() => () => abortRef.current?.abort(), []);

  /** 就地更新最后一条 assistant 消息(流式追加)。 */
  const patchLastAssistant = (fn: (m: Msg) => Msg) => {
    setMessages((msgs) => {
      const next = [...msgs];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant") {
          next[i] = fn(next[i]);
          break;
        }
      }
      return next;
    });
  };

  const send = async (text = input) => {
    const t = text.trim();
    if (!t || pending) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setView("chat");
    setMessages((m) => [...m, { role: "user", text: t }, { role: "assistant", text: "" }]);
    setInput("");
    setPending(true);

    try {
      const res = await fetch("/api/assistant/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${getToken() ?? ""}`,
        },
        body: JSON.stringify({ message: t, conversation_id: convId }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        let msg = `HTTP ${res.status}`;
        try {
          const d = (await res.json()) as { detail?: string };
          if (d.detail) msg = d.detail;
        } catch {
          /* ignore */
        }
        patchLastAssistant((m) => ({ ...m, text: `出错了:${msg}` }));
        return;
      }
      await readSse(res.body, (event, data) => {
        const payload = JSON.parse(data) as Record<string, unknown>;
        switch (event) {
          case "conversation":
            setConvId(String(payload.conversation_id));
            break;
          case "text_delta":
            patchLastAssistant((m) => ({ ...m, text: m.text + String(payload.delta ?? "") }));
            break;
          case "tool_start":
            patchLastAssistant((m) => ({
              ...m,
              tools: [
                ...(m.tools ?? []),
                {
                  callId: String(payload.call_id ?? ""),
                  label: String(payload.label ?? payload.name ?? "工具调用"),
                  status: "running" as const,
                },
              ],
            }));
            break;
          case "tool_end":
            patchLastAssistant((m) => ({
              ...m,
              tools: (m.tools ?? []).map((s) =>
                s.callId === String(payload.call_id ?? "")
                  ? { ...s, status: payload.is_error ? ("error" as const) : ("done" as const) }
                  : s,
              ),
            }));
            break;
          case "confirm_required":
            patchLastAssistant((m) => ({
              ...m,
              confirms: [
                ...(m.confirms ?? []),
                {
                  callId: String(payload.call_id ?? ""),
                  label: String(payload.label ?? payload.name ?? "操作"),
                  name: String(payload.name ?? ""),
                  args: (payload.args ?? {}) as Record<string, unknown>,
                },
              ],
            }));
            break;
          case "confirm_timeout":
            patchLastAssistant((m) => ({
              ...m,
              confirms: (m.confirms ?? []).map((c) =>
                c.callId === String(payload.call_id ?? "") ? { ...c, decided: "timeout" } : c,
              ),
            }));
            break;
          case "error":
            patchLastAssistant((m) => ({
              ...m,
              text: m.text + (m.text ? "\n\n" : "") + `出错了:${String(payload.message ?? "未知错误")}`,
            }));
            break;
          default:
            break;
        }
      });
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        patchLastAssistant((m) => ({
          ...m,
          text: m.text + (m.text ? "\n\n" : "") + "网络中断,回复可能不完整,请重试",
        }));
      }
    } finally {
      setPending(false);
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    setPending(false);
  };

  const decide = async (callId: string, approved: boolean) => {
    patchLastAssistant((m) => ({
      ...m,
      confirms: (m.confirms ?? []).map((c) =>
        c.callId === callId ? { ...c, decided: approved ? "approved" : "rejected" } : c,
      ),
    }));
    try {
      await fetchJSON("/api/assistant/confirm", {
        method: "POST",
        body: JSON.stringify({ call_id: callId, approved }),
      });
    } catch {
      /* 超时/已处理 — 状态以 SSE 回流为准 */
    }
  };

  const openHistory = async () => {
    setView("history");
    setHistoryLoading(true);
    try {
      const res = await fetchJSON<{ items: ConvMeta[] }>("/api/assistant/conversations");
      setConvs(res.items ?? []);
    } catch {
      setConvs([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const openConv = async (id: string) => {
    try {
      const conv = await fetchJSON<{ messages?: unknown[] }>(
        `/api/assistant/conversations/${id}`,
      );
      setConvId(id);
      setMessages(storedToMsgs(conv.messages ?? []));
      setView("chat");
    } catch {
      /* 会话可能已被删 */
    }
  };

  const removeConv = async (id: string) => {
    try {
      await fetchJSON(`/api/assistant/conversations/${id}`, { method: "DELETE" });
    } catch {
      /* ignore */
    }
    setConvs((c) => c.filter((x) => x.id !== id));
    if (convId === id) {
      setConvId(null);
      setMessages([]);
    }
  };

  const newChat = () => {
    abortRef.current?.abort();
    setConvId(null);
    setMessages([]);
    setCap(null);
    setPending(false);
    setView("chat");
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
        {view === "history" ? (
          <HeaderBtn icon={ChevronLeft} label="返回对话" onClick={() => setView("chat")} />
        ) : (
          <AiAvatar size={22} />
        )}
        <span className="font-heading font-semibold text-[14px] text-[var(--color-text-primary)] flex-1 truncate">
          {view === "history" ? "历史会话" : "HROnfire AI 助手"}
        </span>
        <HeaderBtn icon={Plus} label="新对话" onClick={newChat} />
        <HeaderBtn
          icon={History}
          label="历史会话"
          onClick={() => (view === "history" ? setView("chat") : openHistory())}
        />
        <HeaderBtn
          icon={docked ? PictureInPicture2 : PanelRight}
          label={docked ? "浮动窗口" : "贴靠右侧"}
          onClick={onToggleDock}
        />
        <HeaderBtn icon={X} label="关闭" onClick={onClose} />
      </div>

      {/* 内容区 */}
      {view === "history" ? (
        <div className="flex-1 overflow-y-auto px-3 py-3">
          {historyLoading ? (
            <div className="flex justify-center pt-10">
              <Loader2 className="w-5 h-5 animate-spin text-[var(--color-text-tertiary)]" />
            </div>
          ) : convs.length === 0 ? (
            <p className="text-center text-[12px] text-[var(--color-text-tertiary)] pt-10">
              还没有历史会话
            </p>
          ) : (
            <div className="flex flex-col gap-1">
              {convs.map((c) => (
                <div
                  key={c.id}
                  className={cn(
                    "group flex items-center gap-2 rounded-lg px-3 py-2.5 cursor-pointer hover:bg-[var(--color-bg-muted)]",
                    convId === c.id && "bg-[var(--color-accent-soft)]",
                  )}
                  onClick={() => openConv(c.id)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-[13px] text-[var(--color-text-primary)] truncate">
                      {c.title}
                    </div>
                    <div className="text-[11px] text-[var(--color-text-tertiary)]">
                      {c.updated_at.slice(0, 16).replace("T", " ")}
                      {c.message_count != null && ` · ${c.message_count} 条消息`}
                    </div>
                  </div>
                  <button
                    type="button"
                    aria-label="删除会话"
                    className="opacity-0 group-hover:opacity-100 w-7 h-7 grid place-items-center rounded-md text-[var(--color-text-tertiary)] hover:text-[var(--color-danger,#dc2626)] hover:bg-white transition-all"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeConv(c.id);
                    }}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
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
                  <div key={i} className="flex gap-2 max-w-[94%]">
                    <AiAvatar size={24} />
                    <div className="flex flex-col gap-1.5 min-w-0">
                      {/* 工具调用 chips */}
                      {m.tools && m.tools.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {m.tools.map((s, j) => (
                            <span
                              key={`${s.callId}-${j}`}
                              className={cn(
                                "inline-flex items-center gap-1 text-[11px] rounded-full px-2 py-0.5 border",
                                s.status === "running" &&
                                  "border-[var(--color-accent)] text-[var(--color-accent)]",
                                s.status === "done" &&
                                  "border-[var(--color-border-subtle)] text-[var(--color-text-tertiary)]",
                                s.status === "error" &&
                                  "border-[var(--color-danger,#dc2626)] text-[var(--color-danger,#dc2626)]",
                              )}
                            >
                              {s.status === "running" ? (
                                <Loader2 className="w-3 h-3 animate-spin" />
                              ) : s.status === "done" ? (
                                <Check className="w-3 h-3" />
                              ) : (
                                <Wrench className="w-3 h-3" />
                              )}
                              {s.label}
                            </span>
                          ))}
                        </div>
                      )}
                      {m.text && (
                        <div className="bg-[var(--color-bg-muted)] text-[var(--color-text-primary)] text-[13px] leading-relaxed rounded-2xl rounded-tl-sm px-3.5 py-2 whitespace-pre-wrap">
                          {m.text}
                        </div>
                      )}
                      {/* 写操作确认卡 */}
                      {m.confirms?.map((c) => (
                        <div
                          key={c.callId}
                          className="rounded-xl border border-[var(--color-border-subtle)] bg-white px-3.5 py-3 shadow-[var(--shadow-card)]"
                        >
                          <div className="text-[12px] font-bold text-[var(--color-text-primary)]">
                            确认执行:{c.label}
                          </div>
                          {Object.keys(c.args).length > 0 && (
                            <pre className="mt-1.5 max-h-28 overflow-auto rounded-md bg-[var(--color-bg-muted)] px-2 py-1.5 text-[11px] text-[var(--color-text-secondary)] whitespace-pre-wrap break-all">
                              {JSON.stringify(c.args, null, 2)}
                            </pre>
                          )}
                          {!c.decided ? (
                            <div className="mt-2.5 flex gap-2">
                              <button
                                type="button"
                                onClick={() => decide(c.callId, true)}
                                className="flex-1 h-7 rounded-md bg-[var(--color-accent)] text-white text-[12px] font-bold hover:bg-[var(--color-accent-hover)] active:scale-[0.98] transition-all"
                              >
                                确认执行
                              </button>
                              <button
                                type="button"
                                onClick={() => decide(c.callId, false)}
                                className="flex-1 h-7 rounded-md border border-[var(--color-border-subtle)] text-[12px] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-muted)] active:scale-[0.98] transition-all"
                              >
                                取消
                              </button>
                            </div>
                          ) : (
                            <div className="mt-2 text-[11px] text-[var(--color-text-tertiary)]">
                              {c.decided === "approved" && "已确认,执行中…"}
                              {c.decided === "rejected" && "已取消"}
                              {c.decided === "timeout" && "确认超时,操作已取消"}
                            </div>
                          )}
                        </div>
                      ))}
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
      )}

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
          <div className="flex items-center justify-end gap-2 px-2 pb-2">
            {pending && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  stop();
                }}
                aria-label="停止生成"
                title="停止生成"
                className="w-8 h-8 rounded-full grid place-items-center border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-muted)] active:scale-[0.94] transition-all"
              >
                <Square className="w-3.5 h-3.5" />
              </button>
            )}
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
