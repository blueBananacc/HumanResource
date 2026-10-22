/** LLM 回复的 markdown 渲染。
 *
 * - react-markdown 输出 React 组件树而非 HTML 字符串;不加 rehype-raw 时
 *   LLM 输出里的原始 HTML 一律按文本显示,天然防注入;默认 urlTransform
 *   拦截 javascript:/data: 等危险链接协议。
 * - remark-gfm 提供表格/删除线等 GFM 语法(系统提示词让 LLM 输出表格)。
 * - components 逐个映射到本项目的 --color-* 设计变量,不用 typography
 *   插件,气泡内排版保持 13px 密排风格。
 * - memo 包住:流式回复时只有最后一条消息会因 text 变化重解析。
 */
import { memo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

const components: Components = {
  p: ({ children }) => <p className="my-1.5 first:mt-0 last:mb-0">{children}</p>,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-[var(--color-accent)] underline underline-offset-2 hover:text-[var(--color-accent-hover)]"
    >
      {children}
    </a>
  ),
  strong: ({ children }) => <strong className="font-bold">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  del: ({ children }) => <del className="opacity-70">{children}</del>,
  ul: ({ children }) => (
    <ul className="my-1.5 list-disc space-y-0.5 pl-5 marker:text-[var(--color-text-tertiary)]">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="my-1.5 list-decimal space-y-0.5 pl-5 marker:text-[var(--color-text-tertiary)]">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="[&>p]:my-0">{children}</li>,
  h1: ({ children }) => (
    <h1 className="my-2 text-[15px] font-bold first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="my-2 text-[14px] font-bold first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="my-1.5 text-[13px] font-bold first:mt-0">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="my-1.5 text-[13px] font-bold first:mt-0">{children}</h4>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-1.5 border-l-2 border-[var(--color-border-subtle)] pl-3 text-[var(--color-text-secondary)] [&>p]:my-0.5">
      {children}
    </blockquote>
  ),
  code: ({ className, children }) => (
    <code
      className={cn(
        "rounded bg-white/70 px-1 py-0.5 font-mono text-[12px]",
        className,
      )}
    >
      {children}
    </code>
  ),
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-lg bg-[#2b2b33] p-3 font-mono text-[12px] leading-relaxed text-white [&>code]:bg-transparent [&>code]:p-0 [&>code]:text-inherit">
      {children}
    </pre>
  ),
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto rounded-lg border border-[var(--color-border-subtle)]">
      <table className="w-full border-collapse text-[12px]">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-white/60">{children}</thead>,
  tbody: ({ children }) => (
    <tbody className="[&>tr:last-child>td]:border-b-0">{children}</tbody>
  ),
  th: ({ children }) => (
    <th className="border-b border-[var(--color-border-subtle)] px-2.5 py-1.5 text-left font-bold text-[var(--color-text-secondary)]">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border-b border-[var(--color-border-subtle)] px-2.5 py-1.5 align-top">
      {children}
    </td>
  ),
  hr: () => <hr className="my-2 border-[var(--color-border-subtle)]" />,
};

export const MarkdownContent = memo(function MarkdownContent({
  text,
}: {
  text: string;
}) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {text}
    </ReactMarkdown>
  );
});
