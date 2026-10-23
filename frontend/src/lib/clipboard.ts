/**
 * 复制文本到剪贴板,带 insecure context 兜底。
 *
 * ``navigator.clipboard`` 仅在 secure context(HTTPS / localhost)可用;
 * 私有化部署常走 http://内网 IP,此时退到 ``document.execCommand("copy")``。
 * 成功时调用 ``onDone``;两条路径都失败时静默 — 调用方可提供手动复制入口。
 */
export function copyText(text: string, onDone?: () => void) {
  if (window.isSecureContext && navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).then(onDone).catch(legacyCopy);
    return;
  }
  legacyCopy();

  function legacyCopy() {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
      if (document.execCommand("copy")) onDone?.();
    } catch {
      /* 静默失败 */
    } finally {
      document.body.removeChild(ta);
    }
  }
}
