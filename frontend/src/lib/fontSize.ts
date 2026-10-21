// 页面字号偏好。档位依据 docs/design/hibob-style-reference.md §4:
// 基础字号 14px、大字号 16px;另加 18px 作为低视力友好档(§7 第 8 条)。
// 不提供 <14px:规范为无障碍把基础字号从 12 提到 14。
//
// 项目里大量字号写死为 px(text-[13px] 等),改根 font-size 只会放大 rem 部分、
// 打乱层级,所以用 zoom 对整个应用等比缩放(文字/图标/间距同步)。
export const FONT_SIZES = [
  { value: 14, label: "标准", hint: "默认" },
  { value: 16, label: "大", hint: "阅读更轻松" },
  { value: 18, label: "特大", hint: "低视力友好" },
] as const;

export type FontSize = (typeof FONT_SIZES)[number]["value"];

const KEY = "hronfire.ui.fontSize";
const BASE = 14;

export function getFontSize(): FontSize {
  const v = Number(localStorage.getItem(KEY));
  return FONT_SIZES.some((o) => o.value === v) ? (v as FontSize) : BASE;
}

export function applyFontSize(size: FontSize) {
  document.documentElement.style.zoom = size === BASE ? "" : String(size / BASE);
}

export function setFontSize(size: FontSize) {
  localStorage.setItem(KEY, String(size));
  applyFontSize(size);
}
