# HiBob 前端风格与交互参考

本文汇总 HROnfire 前端风格改版时参考的 HiBob（Bob）公开资料，以及在代码中的落地方式，避免以后重复检索。

## 1. 资料来源

| 资料 | 地址 | 说明 |
|------|------|------|
| Brand Guidelines 2024 V2（PDF，60 页） | https://www.hibob.com/wp-content/uploads/BrandGuidelines-2024V2-compressed.pdf | 颜色、字体、形状、纹理、插画、产品 UI 示例 |
| The Great Bob Makeover（官方博文） | https://www.hibob.com/blog/bob-design-makeover-enhancing-ux-and-ui/ | 布局、无障碍、白标定制规则（交互细节最有价值） |
| HiBob Brand Portal | https://brand.hibob.com | JS 渲染站点，内容与 PDF 基本一致，另有 Motion Guidelines（未能打开，动效为近似实现） |
| 真实产品截图 | https://www.hrisresearch.com/hr-software/hibob | 员工主页、经理主页、档案、薪酬工作表等 |

## 2. 设计原则（官方）

四条：**Clarity（清晰）> Efficiency（高效）> Consistency（一致）> Beauty（美观）**。
品牌气质：Organic（有机）、Modern、Purposeful、Global；语气 Personable / Savvy / Supportive / Inclusive。

## 3. 颜色

| 名称 | 色值 | 用途 |
|------|------|------|
| Cherry Syrup | `#EE164F` | 主色：主按钮、强调、徽标 |
| Dark Wine | `#83143D` | 侧栏 |
| Dark Wine 深 | `#60032A` | 顶栏 |
| Orange Juice | `#FAA32B` | 辅助强调：选中指示条、计数徽章、头像底色 |
| Cappuccino Foam | `#FDF6EB` | 浅底 |
| 页面底 | `#FFFBF4` | 页面背景 |
| Black Coffee | `#3A3A37` | 正文文字 |

Tints & shades：`#BB1141`（Cherry 深，hover）、`#9C2443`、`#F36488`、`#FEABB0`、`#FAC7D1`（浅粉）、`#F88400`、`#FEBB59`、`#FFD887`（浅橙）、`#F7E7CE`。
官方提醒：tints 要谨慎使用；颜色组合可大胆也可克制，重点是平衡。

代码落地：`frontend/src/index.css` 的 `:root` CSS 变量（`--color-accent`、`--color-brand`、`--color-brand-deep`、`--color-orange` 等）。

## 4. 字体

| 角色 | 官方字体 | 免费替代（官方 G Suite 替代方案） | 本项目 |
|------|----------|-----------------------------------|--------|
| 标题 | Champion（全大写，仅标题） | Archivo Black | 未使用 |
| 小标题 / 引用 / 数字 | Sentinel（Book/Medium/Semibold/Black/Italic） | Domine | `.font-heading` |
| 正文 | Gotham（Book/Medium/Bold） | Lato | `.font-body` / body |

- 官方字体均为商用授权，项目使用替代字体，通过 `@fontsource/domine`、`@fontsource/lato`（5.3.0，固定版本）自托管，见 `src/main.tsx`。中文回落系统字体。
- 无障碍改版：基础字号 12px→14px，大字号 14px→16px（本项目 body 14px）。
- 官方标题规则：每行 1–3 词，最多 3 行。

## 5. 形状、纹理与插画

- 偏好圆润柔和的曲线，传达亲和与温暖；圆形是第一个形状。
- 主形状 6 种：Circular、Rectangular、Triangular、Curvilinear（弧线）、Star、Freeform。
- 纹理（木/布/石/金属）用于背景或形状内部；图案（Waves、Stripes、Checkers、Curls、Blobs、Sparks）作为插画点缀。
- 拼贴（Collage）：黑白照片 + 插画 + 纹理 + 颗粒。
- 产品图：始终配真人，产品 UI 元素（卡片、按钮）浮在彩色块之上。

代码落地：`frontend/src/components/brand/BrandShape.tsx`（`kind`: circle/rect/triangle/arc/star/free），用于登录页背景和首页横幅。

## 6. Logo

- HiBubble（对话气泡形状）仅在 logo 太小时使用（favicon 等），不替代完整 logo。
- 浅底用黑色 logo，深色/彩色底用白色 logo；logo 避免右对齐。
- 本项目：气泡形圆角方块 + “Hi” + 字标 HROnfire（`rounded-[12px_12px_12px_3px]`）。

## 7. 布局与交互规则（改版博文）

1. 所有导航集中在左侧，右侧最大化工作区。
2. 去掉工作区周围的背景图；自定义背景图/Logo 只作为**首页横幅**出现，进入各模块后消失。
3. 导航栏可展开（显示全名）或收起（仅图标），收起后获得更大工作区。
4. 个人资料、搜索、设置原本在右侧，改版后并入左侧菜单，统一为一个导航区域。
5. 各模块的设置放在对应模块内，而不是集中在公司设置里。
6. 菜单从多色改为**单色**，文字/图标与背景保持高对比度。
7. 按钮三级：primary / secondary / tertiary，各有颜色规则以保证对比度和层级。
8. 目标符合 WCAG 2.2；提供低视力友好的调色板。
9. 白标：Bob style / Pop style 两套调色板；超过 50% 客户启用了白标。

## 8. 产品界面细节（来自截图与 PDF）

- 侧栏窄、图标式、单色渐进色块；顶栏白色，右侧：搜索、铃铛（橙色计数徽章）、头像、全屏按钮。
- 首页问候语用衬线字体：“Hi, glad you're here 👋”。
- 内容卡片：白底、细边框、小圆角、柔和阴影；卡片标题小号加粗，底部居中“View all”链接。
- 右侧小组件：New hires、Anniversaries、People's status（头像色块堆叠）。
- 页面标题和统计数字使用衬线字体；表格表头小号粗体，状态用带色圆点的胶囊标签。
- 标签页（tab）：小号大写文字 + 选中项红色下划线。
- 详情页：居中头像 + 衬线姓名，完成状态用绿色对勾徽章，待处理用灰色时钟徽章。
- 语气：口语化、友好、不说教，避免“你应该”“唯一正确的方式”之类措辞。

## 9. HROnfire 落地对照

| 规范 | 实现位置 |
|------|----------|
| 颜色/字体/阴影变量 | `frontend/src/index.css` |
| 三级按钮 | `frontend/src/components/ui/button.tsx` |
| 卡片（12px 圆角 + `--shadow-card`） | `components/ui/card.tsx` 及各页面内联卡片 |
| 顶栏 | `components/layout/Topbar.tsx` |
| 可折叠侧栏（状态存 localStorage `hronfire.sidebar.collapsed`） | `components/layout/Sidebar.tsx` |
| 布局（顶栏全宽 + 左导航 + 右工作区） | `components/layout/AppShell.tsx` |
| 首页横幅与卡片头部 | `pages/Dashboard.tsx` |
| 登录页品牌背景 | `pages/Login.tsx` |
| 字体大小设置（14 标准 / 16 大 / 18 特大，zoom 等比缩放，存 localStorage `hronfire.ui.fontSize`） | `lib/fontSize.ts`、`components/layout/SettingsModal.tsx`（入口：帮助与支持 → 设置） |
| 导航定义（侧栏与全局搜索共用） | `components/layout/nav.ts` |
| 顶栏全局搜索（功能页 + 候选人，Ctrl/⌘+K） | `components/layout/GlobalSearch.tsx` |
| 右侧工具栏（顶部 AI 助手，底部帮助与支持） | `components/layout/RightRail.tsx` |
| AI 助手面板（浮动可拖动 / 拖到右缘或点按钮贴边；后端未接入，占位回复） | `components/layout/AiAssistant.tsx`（HiBubble 气泡头像 `AiAvatar`） |

## 10. 待办 / 已知取舍

- 白字配 Cherry `#EE164F` 对比度约 4:1，低于 WCAG AA 小字 4.5:1；如需达标，可把按钮色加深（如 `#D9123F` 或使用 `#BB1141`）。
- 尚未实现：Bob 风格 tab、页面内状态标签统一、个人入口并入左侧菜单（当前保留顶栏）。
- 搜索参照 Bob 真实产品放在顶栏居中（“Search for people or features”），而非改版博文所说的左侧菜单。
- 动效参照 Motion Guidelines 未获取，现仅使用 200–300ms 缓动过渡与按下缩放。

## 11. 本地预览

```bash
cd frontend
npm ci
VITE_API_PROXY=http://localhost npx vite --port 5173   # 复用 Docker 中运行的后端
```

定稿后：`cd deploy && docker compose up -d --build frontend`。
