// Round 10: 全局搜索 Ctrl+K → 跨实体分组+facet+导航
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/");
await page.waitForLoadState();
await page.waitForTimeout(1200);

// Ctrl+K 打开命令面板
await page.press("loc=css:body", "Control+k");
await page.waitForTimeout(600);

const snap0 = await page.snapshot({ scope: "full_page" });
const si = snap0.indexOf("搜索");
console.log("palette head:", snap0.slice(Math.max(0, si - 200), si + 1200));

// 输入跨实体关键词(赵六 是候选人名)
await page.fill("loc=css:input[placeholder*='搜索']", "赵六");
await page.waitForTimeout(1600);
const snap = await page.snapshot({ scope: "full_page" });
const ri = snap.indexOf("赵六");
console.log("--- results ---");
console.log(snap.slice(Math.max(0, ri - 600), ri + 2200));
