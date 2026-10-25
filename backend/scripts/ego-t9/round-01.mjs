// T9 UI 验证 Round 1: 候选人详情 → 评分卡区块渲染
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

// 确认登录态;没有则登录
await page.goto("http://localhost/");
await page.waitForLoadState();
if ((await page.url()).includes("/login")) {
  await page.fill("loc=css:input", "admin@example.com");
  await page.fill("loc=css:input[type=password]", "admin123456");
  await page.click("loc=css:button[type=submit]");
  await page.waitForURL(/^(?!.*login).*$/, { timeout: 15000 });
}

const data = await page.evaluate(async () => {
  const t = localStorage.getItem("free-hire-token");
  const r = await fetch("/api/talents?limit=3&offset=0", {
    headers: t ? { Authorization: `Bearer ${t}` } : {},
  });
  return { status: r.status, body: await r.text() };
});
console.log("status:", data.status);
const parsed = JSON.parse(data.body);
const items = parsed.items ?? parsed;
console.log("count:", items.length, "first:", items[0]?.id, items[0]?.name);

await page.goto(`http://localhost/talents/${items[0].id}`);
await page.waitForLoadState();
await page.waitForTimeout(1500);
console.log("url:", await page.url());
const snap = await page.snapshot({ scope: "full_page" });
const idx = snap.indexOf("评分卡");
console.log("has scorecard section:", idx >= 0);
console.log(snap.slice(Math.max(0, idx - 300), idx + 1600));
