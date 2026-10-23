// 批准王浩的加班 → 已批准 → 假期额度页验证调休入账 → 假期政策页
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.fill("ref=121", "辛苦了");
await page.click("ref=122"); // 批准
await page.waitForSelector("text=审批人", { timeout: 8_000 });
console.log("approved");

// 假期额度页 — 王浩调休假应 +1 天
await page.goto("http://localhost/leave/balances");
await page.waitForLoadState();
await page.waitForSelector("css=main h2", { timeout: 10_000 });
const snap = await page.snapshot();
console.log(snap);
