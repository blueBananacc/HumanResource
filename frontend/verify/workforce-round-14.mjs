// 我的考勤页快照 + 点「申请补卡」→ 填表提交
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.goto("http://localhost/me/attendance");
await page.waitForSelector("css=main h2", { timeout: 10_000 });
console.log("=== 我的考勤 ===");
console.log(await page.snapshot());
