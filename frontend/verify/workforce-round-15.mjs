// 我的考勤:点「申请补卡」→ modal → 提交;再看补卡申请列表
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.goto("http://localhost/me/attendance");
await page.waitForSelector("css=main h2", { timeout: 10_000 });
await page.click("loc=role:button[name*=补卡]");
await page.waitForSelector("text=漏打卡时补录", { timeout: 8_000 });
console.log("=== 补卡 modal ===");
console.log(await page.snapshot());
