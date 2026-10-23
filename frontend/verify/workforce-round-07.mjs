// 加班审批页:看 pending → 批准 → 验证调休入账;再去假期政策页
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.goto("http://localhost/overtime");
await page.waitForLoadState();
await page.waitForSelector("css=main h2", { timeout: 10_000 });
console.log("=== 加班审批 ===");
console.log(await page.snapshot());
