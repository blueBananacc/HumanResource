// 批准 pending 补卡 → 状态变已批准;再去加班审批
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.fill("ref=73", "属实,补录");
await page.click("ref=74"); // 批准
await page.waitForSelector("text=已批准", { timeout: 8_000 });
console.log("=== 补卡已批 ===");
console.log(await page.snapshot());
