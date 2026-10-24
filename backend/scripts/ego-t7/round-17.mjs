// Round 17: 月度汇总 tab + 考勤规则 tab
const task = await taskSpace(42);
const page = task.page("p1");
await page.click("loc=role:button[name='月度汇总']");
await page.waitForTimeout(1200);
console.log("=== 月度汇总 ===");
console.log(await page.snapshot());
