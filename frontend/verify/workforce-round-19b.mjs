// 看 Select 下拉 DOM 结构
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");
await page.goto("http://localhost/me/leave");
await page.waitForSelector("css=main h2", { timeout: 10_000 });
await page.click("loc=role:button[name*=申请加班]");
await page.waitForSelector("text=主管批准后按日标准工时折算", { timeout: 8_000 });
await page.click('css=button:has-text("选择调休类型")');
await page.waitForTimeout(600);
console.log(await page.snapshot());
