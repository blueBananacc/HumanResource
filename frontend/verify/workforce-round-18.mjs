// 我的请假:加班调休区块 + 申请加班 modal + 申请请假的工作日自动算
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");
await page.goto("http://localhost/me/leave");
await page.waitForSelector("css=main h2", { timeout: 10_000 });

// 加班调休区块(emp1 有一单已批 8h→1 天)
const otText = await page.evaluate(
  () => document.body.textContent.match(/加班调休[\s\S]{0,300}/)?.[0] ?? "NONE",
);
console.log("OT section:", otText);

// 申请加班 modal
await page.click("loc=role:button[name*=申请加班]");
await page.waitForSelector("text=主管批准后按日标准工时折算", {
  timeout: 8_000,
});
console.log("=== 加班 modal ===");
console.log(await page.snapshot());
