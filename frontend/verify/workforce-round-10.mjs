// 政策页:绑定员工 modal → 勾选一个未绑定员工 → 提交;再开年度发放 modal
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.goto("http://localhost/leave/policies");
await page.waitForSelector("css=main h2", { timeout: 10_000 });
await page.click("loc=role:button[name=绑定员工]");
await page.waitForSelector("text=绑定后立即按今年发放", { timeout: 8_000 });
console.log("=== 绑定员工 modal ===");
console.log(await page.snapshot());
