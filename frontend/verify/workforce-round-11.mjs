// 绑定员工 modal → 勾周凯 → 绑定 → 验证
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.goto("http://localhost/leave/policies");
await page.waitForSelector("css=main h2", { timeout: 10_000 });
await page.click("loc=role:button[name=绑定员工]");
await page.waitForSelector("text=绑定后立即按今年发放", { timeout: 8_000 });
await page.click('css=label:has-text("周凯")');
await page.click('css=button:has-text("绑定") >> nth=-1');
await page.waitForTimeout(1200);
console.log("=== 绑定后 ===");
console.log(await page.snapshot());
