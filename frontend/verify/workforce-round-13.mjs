// ESS 侧:登出 admin → emp1 登录 → 我的考勤 → 申请补卡 modal 提交
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

// 关闭 modal + 登出
await page.keyboard.press("Escape");
await page.goto("http://localhost/");
await page.evaluate(() => localStorage.removeItem("free-hire-token"));
await page.goto("http://localhost/login");
await page.waitForLoadState();
await page.fill("input[type=email]", "emp1@hronfire.dev");
await page.fill("input[type=password]", "Emp@123456");
await page.click("loc=role:button[name*=登录]");
await page.waitForURL(/\/me/, { timeout: 10_000 });
await page.waitForLoadState();
console.log("logged in as emp1");

await page.goto("http://localhost/me/attendance");
await page.waitForSelector("text=我的考勤", { timeout: 10_000 });
console.log("=== 我的考勤 ===");
console.log(await page.snapshot());
