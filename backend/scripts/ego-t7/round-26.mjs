// Round 26: ESS 登录 → /me/attendance
const task = await taskSpace(42);
const page = task.page("p1");
// 清 session 再以员工登录
await page.evaluate(() => localStorage.clear());
await page.goto("http://localhost/login");
await page.waitForLoadState();
await page.fill("loc=css:input[type=email], input[name=email] >> nth=0", "t7-ess-9c9d66@example.com");
await page.fill("loc=css:input[type=password]", "T7ess!pass");
await page.click("loc=role:button[name*='登'] >> nth=0");
await page.waitForTimeout(2000);
console.log("url:", await page.url());
await page.goto("http://localhost/me/attendance");
await page.waitForLoadState();
await page.waitForTimeout(1500);
console.log("url:", await page.url());
console.log(await page.snapshot());
