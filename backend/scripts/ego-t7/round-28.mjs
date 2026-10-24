// Round 28: employee 访问管理页 → 权限拒绝表现
const task = await taskSpace(42);
const page = task.page("p1");
await page.goto("http://localhost/attendance/flags");
await page.waitForLoadState();
await page.waitForTimeout(1200);
console.log("flags url:", await page.url());
const body1 = await page.evaluate(() => document.body.innerText.slice(0, 400));
console.log("flags body:", body1);
await page.goto("http://localhost/settings/automations");
await page.waitForLoadState();
await page.waitForTimeout(1200);
console.log("automations url:", await page.url());
const body2 = await page.evaluate(() => document.body.innerText.slice(0, 400));
console.log("automations body:", body2);
