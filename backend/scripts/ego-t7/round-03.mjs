// Round 3: 新 bundle 下的 /settings/automations
const task = await taskSpace(42);
const page = task.page("p1");
await page.goto("http://localhost/settings/automations");
await page.waitForLoadState();
await page.waitForTimeout(1500);
console.log("url:", await page.url());
console.log(await page.snapshot());
