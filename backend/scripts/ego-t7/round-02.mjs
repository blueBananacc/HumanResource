// Round 2: /settings/automations 页面快照
const task = await taskSpace(42);
const page = task.page("p1");
await page.goto("http://localhost/settings/automations");
await page.waitForLoadState();
await page.waitForTimeout(1200);
console.log("url:", await page.url());
console.log(await page.snapshot());
