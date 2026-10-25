const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");
await page.goto("http://localhost/login");
await page.waitForLoadState();
await page.waitForTimeout(1200);
console.log("url:", await page.url());
console.log(await page.snapshot());
