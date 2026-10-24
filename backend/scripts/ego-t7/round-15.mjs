// Round 15: /attendance/flags 三 tab 页
const task = await taskSpace(42);
const page = task.page("p1");
await page.goto("http://localhost/attendance/flags");
await page.waitForLoadState();
await page.waitForTimeout(1500);
console.log("url:", await page.url());
console.log(await page.snapshot());
