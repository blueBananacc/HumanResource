// Round 21: /attendance/calendar 上下班时间字段
const task = await taskSpace(42);
const page = task.page("p1");
await page.goto("http://localhost/attendance/calendar");
await page.waitForLoadState();
await page.waitForTimeout(1500);
console.log(await page.snapshot());
