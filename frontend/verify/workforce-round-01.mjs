// 第一轮:admin 登录 → 假勤导航 → 节假日历 / 补卡审批 / 加班审批 / 假期政策
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");
console.log({ spaceId: task.spaceId });

await page.goto("http://localhost/");
await page.waitForLoadState();

// 登录
await page.fill("input[type=email]", "admin@example.com");
await page.fill("input[type=password]", "admin123456");
await page.click("loc=role:button[name*=登录]");
await page.waitForURL("http://localhost/");
await page.waitForLoadState();

// 1) 节假日历页
await page.goto("http://localhost/attendance/calendar");
await page.waitForLoadState();
await page.waitForSelector("text=节假日历", { timeout: 10_000 });
await page.waitForSelector("text=调休补班", { timeout: 10_000 });
console.log("=== 节假日历页 ===");
console.log(await page.snapshot());
console.log({ shot: await page.screenshot() });
