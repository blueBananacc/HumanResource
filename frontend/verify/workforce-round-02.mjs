// 续第一轮:已在 /attendance/calendar(已登录)。快照 + 点「添加条目」走通 modal。
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.waitForLoadState();
console.log(await page.snapshot());
