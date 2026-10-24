// Round 9: 当前列表状态
const task = await taskSpace(42);
const page = task.page("p1");
await page.reload();
await page.waitForLoadState();
await page.waitForTimeout(1500);
console.log(await page.snapshot());
