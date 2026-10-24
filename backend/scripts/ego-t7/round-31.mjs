// Round 31: 当前状态快照
const task = await taskSpace(42);
const page = task.page("p1");
console.log("url:", await page.url());
console.log(await page.snapshot());
