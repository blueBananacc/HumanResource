// 节假日历:添加条目 modal → 创建 → 验证出现 → 删除
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.click("ref=27"); // 添加条目
await page.waitForSelector("text=添加日历条目", { timeout: 8_000 });
console.log("=== modal 打开 ===");
console.log(await page.snapshot());
