// 添加 2026-12-25 圣诞节(放假)→ 列表出现 → 删除恢复
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.fill("ref=46", "2026-12-25");
await page.fill("ref=50", "圣诞节");
await page.click("ref=53"); // 保存
await page.waitForSelector("text=2026-12-25", { timeout: 8_000 });
console.log("=== 创建后 ===");
const snap1 = await page.snapshot();
console.log(snap1);
