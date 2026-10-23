// 补卡 modal 填 2026-10-20 09:10 上班卡 → 提交 → 列表出现
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.fill("ref=395", "2026-10-20T09:10");
await page.fill("ref=402", "打卡机故障,当时找前台登记过");
await page.click("ref=404"); // 提交
await page.waitForTimeout(1500);
console.log("=== 提交后 ===");
console.log(await page.snapshot());
