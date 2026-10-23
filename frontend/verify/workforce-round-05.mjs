// 删除刚加的圣诞条目 → 去补卡审批页 → 批准 seed 的 pending 补卡
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.click("ref=54"); // 删除圣诞节
await page.waitForFunction(
  () => !document.body.textContent.includes("2026-12-25"),
  undefined,
  { timeout: 8_000 },
);
console.log("deleted ok");

await page.goto("http://localhost/attendance/requests");
await page.waitForLoadState();
await page.waitForSelector("css=main h2", { timeout: 10_000 });
console.log("=== 补卡审批 ===");
console.log(await page.snapshot());
