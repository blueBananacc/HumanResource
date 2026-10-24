// Round 22: 上班时间 09→08 → 保存 → API 回读 → 还原
const task = await taskSpace(42);
const page = task.page("p1");
await page.fill("ref=413", "08");
await page.fill("ref=414", "30");
await page.click("loc=role:button[name*='保存日历']");
await page.waitForTimeout(1200);
const cal = await page.evaluate(() => {
  const token = localStorage.getItem("free-hire-token");
  return fetch("/api/work-calendar", { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json());
});
console.log("calendar after save:", cal);
// 还原 09:00
await page.fill("ref=413", "09");
await page.fill("ref=414", "00");
await page.click("loc=role:button[name*='保存日历']");
await page.waitForTimeout(1200);
const cal2 = await page.evaluate(() => {
  const token = localStorage.getItem("free-hire-token");
  return fetch("/api/work-calendar", { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json());
});
console.log("calendar restored:", cal2);
