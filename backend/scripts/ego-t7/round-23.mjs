// Round 23: type=time input 填写 + 保存 + API 回读 + 还原
const task = await taskSpace(42);
const page = task.page("p1");
await page.fill("loc=css:input[type=time] >> nth=0", "08:30");
await page.click("loc=role:button[name*='保存日历']");
await page.waitForTimeout(1200);
const cal = await page.evaluate(() => {
  const token = localStorage.getItem("free-hire-token");
  return fetch("/api/work-calendar", { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json());
});
console.log("calendar after save:", cal);
await page.fill("loc=css:input[type=time] >> nth=0", "09:00");
await page.click("loc=role:button[name*='保存日历']");
await page.waitForTimeout(1200);
const cal2 = await page.evaluate(() => {
  const token = localStorage.getItem("free-hire-token");
  return fetch("/api/work-calendar", { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json());
});
console.log("calendar restored:", cal2);
