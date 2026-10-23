// 诊断:date input 值是否真的进 React state + workday-count API 直查
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");
await page.goto("http://localhost/me/leave");
await page.waitForSelector("css=main h2", { timeout: 10_000 });
await page.click("loc=role:button[name*=申请请假]");
await page.waitForSelector("text=提交后进入主管审批", { timeout: 8_000 });

await page.fill("css=input[name=start_date]", "2026-11-06");
await page.fill("css=input[name=end_date]", "2026-11-09");
await page.waitForTimeout(800);
const vals = await page.evaluate(() => ({
  start: document.querySelector("input[name=start_date]")?.value,
  end: document.querySelector("input[name=end_date]")?.value,
}));
console.log("input vals:", vals);

const token = await page.evaluate(() =>
  localStorage.getItem("free-hire-token"));
const wc = await page.fetch(
  "/api/me/workday-count?start=2026-11-06&end=2026-11-09",
  { headers: { authorization: `Bearer ${token}` } },
);
console.log("workday-count:", wc.status, wc.body);
