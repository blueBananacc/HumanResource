// 申请请假 modal:跨周末区间 → 「区间内 N 个工作日」提示出现
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");
await page.goto("http://localhost/me/leave");
await page.waitForSelector("css=main h2", { timeout: 10_000 });

await page.click("loc=role:button[name*=申请请假]");
await page.waitForSelector("text=提交后进入主管审批", { timeout: 8_000 });
// Fri→Mon:2026-11-06 ~ 2026-11-09(周五到周一)= 2 个工作日
await page.fill("css=input[name=start_date]", "2026-11-06");
await page.fill("css=input[name=end_date]", "2026-11-09");
await page.waitForTimeout(1000);
const label = await page.evaluate(
  () =>
    [...document.querySelectorAll("label")]
      .map((l) => l.textContent)
      .filter((t) => t.includes("工作日"))[0] ?? "NONE",
);
console.log("days label:", label);
const placeholder = await page.evaluate(
  () =>
    document.querySelector("input[name=days]")?.placeholder ?? "NONE",
);
console.log("days placeholder:", placeholder);
// 全周末区间 → 0 工作日提示
await page.fill("css=input[name=start_date]", "2026-11-07");
await page.fill("css=input[name=end_date]", "2026-11-08");
await page.waitForTimeout(1000);
const warn = await page.evaluate(
  () =>
    [...document.querySelectorAll("div")]
      .map((d) => d.textContent)
      .filter((t) => t && t.includes("没有工作日"))[0] ?? "NONE",
);
console.log("weekend warning:", warn ? warn.slice(0, 60) : "NONE");
