// 用 native setter + input 事件驱动 React onChange,再验证 workday 提示
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");
await page.goto("http://localhost/me/leave");
await page.waitForSelector("css=main h2", { timeout: 10_000 });
await page.click("loc=role:button[name*=申请请假]");
await page.waitForSelector("text=提交后进入主管审批", { timeout: 8_000 });

await page.evaluate(() => {
  function setDate(name, v) {
    const el = document.querySelector(`input[name=${name}]`);
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, "value",
    ).set;
    setter.call(el, v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }
  setDate("start_date", "2026-11-06");
  setDate("end_date", "2026-11-09");
});
await page.waitForTimeout(1200);
const label = await page.evaluate(
  () =>
    [...document.querySelectorAll("label")]
      .map((l) => l.textContent)
      .filter((t) => t.includes("工作日") || t.includes("天数"))[0] ??
      "NONE",
);
console.log("days label:", label);
const placeholder = await page.evaluate(
  () => document.querySelector("input[name=days]")?.placeholder ?? "NONE",
);
console.log("placeholder:", placeholder);
