// 全周末区间 → 「没有工作日」警告;截图请假 modal + 管理端节假日历
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.evaluate(() => {
  function setDate(name, v) {
    const el = document.querySelector(`input[name=${name}]`);
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, "value",
    ).set;
    setter.call(el, v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }
  setDate("start_date", "2026-11-07");
  setDate("end_date", "2026-11-08");
});
await page.waitForTimeout(1200);
const warn = await page.evaluate(
  () => document.body.textContent.includes("没有工作日"),
);
console.log("zero-workday warning visible:", warn);
console.log({ shot: await page.screenshot() });
