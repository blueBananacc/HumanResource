// Round 24: native setter + input event 驱动 React 受控 time input
const task = await taskSpace(42);
const page = task.page("p1");
const set = await page.evaluate(() => {
  const input = document.querySelector("input[type=time]");
  if (!input) return "no-input";
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
  setter.call(input, "08:30");
  input.dispatchEvent(new Event("input", { bubbles: true }));
  return input.value;
});
console.log("input value now:", set);
await page.click("loc=role:button[name*='保存日历']");
await page.waitForTimeout(1200);
const cal = await page.evaluate(() => {
  const token = localStorage.getItem("free-hire-token");
  return fetch("/api/work-calendar", { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json());
});
console.log("calendar after save:", cal);
