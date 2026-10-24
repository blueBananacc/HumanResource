// Round 25: 还原 09:00
const task = await taskSpace(42);
const page = task.page("p1");
await page.evaluate(() => {
  const input = document.querySelector("input[type=time]");
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
  setter.call(input, "09:00");
  input.dispatchEvent(new Event("input", { bubbles: true }));
});
await page.click("loc=role:button[name*='保存日历']");
await page.waitForTimeout(1200);
const cal = await page.evaluate(() => {
  const token = localStorage.getItem("free-hire-token");
  return fetch("/api/work-calendar", { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json());
});
console.log("restored:", cal);
