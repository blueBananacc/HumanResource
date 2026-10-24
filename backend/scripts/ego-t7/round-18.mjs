// Round 18: 表头 + 考勤规则 tab
const task = await taskSpace(42);
const page = task.page("p1");
const headers = await page.evaluate(() =>
  [...document.querySelectorAll("thead th, thead td")].map((h) => h.textContent?.trim()));
console.log("headers:", headers);
await page.click("loc=role:button[name*='考勤规则']");
await page.waitForTimeout(1200);
console.log("=== 考勤规则 ===");
console.log(await page.snapshot());
