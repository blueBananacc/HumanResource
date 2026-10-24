// Round 12b: 关抽屉 → 启停开关验证(PATCH is_active)
const task = await taskSpace(42);
const page = task.page("p1");
await page.click('loc=role:button[name*="关闭"] >> nth=0');
await page.waitForTimeout(500);
const before = await page.evaluate(() => {
  const sw = document.querySelector("button[role=switch]");
  return sw ? sw.getAttribute("data-state") || sw.getAttribute("aria-checked") : "no-switch";
});
console.log("toggle before:", before);
await page.click("loc=css:button[role=switch] >> nth=0");
await page.waitForTimeout(800);
const after = await page.evaluate(() => {
  const sw = document.querySelector("button[role=switch]");
  return sw?.getAttribute("data-state") || sw?.getAttribute("aria-checked");
});
console.log("toggle after:", after);
// 恢复开启
await page.click("loc=css:button[role=switch] >> nth=0");
await page.waitForTimeout(500);
