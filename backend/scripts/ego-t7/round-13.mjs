// Round 13: evaluate 关抽屉 → 启停开关验证
const task = await taskSpace(42);
const page = task.page("p1");
const closed = await page.evaluate(() => {
  const btns = [...document.querySelectorAll("button")].filter((b) => b.textContent?.trim() === "关闭");
  if (btns.length) { btns[btns.length - 1].click(); return btns.length; }
  return 0;
});
console.log("close clicked:", closed);
await page.waitForTimeout(500);
const before = await page.evaluate(() => {
  const sw = document.querySelector("button[role=switch]");
  return sw ? sw.getAttribute("data-state") || sw.getAttribute("aria-checked") || "switch-no-state" : "no-switch";
});
console.log("toggle before:", before);
if (before !== "no-switch") {
  await page.click("loc=css:button[role=switch] >> nth=0");
  await page.waitForTimeout(800);
  const after = await page.evaluate(() => {
    const sw = document.querySelector("button[role=switch]");
    return sw?.getAttribute("data-state") || sw?.getAttribute("aria-checked");
  });
  console.log("toggle after:", after);
  const badge = await page.evaluate(() => {
    const card = [...document.querySelectorAll("div")].find((d) => d.textContent?.includes("T7-UI验证规则"));
    return card?.textContent?.slice(0, 200);
  });
  console.log("card after off:", badge);
  // 恢复开启
  await page.click("loc=css:button[role=switch] >> nth=0");
  await page.waitForTimeout(500);
  console.log("toggled back on");
}
