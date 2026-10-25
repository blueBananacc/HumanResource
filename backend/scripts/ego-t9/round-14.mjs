// Round 14: 清理 —— UI 删除评分卡(验证删除流) → finish
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/talents/9f282fd4-1b2c-4dc2-938d-5de3d32871a8");
await page.waitForLoadState();
await page.waitForTimeout(1200);
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.getAttribute("aria-label") === "展开")?.click();
});
await page.waitForTimeout(600);
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.getAttribute("aria-label") === "删除评分卡")?.click();
});
await page.waitForTimeout(700);
// 确认弹窗
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => /^(删除|确认|确定)/.test(x.textContent.trim()) && x.offsetParent)?.click();
});
await page.waitForTimeout(1200);
const snap = await page.snapshot({ scope: "full_page" });
console.log("card gone:", !snap.includes("T9验证"));
console.log("empty state:", snap.includes("还没有评分卡"));

await finish({ keep: [] });
