// Round 8: 展开评分卡 → 反馈明细 + AI 汇总按钮
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/talents/9f282fd4-1b2c-4dc2-938d-5de3d32871a8");
await page.waitForLoadState();
await page.waitForTimeout(1200);

// 展开
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.getAttribute("aria-label") === "展开")?.click();
});
await page.waitForTimeout(800);
const snap = await page.snapshot({ scope: "full_page" });
const idx = snap.indexOf("T9验证");
console.log(snap.slice(idx, idx + 3400));
