// Round 9: AI 聚合(mock LLM → 机械兜底) → 关闭评分卡
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
  [...document.querySelectorAll("button")].find((x) => x.textContent.includes("AI 聚合"))?.click();
});
await page.waitForTimeout(3500);

const snap = await page.snapshot({ scope: "full_page" });
const idx = snap.indexOf("AI 汇总");
const idx2 = snap.indexOf("汇总");
const start = Math.max(idx, idx2);
console.log("summary idx:", start);
if (start >= 0) console.log(snap.slice(start - 100, start + 2000));
else {
  const i = snap.indexOf("T9验证");
  console.log(snap.slice(i, i + 3000));
}
