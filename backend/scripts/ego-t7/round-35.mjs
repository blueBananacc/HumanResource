// Round 35: 精确点 bad-62455a 卡内第4按钮(删除) → 处理确认框 → 验证
const task = await taskSpace(42);
const page = task.page("p1");
await page.evaluate(() => {
  const cards = [...document.querySelectorAll("[class]")].filter(
    (el) => el.childElementCount >= 1 && el.textContent?.includes("bad-62455a"));
  const card = cards.sort((a, b) => a.textContent.length - b.textContent.length)[0];
  card?.querySelectorAll("button")?.[3]?.click();
});
await page.waitForTimeout(700);
// 是否有原生 confirm/对话框按钮
const dlgBtns = await page.evaluate(() =>
  [...document.querySelectorAll("button")].filter((b) => ["确认", "确定", "删除", "取消"].includes(b.textContent?.trim() ?? ""))
    .map((b) => b.textContent?.trim()));
console.log("confirm buttons:", dlgBtns);
if (dlgBtns.length) {
  await page.evaluate(() => {
    [...document.querySelectorAll("button")].find((b) => ["确认", "确定", "删除"].includes(b.textContent?.trim()))?.click();
  });
  await page.waitForTimeout(800);
}
const gone = await page.evaluate(() => !document.body.textContent?.includes("bad-62455a"));
console.log("bad-62455a deleted:", gone);
