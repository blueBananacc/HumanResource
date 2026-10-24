// Round 33: 关编辑抽屉 → 删除 bad-62455a(confirm 处理)→ 验证消失
const task = await taskSpace(42);
const page = task.page("p1");
await page.evaluate(() => {
  const b = [...document.querySelectorAll("button")].find((x) => x.textContent?.trim() === "关闭");
  b?.click();
});
await page.waitForTimeout(500);
// bad-62455a 行的删除按钮(行内第三个按钮)
await page.evaluate(() => {
  const cards = [...document.querySelectorAll("div")].filter((d) =>
    d.firstElementChild?.tagName === "DIV" && d.textContent?.startsWith("bad-62455a"));
  const row = cards.find((d) => d.querySelector("button[title]"));
  // 直接找含 bad-62455a 文本的行内垃圾桶按钮
  const all = [...document.querySelectorAll("button")];
  const target = all.find((b) => {
    const rowEl = b.closest("div");
    return rowEl?.textContent?.includes("bad-62455a") && !b.textContent?.trim();
  });
  target?.click();
  return !!target;
});
await page.waitForTimeout(600);
// 处理 confirm 对话框
const dlg = await page.evaluate(() => {
  const btns = [...document.querySelectorAll("button")].map((b) => b.textContent?.trim());
  return btns;
});
console.log("buttons after delete click:", dlg.slice(-15));
