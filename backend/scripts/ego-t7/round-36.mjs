// Round 36: 执行记录抽屉(T7-UI验证规则 有 run)
const task = await taskSpace(42);
const page = task.page("p1");
await page.evaluate(() => {
  const cards = [...document.querySelectorAll("[class]")].filter(
    (el) => el.childElementCount >= 1 && el.textContent?.includes("T7-UI验证规则"));
  const card = cards.sort((a, b) => a.textContent.length - b.textContent.length)[0];
  [...card.querySelectorAll("button")].find((b) => b.title === "执行记录")?.click();
});
await page.waitForTimeout(900);
console.log(await page.snapshot());
