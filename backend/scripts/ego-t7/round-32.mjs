// Round 32: evaluate 点编辑 → 抽屉快照
const task = await taskSpace(42);
const page = task.page("p1");
await page.evaluate(() => {
  const btn = [...document.querySelectorAll("button")].find((b) => b.textContent?.trim() === "编辑");
  btn?.click();
});
await page.waitForTimeout(800);
console.log(await page.snapshot({ scope: "subtree", root: "@176" }).catch(() => page.snapshot()));
