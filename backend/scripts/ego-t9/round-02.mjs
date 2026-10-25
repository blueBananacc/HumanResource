// Round 2: 新建评分卡表单 → 填写 → 提交
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/talents/9f282fd4-1b2c-4dc2-938d-5de3d32871a8");
await page.waitForLoadState();
await page.waitForTimeout(1200);

// DOM 点击「新建评分卡」(空态里的按钮)
const clicked = await page.evaluate(() => {
  const btns = [...document.querySelectorAll("button")];
  const b = btns.find((x) => x.textContent.includes("新建评分卡"));
  if (!b) return false;
  b.click();
  return true;
});
console.log("clicked:", clicked);
await page.waitForTimeout(800);
const snap = await page.snapshot({ scope: "full_page" });
const idx = snap.lastIndexOf("新建评分卡");
console.log("modal idx:", idx);
console.log(snap.slice(Math.max(0, idx - 200), idx + 3000));
