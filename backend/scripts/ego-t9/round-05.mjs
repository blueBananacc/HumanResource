// Round 5: 填反馈 — 逐项评分 + verdict + 评语 + 提交
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/talents/9f282fd4-1b2c-4dc2-938d-5de3d32871a8");
await page.waitForLoadState();
await page.waitForTimeout(1200);
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.textContent.includes("填反馈"))?.click();
});
await page.waitForTimeout(800);

// 模态头部结构(评分控件)
const snap = await page.snapshot({ scope: "full_page" });
const idx = snap.indexOf("提交反馈");
// 从「填反馈」标题往前找
const head = snap.indexOf("总评");
console.log(snap.slice(Math.max(0, snap.lastIndexOf("维度评分", idx)), idx + 200));
