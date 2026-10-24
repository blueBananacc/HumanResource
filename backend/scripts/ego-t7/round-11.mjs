// Round 11: 执行记录抽屉
const task = await taskSpace(42);
const page = task.page("p1");
await page.goto("http://localhost/settings/automations");
await page.waitForLoadState();
await page.waitForTimeout(1500);
// 第一条规则卡的历史按钮(svg icon 按钮,位置在 编辑 之后)
const histBtn = await page.evaluate(() => {
  const cards = [...document.querySelectorAll("button")];
  const first = cards.find((b) => b.textContent?.includes("T7-UI验证规则"));
  return first ? "found" : "missing";
});
console.log("rule card:", histBtn);
// 用 title 属性定位 History 按钮(lucide History 图标,title=执行记录)
await page.click("loc=css:button[title*='执行记录'] >> nth=0");
await page.waitForTimeout(800);
console.log(await page.snapshot());
