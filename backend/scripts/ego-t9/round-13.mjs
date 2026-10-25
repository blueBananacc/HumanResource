// Round 13: facet chip 精确验证 —— 过滤后记录项只剩岗位
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/");
await page.waitForLoadState();
await page.waitForTimeout(1000);
await page.press("loc=css:body", "Control+k");
await page.waitForTimeout(500);
await page.fill("loc=css:input[placeholder*='全局搜索']", "链路");
await page.waitForTimeout(1500);

// 点「岗位 N」chip
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((b) => /^岗位\s*\d/.test(b.textContent.trim()))?.click();
});
await page.waitForTimeout(1200);

// 判定:kb 的 quickstart 记录是否消失
const state = await page.evaluate(() => {
  const txt = document.body.innerText;
  return {
    hasQuickstart: txt.includes("quickstart"),
    hasJobRecord: txt.includes("链路岗") || /链路/.test(txt),
    totalText: (txt.match(/全部\s*\d+\s*条/) || [null])[0],
  };
});
console.log("filtered:", state);
