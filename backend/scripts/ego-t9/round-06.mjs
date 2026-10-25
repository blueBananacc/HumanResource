// Round 6: 填反馈并提交
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/talents/9f282fd4-1b2c-4dc2-938d-5de3d32871a8");
await page.waitForLoadState();
await page.waitForTimeout(1200);
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.textContent.includes("填反馈"))?.click();
});
await page.waitForTimeout(800);

// 逐项评分:系统设计 4 分,沟通协作 4 分
await page.click('loc=css:button[aria-label*="系统设计能力 4"]');
await page.click('loc=css:button[aria-label*="沟通协作 4"]');
// 维度评语
await page.evaluate(() => {
  const areas = [...document.querySelectorAll("input[placeholder*='评语/证据']")];
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
  areas.forEach((a, i) => {
    setter.call(a, i === 0 ? "架构思路清晰,缓存方案合理" : "表达顺畅,主动确认需求");
    a.dispatchEvent(new Event("input", { bubbles: true }));
  });
});
// 总评 4 分(模态内最后一组 1-5)
await page.evaluate(() => {
  const modal = [...document.querySelectorAll("button")].find((b) => b.textContent.trim() === "提交反馈").closest("div");
  const scope = modal?.parentElement ?? document;
  const nums = [...scope.querySelectorAll("button")].filter((b) => /^[1-5]$/.test(b.textContent.trim()) && !b.getAttribute("aria-label"));
  nums.find((b) => b.textContent.trim() === "4")?.click();
});
// 录用表态 → 推荐
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.textContent.includes("待定"))?.click();
});
await page.waitForTimeout(400);
const optSnap = await page.snapshot({ scope: "full_page" });
const oi = optSnap.indexOf("推荐");
console.log("verdict options idx:", oi, optSnap.slice(Math.max(0, oi - 300), oi + 400));
