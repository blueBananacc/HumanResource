// Round 7: 选推荐 → 填总体评语 → 提交 → 验证卡内反馈
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/talents/9f282fd4-1b2c-4dc2-938d-5de3d32871a8");
await page.waitForLoadState();
await page.waitForTimeout(1200);
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.textContent.includes("填反馈"))?.click();
});
await page.waitForTimeout(700);

// 评分
await page.click('loc=css:button[aria-label*="系统设计能力 4"]');
await page.click('loc=css:button[aria-label*="沟通协作 4"]');
// 总评 4
await page.evaluate(() => {
  const submit = [...document.querySelectorAll("button")].find((b) => b.textContent.trim() === "提交反馈");
  const scope = submit?.closest("div")?.parentElement ?? document;
  [...scope.querySelectorAll("button")]
    .filter((b) => /^[1-5]$/.test(b.textContent.trim()) && !b.getAttribute("aria-label"))
    .find((b) => b.textContent.trim() === "4")?.click();
});
// verdict 推荐
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.textContent.trim() === "待定" && x.closest("div")?.textContent.includes("录用表态"))?.click()
    ?? [...document.querySelectorAll("button")].filter((x) => x.textContent.includes("待定")).pop()?.click();
});
await page.waitForTimeout(400);
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.textContent.includes("推荐进入"))?.click();
});
// 总体评语
await page.evaluate(() => {
  const ta = document.querySelector("textarea[placeholder*='整体印象']");
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  setter.call(ta, "整体表现扎实,系统设计维度突出,建议进入 offer 评估。");
  ta.dispatchEvent(new Event("input", { bubbles: true }));
});
// 提交
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.textContent.trim() === "提交反馈")?.click();
});
await page.waitForTimeout(1800);

const snap = await page.snapshot({ scope: "full_page" });
const idx = snap.indexOf("已提交");
const c = snap.indexOf("均分");
console.log("submitted:", idx >= 0, "| avg:", c >= 0 ? snap.slice(c - 60, c + 80) : "n/a");
console.log(snap.slice(Math.max(0, snap.indexOf("T9验证") - 100), snap.indexOf("T9验证") + 1200));
