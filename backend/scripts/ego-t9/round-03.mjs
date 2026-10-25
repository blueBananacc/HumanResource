// Round 3: 填评分卡表单 → 提交 → 卡片出现
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/talents/9f282fd4-1b2c-4dc2-938d-5de3d32871a8");
await page.waitForLoadState();
await page.waitForTimeout(1200);

// 打开模态
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.textContent.includes("新建评分卡"))?.click();
});
await page.waitForTimeout(700);

// 填标题 + 地点
await page.fill("loc=css:input[placeholder*='二面']", "T9验证 · 终面评分卡");
await page.fill("loc=css:input[placeholder*='线上']", "线上 · 腾讯会议");

// 填第一个维度名
await page.fill("loc=css:input[placeholder*='维度名']", "系统设计能力");
// 加一个维度
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.textContent.includes("加维度"))?.click();
});
await page.waitForTimeout(400);
// 填第二个维度名(最后一个维度输入)
await page.evaluate(() => {
  const inputs = [...document.querySelectorAll("input[placeholder*='维度名']")];
  const last = inputs[inputs.length - 1];
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
  setter.call(last, "沟通协作");
  last.dispatchEvent(new Event("input", { bubbles: true }));
});

// 勾 admin 面试官
const checked = await page.evaluate(() => {
  const labels = [...document.querySelectorAll("button")].filter((x) =>
    x.getAttribute("aria-label") === "admin@example.com"
  );
  if (!labels.length) return "not-found";
  labels[0].click();
  return "ok";
});
console.log("admin checkbox:", checked);
await page.waitForTimeout(300);

// 提交(找模态里的主按钮 创建/保存)
const sub = await page.evaluate(() => {
  const btns = [...document.querySelectorAll("button")];
  const b = btns.find((x) => /^(创建|保存|确定|提交)/.test(x.textContent.trim()) && x.offsetParent);
  b?.click();
  return b?.textContent.trim();
});
console.log("submit btn:", sub);
await page.waitForTimeout(1500);

const snap = await page.snapshot({ scope: "full_page" });
const idx = snap.indexOf("T9验证");
console.log("card created:", idx >= 0);
console.log(snap.slice(Math.max(0, idx - 200), idx + 1800));
