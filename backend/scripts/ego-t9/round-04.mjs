// Round 4: 打开「填反馈」表单 → 看结构
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/talents/9f282fd4-1b2c-4dc2-938d-5de3d32871a8");
await page.waitForLoadState();
await page.waitForTimeout(1200);

await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((x) => x.textContent.includes("填反馈"))?.click();
});
await page.waitForTimeout(800);
const snap = await page.snapshot({ scope: "full_page" });
const idx = snap.lastIndexOf("反");
console.log(snap.slice(Math.max(0, idx - 400), idx + 2800));
