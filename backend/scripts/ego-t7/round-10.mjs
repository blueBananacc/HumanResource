// Round 10: 页面内 fetch 建岗位触发规则
const task = await taskSpace(42);
const page = task.page("p1");
const res = await page.evaluate(() => {
  const token = localStorage.getItem("free-hire-token");
  return fetch("/api/jobs/", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ title: "UI链路岗位-自动化验证", level: "intermediate" }),
  }).then(async (r) => ({ status: r.status, body: await r.text() }));
});
console.log("job create:", res);
await page.waitForTimeout(1000);
// 看首页通知卡片是否有新通知
await page.goto("http://localhost/");
await page.waitForLoadState();
await page.waitForTimeout(1500);
console.log(await page.snapshot());
