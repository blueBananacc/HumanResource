// 检查补卡申请列表 + 我的请假页(加班 modal + 调休卡)
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");

await page.goto("http://localhost/me/attendance");
await page.waitForSelector("css=main h2", { timeout: 10_000 });
const token = await page.evaluate(() =>
  localStorage.getItem("free-hire-token"));
const reqs = await page.fetch("/api/me/attendance-requests", {
  headers: { authorization: `Bearer ${token}` },
});
console.log("att-requests:", JSON.parse(reqs.body).map((r) => ({
  ts: r.ts, kind: r.kind, status: r.status,
})));

await page.goto("http://localhost/me/leave");
await page.waitForSelector("css=main h2", { timeout: 10_000 });
console.log("=== 我的请假 ===");
console.log(await page.snapshot());
