// 数据面验证:加班批准后的调休额度/台账 + 政策页快照
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");
await page.goto("http://localhost/leave/balances");
await page.waitForLoadState();

const token = await page.evaluate(() =>
  localStorage.getItem("free-hire-token"));
const hdrs = { authorization: `Bearer ${token}` };

const bal = await page.fetch("/api/leave/balances?year=2026", {
  headers: hdrs,
});
const rows = JSON.parse(bal.body);
console.log(
  (Array.isArray(rows) ? rows : rows.items ?? []).map((r) => ({
    emp: r.employee_name, type: r.leave_type_name,
    allocated: r.allocated, balance: r.balance,
  })),
);

const led = await page.fetch("/api/leave/ledger?limit=50", {
  headers: hdrs,
});
const ledBody = JSON.parse(led.body);
console.log(
  (ledBody.items ?? ledBody).map((l) => ({
    emp: l.employee_name, type: l.leave_type_name, delta: l.delta_days,
    ref: l.ref_type,
  })),
);

await page.goto("http://localhost/leave/policies");
await page.waitForSelector("css=main h2", { timeout: 10_000 });
console.log("=== 假期政策页 ===");
console.log(await page.snapshot());
