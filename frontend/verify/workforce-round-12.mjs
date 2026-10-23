// 验证周凯已绑定 + 台账 policy_run;再开年度发放 modal 验证
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");
await page.goto("http://localhost/leave/policies");
await page.waitForLoadState();

const token = await page.evaluate(() =>
  localStorage.getItem("free-hire-token"));
const hdrs = { authorization: `Bearer ${token}` };

const pols = await page.fetch("/api/leave/policies", { headers: hdrs });
console.log(JSON.parse(pols.body).map((p) => ({
  name: p.name, assigned: p.assigned_count, carry: p.carry_forward,
})));

const led = await page.fetch("/api/leave/ledger?limit=10", { headers: hdrs });
console.log(JSON.parse(led.body).items.filter((l) => l.ref_type === "policy_run").map((l) => ({
  emp: l.employee_name, type: l.leave_type_name, delta: l.delta_days,
})));

// 年度发放 modal
await page.click("loc=role:button[name=年度发放]");
await page.waitForSelector("text=发放年份", { timeout: 8_000 });
console.log("=== 年度发放 modal ===");
console.log(await page.snapshot());
