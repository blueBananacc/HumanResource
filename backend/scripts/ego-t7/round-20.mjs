// Round 20: 改上班宽限 15→20 → 保存 → API 回读
const task = await taskSpace(42);
const page = task.page("p1");
await page.fill("ref=380", "20");
await page.click("loc=role:button[name*='保存规则']");
await page.waitForTimeout(1200);
// API 回读 policy
const policy = await page.evaluate(() => {
  const token = localStorage.getItem("free-hire-token");
  return fetch("/api/attendance/policy", { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json());
});
console.log("policy after UI save:", policy);
