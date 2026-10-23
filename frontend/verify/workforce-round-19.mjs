// 加班 modal 一次走完:开 → 选类型 → 填 → 提交 → 校验
const task = await taskSpace("hris-workforce-round1");
const page = task.page("p1");
await page.goto("http://localhost/me/leave");
await page.waitForSelector("css=main h2", { timeout: 10_000 });

await page.click("loc=role:button[name*=申请加班]");
await page.waitForSelector("text=主管批准后按日标准工时折算", {
  timeout: 8_000,
});
await page.click('css=button:has-text("选择调休类型")');
await page.waitForTimeout(600);
await page.click('css=button:has-text("调休假")');
await page.fill("css=input[name=work_date]", "2026-10-21");
await page.fill("css=input[name=hours]", "5");
await page.fill("css=input[name=reason]", "联调环境修复");
await page.click('css=button[type=submit] >> nth=-1');
await page.waitForTimeout(1500);

const token = await page.evaluate(() =>
  localStorage.getItem("free-hire-token"));
const ots = await page.fetch("/api/me/overtime-requests", {
  headers: { authorization: `Bearer ${token}` },
});
console.log("ot-requests:", JSON.parse(ots.body).map((r) => ({
  date: r.work_date, hours: r.hours, comp: r.comp_days, status: r.status,
})));
