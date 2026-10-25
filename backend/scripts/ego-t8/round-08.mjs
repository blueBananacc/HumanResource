// T8 UI Round 8: 员工列表验证落库 → 回导入页删除历史记录(确认框)
const task = await taskSpace(43);
const page = task.page("p1");

await page.goto("http://localhost/hris/employees");
await page.waitForLoadState();
await page.waitForTimeout(1500);
// 搜索框找 UI验证
const searchSel = await page.evaluate(() => {
  const inp = [...document.querySelectorAll("input")].find(
    (i) => i.placeholder && /搜索|search|姓名/i.test(i.placeholder),
  );
  if (inp) {
    inp.setAttribute("data-ego-target", "emp-search");
    return true;
  }
  return false;
});
console.log("search input:", searchSel);
if (searchSel) {
  await page.fill("css=[data-ego-target=emp-search]", "UI验证");
  await page.keyboard.press("Enter");
  await page.waitForTimeout(1200);
}
const rows = await page.evaluate(() =>
  [...document.querySelectorAll("tr")]
    .map((r) => r.textContent.trim())
    .filter((t) => t.includes("UI验证")),
);
console.log("employee rows:", rows);

// 回导入页删除记录
await page.goto("http://localhost/settings/imports");
await page.waitForLoadState();
await page.waitForTimeout(1200);
const marked = await page.evaluate(() => {
  const row = [...document.querySelectorAll("div")]
    .filter((d) => d.textContent.includes("staff-ui.csv"))
    .pop();
  if (!row) return false;
  const btn = row.querySelector("button");
  if (!btn) return false;
  btn.setAttribute("data-ego-del", "x");
  return true;
});
console.log("del btn:", marked);
if (marked) {
  await page.click("css=[data-ego-del]", { label: "删除导入记录" });
  await page.waitForTimeout(800);
  console.log("confirm dialog:", await page.snapshot());
}
