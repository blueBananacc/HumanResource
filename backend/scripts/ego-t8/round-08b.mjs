// T8 UI Round 8b: 员工列表验证 → 删除导入记录(确认框)
const task = await taskSpace(43);
const page = task.page("p1");

await page.goto("http://localhost/hris/employees");
await page.waitForLoadState();
await page.waitForTimeout(1500);
await page.evaluate(() => {
  const inp = [...document.querySelectorAll("input")].find((i) =>
    /搜姓名/.test(i.placeholder || ""),
  );
  inp?.setAttribute("data-ego-target", "emp-search");
});
await page.fill("css=[data-ego-target=emp-search]", "UI验证");
await page.waitForTimeout(2500); // SearchInput 防抖 + 请求
let rows = await page.evaluate(() =>
  [...document.querySelectorAll("tr")]
    .map((r) => r.textContent.trim().slice(0, 120))
    .filter((t) => t.includes("UI验证")),
);
console.log("employee rows:", rows);

// 回导入页:从历史行爬父级找删除按钮
await page.goto("http://localhost/settings/imports");
await page.waitForLoadState();
await page.waitForTimeout(1200);
const marked = await page.evaluate(() => {
  const span = [...document.querySelectorAll("span")].find(
    (s) => s.textContent.trim() === "staff-ui.csv",
  );
  if (!span) return "no-span";
  let el = span.parentElement;
  for (let i = 0; i < 6 && el; i++) {
    if (el.querySelector("button")) {
      el.querySelector("button").setAttribute("data-ego-del", "x");
      return "ok";
    }
    el = el.parentElement;
  }
  return "no-btn";
});
console.log("del btn:", marked);
if (marked === "ok") {
  await page.click("css=[data-ego-del]", { label: "删除导入记录" });
  await page.waitForTimeout(800);
  console.log(await page.snapshot());
}
