// T8 UI Round 8d: /employees 验证落库 → 导入页删除记录(确认框)
const task = await taskSpace(43);
const page = task.page("p1");

await page.goto("http://localhost/employees");
await page.waitForLoadState();
await page.waitForTimeout(2000);
console.log("url:", await page.url());
await page.evaluate(() => {
  const inp = [...document.querySelectorAll("input")].find((i) =>
    /搜姓名/.test(i.placeholder || ""),
  );
  inp?.setAttribute("data-ego-target", "emp-search");
});
await page.fill("css=[data-ego-target=emp-search]", "UI验证");
await page.waitForTimeout(2500);
const rows = await page.evaluate(() =>
  [...document.querySelectorAll("tr")]
    .map((r) => r.textContent.trim().slice(0, 140))
    .filter((t) => t.includes("UI验证")),
);
console.log("employee rows:", JSON.stringify(rows, null, 1));

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
  // 确认对话框:点"删除"
  const dlg = await page.evaluate(() => {
    const btns = [...document.querySelectorAll("button")].map((b) =>
      b.textContent.trim(),
    );
    return btns.filter((t) => /删除|取消/.test(t));
  });
  console.log("dialog buttons:", dlg);
  await page.evaluate(() => {
    const b = [...document.querySelectorAll("button")].find(
      (x) => x.textContent.trim() === "删除",
    );
    b?.setAttribute("data-ego-confirm", "x");
  });
  await page.click("css=[data-ego-confirm]", { label: "确认删除" });
  await page.waitForTimeout(1200);
  const still = await page.evaluate(() =>
    [...document.querySelectorAll("span")].some(
      (s) => s.textContent.trim() === "staff-ui.csv",
    ),
  );
  console.log("record still present:", still);
}
