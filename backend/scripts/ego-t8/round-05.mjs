// T8 UI Round 5: 入职日期 解绑→重绑(人工标记)→ 下一步校验页
const task = await taskSpace(43);
const page = task.page("p1");

// 标记入职日期行的 Select 按钮
const ok = await page.evaluate(() => {
  const row = [...document.querySelectorAll("tr")].find((r) =>
    r.textContent.trim().startsWith("入职日期"),
  );
  if (!row) return false;
  row.querySelector("button").setAttribute("data-ego-target", "hire-select");
  return true;
});
console.log("hire row:", ok);
// 打开下拉 → 选"(不映射)"
await page.click("css=[data-ego-target=hire-select]", { label: "打开入职日期下拉" });
await page.waitForTimeout(500);
await page.click("text=(不映射)", { label: "选不映射" });
await page.waitForTimeout(500);
// 重新打开 → 选回"入职日期"
await page.click("css=[data-ego-target=hire-select]", { label: "重开下拉" });
await page.waitForTimeout(500);
const opts = await page.evaluate(() =>
  [...document.querySelectorAll("[role=option], [role=menuitem], button")]
    .map((b) => b.textContent.trim())
    .filter((t) => t.includes("入职日期") || t.includes("不映射")),
);
console.log("options:", opts);
await page.click("text=入职日期", { label: "选回入职日期列" });
await page.waitForTimeout(500);
// 徽标应变"人工"
const badges = await page.evaluate(() =>
  [...document.querySelectorAll("td span")]
    .map((s) => s.textContent.trim())
    .filter(Boolean),
);
console.log("badges now:", badges.filter((b) => /AI|启发|人工/.test(b)));

// 下一步校验
await page.click("text=下一步", { label: "下一步校验" });
await page.waitForTimeout(3000);
console.log(await page.snapshot());
