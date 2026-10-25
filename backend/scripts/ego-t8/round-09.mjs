// T8 UI Round 9: 员工卡片验证落库(div,非 tr)
const task = await taskSpace(43);
const page = task.page("p1");

await page.goto("http://localhost/employees");
await page.waitForLoadState();
await page.waitForTimeout(2000);
await page.evaluate(() => {
  const inp = [...document.querySelectorAll("input")].find((i) =>
    /搜姓名/.test(i.placeholder || ""),
  );
  inp?.setAttribute("data-ego-target", "emp-search");
});
await page.fill("css=[data-ego-target=emp-search]", "UI验证");
await page.waitForTimeout(2500);
const cards = await page.evaluate(() =>
  [...document.querySelectorAll("a")]
    .map((a) => a.textContent.trim().replace(/\s+/g, " ").slice(0, 160))
    .filter((t) => t.includes("UI验证")),
);
console.log("cards:", JSON.stringify(cards, null, 1));
const total = await page.evaluate(() => document.body.textContent.match(/共\s*\d+\s*人/)?.[0]);
console.log("total label:", total);
