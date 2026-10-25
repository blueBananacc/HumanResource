// T8 UI Round 5b: 入职日期 解绑→重绑(人工标记)→ 下一步校验页
const task = await taskSpace(43);
const page = task.page("p1");

async function markRowBtn(label) {
  return page.evaluate((lbl) => {
    const row = [...document.querySelectorAll("tr")].find((r) =>
      r.textContent.trim().startsWith(lbl),
    );
    if (!row) return false;
    row.querySelector("button").setAttribute("data-ego-target", "sel-btn");
    return true;
  }, label);
}
async function markOption(text) {
  return page.evaluate((t) => {
    const panel = document.querySelector("div.absolute.z-40");
    if (!panel) return "no-panel";
    const btn = [...panel.querySelectorAll("button")].find(
      (b) => b.textContent.trim() === t,
    );
    if (!btn) return "no-option";
    btn.setAttribute("data-ego-option", "pick");
    return "ok";
  }, text);
}
async function clickOpt(text) {
  const r = await markOption(text);
  console.log("mark option", text, "→", r);
  if (r !== "ok") return false;
  await page.click("css=[data-ego-option=pick]", { label: `选项 ${text}` });
  await page.waitForTimeout(500);
  await page.evaluate(() =>
    document.querySelector("[data-ego-option]")?.removeAttribute("data-ego-option"),
  );
  return true;
}

console.log("mark row:", await markRowBtn("入职日期"));
await page.click("css=[data-ego-target=sel-btn]", { label: "打开入职日期下拉" });
await page.waitForTimeout(500);
await clickOpt("(不映射)");
await page.click("css=[data-ego-target=sel-btn]", { label: "重开下拉" });
await page.waitForTimeout(500);
await clickOpt("入职日期");

const badges = await page.evaluate(() =>
  [...document.querySelectorAll("td span")]
    .map((s) => s.textContent.trim())
    .filter((t) => /AI|启发|人工/.test(t)),
);
console.log("badges now:", badges);

await page.click("text=下一步", { label: "下一步校验" });
await page.waitForTimeout(3000);
console.log(await page.snapshot());
