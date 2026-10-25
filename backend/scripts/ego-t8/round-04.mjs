// T8 UI Round 4: 手改"手机"列映射(Select 下拉)→ 下一步校验页
const task = await taskSpace(43);
const page = task.page("p1");

// 找"手机"行里的 (不映射) 按钮 → 打开下拉 → 选"电子邮箱"
const ref = await page.evaluate(() => {
  const rows = [...document.querySelectorAll("tr")];
  const row = rows.find((r) => r.textContent.trim().startsWith("手机"));
  if (!row) return null;
  const btn = row.querySelector("button");
  // 给按钮打临时标记便于 loc 定位
  btn.setAttribute("data-ego-target", "phone-select");
  return !!btn;
});
console.log("phone row found:", ref);
if (ref) {
  await page.click("css=[data-ego-target=phone-select]", { label: "打开手机列映射下拉" });
  await page.waitForTimeout(600);
  console.log(await page.snapshot());
}
