// T8 UI Round 7: 重置回上传页(历史含新任务)→ 员工列表验证落库
const task = await taskSpace(43);
const page = task.page("p1");

await page.click("text=再导一个文件", { label: "再导一个" });
await page.waitForTimeout(1200);
// 历史区应含 staff-ui.csv
const history = await page.evaluate(() =>
  [...document.querySelectorAll("span")]
    .map((s) => s.textContent.trim())
    .filter((t) => t.includes("staff-ui.csv") || t.includes("新建")),
);
console.log("history entries:", history);
console.log(await page.snapshot());
