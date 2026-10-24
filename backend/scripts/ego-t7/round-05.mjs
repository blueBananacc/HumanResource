// Round 5: UI 建规则 — 填名/选实体/选触发动作/改收件人/写模板/保存
const task = await taskSpace(42);
const page = task.page("p1");

await page.fill("loc=css:input[name=name]", "T7-UI验证规则");
// 实体 Select -> 岗位
await page.click("ref=179");
await page.waitForTimeout(300);
console.log("=== 实体选项 ===");
console.log(await page.snapshot({ scope: "subtree", root: "@176" }).catch(() => page.snapshot()));
