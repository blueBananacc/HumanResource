// Round 7: 选创建 + 收件人改操作者本人 + 标题模板 + 保存
const task = await taskSpace(42);
const page = task.page("p1");
await page.click("ref=206"); // 创建
await page.waitForTimeout(300);
// 收件人 kind 下拉
await page.click("ref=183");
await page.waitForTimeout(300);
console.log(await page.snapshot({ scope: "subtree", root: "@176" }));
