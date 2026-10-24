// Round 8: 选操作者本人 + 标题模板 + 保存
const task = await taskSpace(42);
const page = task.page("p1");
await page.click("ref=209"); // 操作者本人
await page.waitForTimeout(300);
await page.fill("loc=css:input[name=title_tpl]", "UI规则触发 {{title}}");
await page.click("loc=role:button[name='保存规则']");
await page.waitForTimeout(1200);
console.log("url:", await page.url());
console.log(await page.snapshot());
