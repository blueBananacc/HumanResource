// Round 4: 新建规则抽屉 — 表单结构快照
const task = await taskSpace(42);
const page = task.page("p1");
await page.click("loc=role:button[name='新建规则']");
await page.waitForTimeout(600);
console.log(await page.snapshot({ scope: "full_page" }));
