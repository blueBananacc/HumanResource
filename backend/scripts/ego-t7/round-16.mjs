// Round 16: 行内免计 → 验证徽标变化
const task = await taskSpace(42);
const page = task.page("p1");
await page.fill("ref=367", "UI验证-客户拜访");
await page.click("ref=368"); // 免计按钮
await page.waitForTimeout(1200);
console.log(await page.snapshot());
