// Round 29: admin 重登 → 编辑抽屉 → 删除 bad 规则
const task = await taskSpace(42);
const page = task.page("p1");
await page.evaluate(() => localStorage.clear());
await page.goto("http://localhost/login");
await page.waitForLoadState();
await page.fill("loc=css:input[type=email] >> nth=0", "admin@example.com");
await page.fill("loc=css:input[type=password]", "admin123456");
await page.click("loc=css:button[type=submit] >> nth=0");
await page.waitForTimeout(2000);
await page.goto("http://localhost/settings/automations");
await page.waitForLoadState();
await page.waitForTimeout(1500);
// 点第一条规则的 编辑
await page.click("loc=role:button[name='编辑'] >> nth=0");
await page.waitForTimeout(800);
console.log("=== 编辑抽屉 ===");
console.log(await page.snapshot({ scope: "subtree", root: "@176" }).catch(() => page.snapshot()));
