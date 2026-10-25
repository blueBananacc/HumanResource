// T8 UI 验证 Round 1: 登录 + 导入向导页 + 选员工数据集 + 上传 → 映射页
const task = await taskSpace("t8-ui-verify");
const page = task.page("p1");
console.log({ spaceId: task.spaceId });

await page.goto("http://localhost/login");
await page.waitForLoadState();
const url = await page.url();
if (url.includes("/login")) {
  await page.fill("loc=css:input[type=email] >> nth=0", "admin@example.com");
  await page.fill("loc=css:input[type=password]", "admin123456");
  await page.click("loc=css:button[type=submit]");
  await page.waitForURL(/^(?!.*login).*$/, { timeout: 15000 });
}
await page.goto("http://localhost/settings/imports");
await page.waitForLoadState();
await page.waitForTimeout(1500);
console.log("imports url:", await page.url());
console.log(await page.snapshot());
