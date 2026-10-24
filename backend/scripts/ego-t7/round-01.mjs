// T7 UI 验证 Round 1: 登录 + 进自动化规则页
const task = await taskSpace("t7-ui-verify");
const page = task.page("p1");
console.log({ spaceId: task.spaceId });

await page.goto("http://localhost/login");
await page.waitForLoadState();
// 已登录则直接跳首页;否则填表登录
const url = await page.url();
console.log("landing:", url);
if (url.includes("/login")) {
  await page.fill("loc=css:input[type=email], input[name=email] >> nth=0", "admin@example.com");
  await page.fill("loc=css:input[type=password]", "admin123456");
  await page.click("loc=role:button[name*=登录], button[type=submit] >> nth=0");
  await page.waitForURL(/^(?!.*login).*$/, { timeout: 15000 });
}
await page.goto("http://localhost/automations");
await page.waitForLoadState();
await page.waitForTimeout(1200);
console.log("automations url:", await page.url());
console.log(await page.snapshot());
