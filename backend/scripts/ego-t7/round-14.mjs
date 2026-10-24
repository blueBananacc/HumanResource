// Round 14: 启停开关(button[title=点击停用/启用])
const task = await taskSpace(42);
const page = task.page("p1");
const t1 = await page.evaluate(() => document.querySelector("button[title^=点击]")?.title ?? "none");
console.log("before:", t1);
await page.click("loc=css:button[title^=点击] >> nth=0");
await page.waitForTimeout(900);
const t2 = await page.evaluate(() => document.querySelector("button[title^=点击]")?.title ?? "none");
console.log("after off:", t2);
// 恢复开启
await page.click("loc=css:button[title^=点击] >> nth=0");
await page.waitForTimeout(900);
const t3 = await page.evaluate(() => document.querySelector("button[title^=点击]")?.title ?? "none");
console.log("restored:", t3);
