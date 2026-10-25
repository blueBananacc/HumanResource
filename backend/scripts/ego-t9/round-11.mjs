// Round 11: 多实体分组搜索 + facet chip + 点击跳转
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/");
await page.waitForLoadState();
await page.waitForTimeout(1000);
await page.press("loc=css:body", "Control+k");
await page.waitForTimeout(600);

// "链路" 同时出现在岗位名和候选人名里
await page.fill("loc=css:input[placeholder*='搜索']", "链路");
await page.waitForTimeout(1600);
let snap = await page.snapshot({ scope: "full_page" });
let ri = snap.indexOf("链路");
console.log("=== groups ===");
console.log(snap.slice(Math.max(0, ri - 800), ri + 1600));

// 点 facet chip(岗位)过滤
await page.evaluate(() => {
  const b = [...document.querySelectorAll("button")].find((x) => x.textContent.trim().startsWith("岗位"));
  b?.click();
});
await page.waitForTimeout(1200);
snap = await page.snapshot({ scope: "full_page" });
ri = snap.indexOf("岗位");
console.log("=== after facet 岗位 ===");
console.log(snap.slice(Math.max(0, ri - 400), ri + 1400));
