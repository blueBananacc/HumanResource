// Round 12: facet chip 过滤 → 点「岗位」chip → 只剩岗位组 → 取消 → 点结果跳转
const task = await taskSpace("t9-ui-verify");
const page = task.page("p1");

await page.goto("http://localhost/");
await page.waitForLoadState();
await page.waitForTimeout(1000);
await page.press("loc=css:body", "Control+k");
await page.waitForTimeout(500);
await page.fill("loc=css:input[placeholder*='全局搜索']", "链路");
await page.waitForTimeout(1500);

// 点「岗位」chip
await page.evaluate(() => {
  const chips = [...document.querySelectorAll("button")].filter((b) =>
    /^岗位\s*\d/.test(b.textContent.trim()),
  );
  chips[0]?.click();
});
await page.waitForTimeout(1200);
let snap = await page.snapshot({ scope: "full_page" });
let has = { jobGroup: snap.indexOf("岗位") >= 0, kbGroup: snap.indexOf("知识文章") >= 0 };
console.log("after 岗位 chip:", has);

// 点「全部」chip 取消过滤
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((b) => b.textContent.trim() === "全部")?.click();
});
await page.waitForTimeout(1200);
snap = await page.snapshot({ scope: "full_page" });
console.log("after 全部 chip: kbGroup back =", snap.indexOf("知识文章") >= 0);

// 点击第一条岗位结果 → 跳转
await page.evaluate(() => {
  [...document.querySelectorAll("button")].find((b) => b.textContent.includes("链路"))?.click();
});
await page.waitForTimeout(1500);
console.log("navigated to:", await page.url());
