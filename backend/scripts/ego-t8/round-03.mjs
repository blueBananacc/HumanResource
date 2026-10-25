// T8 UI Round 3: AI 智能映射 → 置信度徽标 → 下一步校验页
const task = await taskSpace(43);
const page = task.page("p1");

await page.click("text=AI 智能映射", { label: "点 AI 智能映射" });
await page.waitForTimeout(4000);
// 映射来源徽标应出现 AI x%
const srcBadges = await page.evaluate(() =>
  [...document.querySelectorAll("td span")]
    .map((s) => s.textContent.trim())
    .filter((t) => /AI|启发式|人工/.test(t)),
);
console.log("source badges:", srcBadges);
console.log(await page.snapshot());
