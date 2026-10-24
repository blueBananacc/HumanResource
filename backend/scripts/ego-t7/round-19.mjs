// Round 19: 展开 IP 白名单 + 改宽限 → 保存 → API 回读
const task = await taskSpace(42);
const page = task.page("p1");
await page.click("ref=384"); // 打卡 IP 白名单
await page.waitForTimeout(500);
console.log(await page.snapshot({ scope: "subtree", root: "@357" }).catch(() => "subtree failed"));
