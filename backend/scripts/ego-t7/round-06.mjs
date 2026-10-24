// Round 6: 选岗位 + 开触发动作下拉
const task = await taskSpace(42);
const page = task.page("p1");
await page.click("ref=193"); // 岗位
await page.waitForTimeout(300);
await page.click("ref=180"); // 触发动作下拉
await page.waitForTimeout(300);
console.log(await page.snapshot({ scope: "subtree", root: "@176" }));
