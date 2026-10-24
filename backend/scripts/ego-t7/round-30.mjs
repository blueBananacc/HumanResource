// Round 30: 编辑抽屉(text 选择器)
const task = await taskSpace(42);
const page = task.page("p1");
await page.click('text="编辑" >> nth=0');
await page.waitForTimeout(800);
console.log("=== 编辑抽屉 ===");
console.log(await page.snapshot({ scope: "subtree", root: "@176" }).catch(() => page.snapshot()));
