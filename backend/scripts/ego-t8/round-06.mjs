// T8 UI Round 6: 下载错误 CSV → 执行导入 → 结果页 → 历史记录
const task = await taskSpace(43);
const page = task.page("p1");

// 下载错误行 CSV
const dl = page.waitForEvent("download", { timeout: 15000 });
await page.click("text=下载错误行", { label: "下载错误行" });
const d = await dl;
const saveTo = "C:/Users/666/Desktop/HROnfire/backend/scripts/ego-t8/errors-ui.csv";
await d.saveAs(saveTo);
console.log("downloaded:", d.suggestedFilename());
const fs = await import("node:fs/promises");
console.log("csv content:", (await fs.readFile(saveTo, "utf8")).slice(0, 300));

// 执行导入
await page.click("text=执行导入", { label: "执行导入" });
await page.waitForTimeout(3000);
console.log(await page.snapshot());
