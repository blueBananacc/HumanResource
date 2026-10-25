// T8 UI Round 2: 选员工数据集 → 上传 CSV → 列映射页
const task = await taskSpace(43);
const page = task.page("p1");

// 选员工数据集(卡片 button,用唯一副标题文本定位)
await page.click("text=姓名/工号/部门/入职信息", { label: "选员工数据集" });
await page.waitForTimeout(800);

// 上传文件
await page.setInputFiles(
  "css=input[type=file]",
  ["C:/Users/666/Desktop/HROnfire/backend/scripts/ego-t8/staff-ui.csv"],
);
await page.waitForTimeout(2000);
console.log("after upload url:", await page.url());
console.log(await page.snapshot());
