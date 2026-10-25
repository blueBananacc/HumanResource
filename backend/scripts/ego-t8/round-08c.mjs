// T8 UI Round 8c: 诊断员工页输入框
const task = await taskSpace(43);
const page = task.page("p1");
console.log("url:", await page.url());
const inputs = await page.evaluate(() =>
  [...document.querySelectorAll("input")].map((i) => ({
    ph: i.placeholder, type: i.type, name: i.name,
  })),
);
console.log("inputs:", JSON.stringify(inputs));
const texts = await page.evaluate(() =>
  document.body.textContent.slice(0, 400));
console.log("body:", texts);
