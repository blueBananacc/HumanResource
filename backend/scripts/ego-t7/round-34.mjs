// Round 34: bad-62455a 是否已删 + 删除按钮真实结构探查
const task = await taskSpace(42);
const page = task.page("p1");
const probe = await page.evaluate(() => {
  // 找到包含 bad-62455a 的规则卡片,列出其全部按钮 title/text
  const cards = [...document.querySelectorAll("[class]")].filter(
    (el) => el.childElementCount >= 1 && el.textContent?.includes("bad-62455a"));
  const card = cards.sort((a, b) => a.textContent.length - b.textContent.length)[0];
  if (!card) return { found: false };
  return {
    found: true,
    buttons: [...card.querySelectorAll("button")].map((b) => ({
      title: b.title, text: b.textContent?.trim().slice(0, 12), aria: b.getAttribute("aria-label"),
    })),
  };
});
console.log(JSON.stringify(probe, null, 1));
