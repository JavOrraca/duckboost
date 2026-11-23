const copyButton = document.getElementById("copy");
const copyLabel = copyButton.querySelector("span");
const sql = document.getElementById("sql");
let resetTimer;

copyButton.addEventListener("click", async () => {
  clearTimeout(resetTimer);
  try {
    await navigator.clipboard.writeText(sql.textContent);
    copyLabel.textContent = "Copied!";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(sql);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyLabel.textContent = "SQL selected";
  }
  resetTimer = setTimeout(() => { copyLabel.textContent = "Copy SQL"; }, 2400);
});
