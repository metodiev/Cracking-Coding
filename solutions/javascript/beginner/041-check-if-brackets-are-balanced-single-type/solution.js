/** Task 041 - Check if brackets are balanced (single type). (Tier: Beginner, CTCI Ch 3) */
function isBalanced(s) {
  let depth = 0;
  for (const ch of s) {
    if (ch === "(") depth++;
    else if (ch === ")") { depth--; if (depth < 0) return false; }
  }
  return depth === 0;
}

module.exports = { isBalanced };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isBalanced("((()))"), true);
  assert.strictEqual(isBalanced("()()()"), true);
  assert.strictEqual(isBalanced("(()"), false);
  assert.strictEqual(isBalanced(")("), false);
  console.log("PASS");
}
