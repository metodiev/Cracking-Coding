/** Task 044 - Trim leading and trailing spaces from string. (Tier: Beginner, CTCI Ch 1) */
function trim(s) {
  let lo = 0, hi = s.length;
  while (lo < hi && s[lo] === " ") lo++;
  while (hi > lo && s[hi - 1] === " ") hi--;
  return s.slice(lo, hi);
}

module.exports = { trim };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(trim("  hello  "), "hello");
  assert.strictEqual(trim("   "), "");
  assert.strictEqual(trim("  a b "), "a b");
  console.log("PASS");
}
