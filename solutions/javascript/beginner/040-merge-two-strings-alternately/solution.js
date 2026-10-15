/** Task 040 - Merge two strings alternately. (Tier: Beginner, CTCI Ch 1) */
function mergeAlternately(a, b) {
  const out = [];
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if (i < a.length) out.push(a[i]);
    if (i < b.length) out.push(b[i]);
  }
  return out.join("");
}

module.exports = { mergeAlternately };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(mergeAlternately("abc", "123"), "a1b2c3");
  assert.strictEqual(mergeAlternately("ab", "1234"), "a1b234");
  assert.strictEqual(mergeAlternately("", "xyz"), "xyz");
  console.log("PASS");
}
