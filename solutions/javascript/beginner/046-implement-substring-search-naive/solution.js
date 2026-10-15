/** Task 046 - Implement substring search (naive). (Tier: Beginner, CTCI Ch 10) */
function substringSearch(text, pattern) {
  const n = text.length, m = pattern.length;
  if (m === 0) return 0;
  outer:
  for (let i = 0; i + m <= n; i++) {
    for (let j = 0; j < m; j++) if (text[i + j] !== pattern[j]) continue outer;
    return i;
  }
  return -1;
}

module.exports = { substringSearch };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(substringSearch("hello world", "world"), 6);
  assert.strictEqual(substringSearch("abc", "xyz"), -1);
  console.log("PASS");
}
