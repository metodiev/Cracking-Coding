/** Task 047 - Find first non-repeating character in a string. (Tier: Beginner, CTCI Ch 1) */
function firstNonRepeating(s) {
  const counts = new Map();
  for (const ch of s) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  for (const ch of s) if (counts.get(ch) === 1) return ch;
  return null;
}

module.exports = { firstNonRepeating };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(firstNonRepeating("aabbcde"), "c");
  assert.strictEqual(firstNonRepeating("aabb"), null);
  assert.strictEqual(firstNonRepeating("x"), "x");
  console.log("PASS");
}
