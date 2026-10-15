/** Task 016 - Count occurrences of each character in a string. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function charCounts(s) {
  const counts = new Map();
  for (const ch of s) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  return Object.fromEntries(counts);
}

module.exports = { charCounts };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(charCounts("aabbbc"), { a: 2, b: 3, c: 1 });
  assert.deepStrictEqual(charCounts(""), {});
  console.log("PASS");
}
