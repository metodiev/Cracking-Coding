/** Task 169 - Sliding window for unique characters (longest unique substring). (Tier: Beginner, CTCI Ch 1) */
"use strict";

function longestUniqueSubstring(s) {
  const last = new Map();
  let bestLen = 0, bestStart = 0, lo = 0;
  for (let hi = 0; hi < s.length; hi++) {
    const ch = s[hi];
    if (last.has(ch) && last.get(ch) >= lo) lo = last.get(ch) + 1;
    last.set(ch, hi);
    if (hi - lo + 1 > bestLen) {
      bestLen = hi - lo + 1;
      bestStart = lo;
    }
  }
  return { length: bestLen, substring: s.slice(bestStart, bestStart + bestLen) };
}

module.exports = { longestUniqueSubstring };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(longestUniqueSubstring("abcabcbb"), { length: 3, substring: "abc" });
  assert.deepStrictEqual(longestUniqueSubstring("bbbbb"), { length: 1, substring: "b" });
  assert.deepStrictEqual(longestUniqueSubstring("pwwkew"), { length: 3, substring: "wke" });
  assert.deepStrictEqual(longestUniqueSubstring(""), { length: 0, substring: "" });
  assert.deepStrictEqual(longestUniqueSubstring("dvdf"), { length: 3, substring: "vdf" });
  console.log("PASS");
}
