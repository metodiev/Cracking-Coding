/** Task 094 - Find longest common prefix in array of strings. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function longestCommonPrefix(strs) {
  if (strs.length === 0) return "";
  const first = strs[0];
  for (let i = 0; i < first.length; i++) {
    const ch = first[i];
    for (let k = 1; k < strs.length; k++) {
      if (i >= strs[k].length || strs[k][i] !== ch) return first.slice(0, i);
    }
  }
  return first;
}

module.exports = { longestCommonPrefix };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestCommonPrefix(["flower", "flow", "flight"]), "fl");
  assert.strictEqual(longestCommonPrefix(["dog", "racecar", "car"]), "");
  assert.strictEqual(longestCommonPrefix(["apple"]), "apple");
  assert.strictEqual(longestCommonPrefix([]), "");
  assert.strictEqual(longestCommonPrefix([""]), "");
  assert.strictEqual(longestCommonPrefix(["", "a"]), "");
  assert.strictEqual(longestCommonPrefix(["a", "ab"]), "a");
  assert.strictEqual(longestCommonPrefix(["abc", "abc", "abc"]), "abc");
  assert.strictEqual(longestCommonPrefix(["prefix", "pref"]), "pref");
  assert.strictEqual(longestCommonPrefix(["ab", "ac"]), "a");
  console.log("PASS");
}
