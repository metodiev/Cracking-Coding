/** Task 015 - Check if two strings are anagrams. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function areAnagrams(a, b) {
  if (a.length !== b.length) return false;
  const count = new Array(128).fill(0);
  for (let i = 0; i < a.length; i++) {
    count[a.charCodeAt(i)]++;
    count[b.charCodeAt(i)]--;
  }
  return count.every(c => c === 0);
}

module.exports = { areAnagrams };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(areAnagrams("listen", "silent"), true);
  assert.strictEqual(areAnagrams("hello", "world"), false);
  assert.strictEqual(areAnagrams("abc", "ab"), false);
  console.log("PASS");
}
