/** Task 303 - Count palindromic substrings. (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function countPalindromicSubstrings(s) {
  const n = s.length;
  let count = 0;
  for (let center = 0; center < n; center++) {
    for (const [left0, right0] of [[center, center], [center, center + 1]]) {
      let left = left0;
      let right = right0;
      while (left >= 0 && right < n && s[left] === s[right]) {
        count++;
        left--;
        right++;
      }
    }
  }
  return count;
}

function bruteForce(s) {
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    for (let j = i + 1; j <= s.length; j++) {
      const part = s.slice(i, j);
      if (part === [...part].reverse().join("")) total++;
    }
  }
  return total;
}

function lcgStrings(length, count, seed = 2468) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += "abc"[seed % 3];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { countPalindromicSubstrings, bruteForce, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countPalindromicSubstrings("abc"), 3);
  assert.strictEqual(countPalindromicSubstrings("aaa"), 6);
  assert.strictEqual(countPalindromicSubstrings("abba"), 6);
  assert.strictEqual(countPalindromicSubstrings(""), 0);
  assert.strictEqual(countPalindromicSubstrings("a"), 1);
  for (const s of ["aab", "aba", "abcba", "abccba", "aaaa"]) {
    assert.strictEqual(countPalindromicSubstrings(s), bruteForce(s), s);
  }
  for (const s of lcgStrings(9, 60)) assert.strictEqual(countPalindromicSubstrings(s), bruteForce(s), s);
  console.log("PASS");
}
