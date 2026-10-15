/** Task 230 - Longest palindrome substring (expand center). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function longestPalindromeSubstring(s) {
  if (s.length === 0) return "";
  let start = 0, length = 1;
  for (let c = 0; c < s.length; c++) {
    for (const [l, r] of [[c, c], [c, c + 1]]) {   // odd center, then even center
      let a = l, b = r;
      while (a >= 0 && b < s.length && s[a] === s[b]) { a--; b++; }
      const size = b - a - 1;
      if (size > length) {                          // strict: first found wins ties
        start = a + 1;
        length = size;
      }
    }
  }
  return s.slice(start, start + length);
}

module.exports = { longestPalindromeSubstring };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestPalindromeSubstring("babad"), "bab");
  assert.strictEqual(longestPalindromeSubstring("cbbd"), "bb");
  assert.strictEqual(longestPalindromeSubstring("a"), "a");
  assert.strictEqual(longestPalindromeSubstring(""), "");
  assert.strictEqual(longestPalindromeSubstring("aaaa"), "aaaa");
  assert.strictEqual(longestPalindromeSubstring("forgeeksskeegfor"), "geeksskeeg");
  assert.strictEqual(longestPalindromeSubstring("abaxyzzyxf"), "xyzzyx");
  assert.strictEqual(longestPalindromeSubstring("ac"), "a");
  assert.strictEqual(longestPalindromeSubstring("aac"), "aa");
  console.log("PASS");
}
