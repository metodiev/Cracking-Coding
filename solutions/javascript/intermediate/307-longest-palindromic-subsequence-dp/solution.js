/** Task 307 - Longest palindromic subsequence (DP). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function longestPalindromicSubsequence(s) {
  const n = s.length;
  if (n === 0) return 0;
  const dp = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) dp[i][i] = 1;
  for (let length = 2; length <= n; length++) {
    for (let i = 0; i + length - 1 < n; i++) {
      const j = i + length - 1;
      if (s[i] === s[j]) dp[i][j] = length === 2 ? 2 : dp[i + 1][j - 1] + 2;
      else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
    }
  }
  return dp[0][n - 1];
}

function bruteForce(s) {
  let best = 0;
  for (let mask = 0; mask < 1 << s.length; mask++) {
    let candidate = "";
    for (let i = 0; i < s.length; i++) if ((mask >> i) & 1) candidate += s[i];
    if (candidate.length > best && candidate === [...candidate].reverse().join("")) best = candidate.length;
  }
  return best;
}

function lcgStrings(length, count, seed = 9753) {
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

module.exports = { longestPalindromicSubsequence, bruteForce, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestPalindromicSubsequence("bbbab"), 4);
  assert.strictEqual(longestPalindromicSubsequence("cbbd"), 2);
  assert.strictEqual(longestPalindromicSubsequence("abcde"), 1);
  assert.strictEqual(longestPalindromicSubsequence("racecar"), 7);
  assert.strictEqual(longestPalindromicSubsequence(""), 0);
  assert.strictEqual(longestPalindromicSubsequence("a"), 1);
  for (const s of lcgStrings(10, 60)) {
    assert.strictEqual(longestPalindromicSubsequence(s), bruteForce(s), s);
  }
  console.log("PASS");
}
