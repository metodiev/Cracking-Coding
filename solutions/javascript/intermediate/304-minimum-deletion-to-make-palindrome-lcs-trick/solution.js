/** Task 304 - Minimum deletions to make a palindrome (LCS trick). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function lcsLength(a, b) {
  let previous = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    const current = new Array(b.length + 1).fill(0);
    for (let j = 1; j <= b.length; j++) {
      current[j] = a[i - 1] === b[j - 1] ? previous[j - 1] + 1 : Math.max(previous[j], current[j - 1]);
    }
    previous = current;
  }
  return previous[b.length];
}

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

const minimumDeletions = (s) => s.length - longestPalindromicSubsequence(s);

function lcgStrings(length, count, seed = 3579) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += "abcd"[seed % 4];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { lcsLength, longestPalindromicSubsequence, minimumDeletions, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(minimumDeletions("aebcbda"), 2);
  assert.strictEqual(minimumDeletions("abc"), 2);
  assert.strictEqual(minimumDeletions("abcd"), 3);
  assert.strictEqual(minimumDeletions("aa"), 0);
  assert.strictEqual(minimumDeletions("racecar"), 0);
  assert.strictEqual(minimumDeletions(""), 0);
  assert.strictEqual(minimumDeletions("a"), 0);

  for (const s of ["aebcbda", "geeksforgeeks", "abca", "abbc", ""]) {
    assert.strictEqual(longestPalindromicSubsequence(s), lcsLength(s, [...s].reverse().join("")), s);
  }
  for (const s of lcgStrings(10, 60)) {
    const lps = longestPalindromicSubsequence(s);
    assert.strictEqual(lps, lcsLength(s, [...s].reverse().join("")), s);
    assert.ok(s.length - lps >= 0 && s.length - lps <= s.length);
  }
  console.log("PASS");
}
