/** Task 121 - Longest common subsequence (LCS) DP. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function lcsLength(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1] + 1
        : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  return dp[n][m];
}

function lcsString(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1] + 1
        : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  const out = [];
  let x = n, y = m;
  while (x > 0 && y > 0) {
    if (a[x - 1] === b[y - 1]) {
      out.push(a[x - 1]);
      x--; y--;
    } else if (dp[x - 1][y] >= dp[x][y - 1]) {
      x--;
    } else {
      y--;
    }
  }
  return out.reverse().join("");
}

function isSubsequence(s, t) {
  let j = 0;
  for (let i = 0; i < t.length && j < s.length; i++) if (t[i] === s[j]) j++;
  return j === s.length;
}

module.exports = { lcsLength, lcsString };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(lcsLength("ABCBDAB", "BDCABA"), 4);
  assert.strictEqual(lcsLength("", "abc"), 0);
  assert.strictEqual(lcsLength("abc", "abc"), 3);
  const seq = lcsString("ABCBDAB", "BDCABA");
  assert.strictEqual(seq.length, 4);
  assert.ok(isSubsequence(seq, "ABCBDAB") && isSubsequence(seq, "BDCABA"));
  console.log("PASS");
}
