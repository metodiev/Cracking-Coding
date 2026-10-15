/** Task 123 - Edit distance (Levenshtein) DP. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function editDistance(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[n][m];
}

function editDistanceRollingRow(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = new Array(b.length + 1).fill(0);
    cur[0] = i;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = a[i - 1] === b[j - 1]
        ? prev[j - 1]
        : 1 + Math.min(prev[j], cur[j - 1], prev[j - 1]);
    }
    prev = cur;
  }
  return prev[b.length];
}

module.exports = { editDistance, editDistanceRollingRow };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(editDistance("kitten", "sitting"), 3);
  assert.strictEqual(editDistance("", "abc"), 3);
  assert.strictEqual(editDistance("same", "same"), 0);
  assert.strictEqual(editDistance("flaw", "lawn"), 2);
  assert.strictEqual(editDistance("abc", "yabd"), editDistanceRollingRow("abc", "yabd"));
  console.log("PASS");
}
