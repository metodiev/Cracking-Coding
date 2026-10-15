/** Task 122 - Longest common substring (DP). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function longestCommonSubstring(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  let bestLen = 0, bestEnd = 0;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
        if (dp[i][j] > bestLen) {
          bestLen = dp[i][j];
          bestEnd = i;
        }
      }
    }
  }
  return { length: bestLen, substring: a.slice(bestEnd - bestLen, bestEnd) };
}

module.exports = { longestCommonSubstring };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(longestCommonSubstring("abcdef", "zbcdf"), { length: 3, substring: "bcd" });
  assert.deepStrictEqual(longestCommonSubstring("abc", "abc"), { length: 3, substring: "abc" });
  assert.strictEqual(longestCommonSubstring("abc", "xyz").length, 0);
  assert.strictEqual(longestCommonSubstring("", "abc").length, 0);
  console.log("PASS");
}
