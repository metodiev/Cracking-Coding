/** Task 125 - Climbing stairs DP (count ways). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function climbWays(n) {
  if (n < 0) throw new Error("n must be >= 0");
  let a = 1, b = 1;
  for (let i = 0; i < n; i++) [a, b] = [b, a + b];
  return a;
}

function climbWaysGeneral(n, steps) {
  const dp = new Array(n + 1).fill(0);
  dp[0] = 1;
  for (let i = 1; i <= n; i++) {
    for (const s of steps) if (s <= i) dp[i] += dp[i - s];
  }
  return dp[n];
}

module.exports = { climbWays, climbWaysGeneral };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual([0, 1, 2, 3, 4, 5].map(climbWays), [1, 1, 2, 3, 5, 8]);
  assert.strictEqual(climbWays(35), 14930352);
  assert.strictEqual(climbWaysGeneral(4, [1, 2, 3]), 7);
  assert.strictEqual(climbWaysGeneral(0, [1, 2]), 1);
  console.log("PASS");
}
