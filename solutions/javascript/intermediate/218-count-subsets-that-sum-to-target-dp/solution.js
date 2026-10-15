/** Task 218 - Count subsets that sum to target (DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function countSubsets(nums, target) {
  if (target < 0) return 0;
  const dp = new Array(target + 1).fill(0);
  dp[0] = 1;
  for (const x of nums) {
    for (let s = target; s >= x; s--) {
      dp[s] += dp[s - x];
    }
  }
  return dp[target];
}

module.exports = { countSubsets };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countSubsets([1, 1, 1, 1], 2), 6);
  assert.strictEqual(countSubsets([1, 2, 3], 3), 2);
  assert.strictEqual(countSubsets([], 0), 1);
  assert.strictEqual(countSubsets([0], 0), 2);
  assert.strictEqual(countSubsets([2, 3, 5], 5), 2);
  console.log("PASS");
}
