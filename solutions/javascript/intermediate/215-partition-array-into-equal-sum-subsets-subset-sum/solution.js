/** Task 215 - Partition array into equal-sum subsets (subset sum). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function canPartition(nums) {
  const total = nums.reduce((a, b) => a + b, 0);
  if (total % 2 !== 0) return false;
  const half = total / 2;
  const dp = new Array(half + 1).fill(false);
  dp[0] = true;
  for (const x of nums) {
    for (let s = half; s >= x; s--) {
      if (dp[s - x]) dp[s] = true;
    }
  }
  return dp[half];
}

module.exports = { canPartition };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(canPartition([1, 5, 11, 5]), true);
  assert.strictEqual(canPartition([1, 2, 3, 5]), false);
  assert.strictEqual(canPartition([2, 2]), true);
  assert.strictEqual(canPartition([1]), false);
  assert.strictEqual(canPartition([]), true);
  console.log("PASS");
}
