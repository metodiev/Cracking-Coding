/** Task 217 - Subset sum decision problem (DP boolean). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function subsetSum(nums, target) {
  if (target < 0) return false;
  const dp = new Array(target + 1).fill(false);
  dp[0] = true;
  for (const x of nums) {
    for (let s = target; s >= x; s--) {
      if (dp[s - x]) dp[s] = true;
    }
  }
  return dp[target];
}

function subsetSumRecursive(nums, target) {
  const memo = new Map();
  const go = (i, remaining) => {
    if (remaining === 0) return true;
    if (i === nums.length || remaining < 0) return false;
    const key = `${i}:${remaining}`;
    if (memo.has(key)) return memo.get(key);
    const value = go(i + 1, remaining) || go(i + 1, remaining - nums[i]);
    memo.set(key, value);
    return value;
  };
  return go(0, target);
}

module.exports = { subsetSum, subsetSumRecursive };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(subsetSum([3, 34, 4, 12, 5, 2], 9), true);
  assert.strictEqual(subsetSum([3, 34, 4, 12, 5, 2], 30), false);
  assert.strictEqual(subsetSum([], 0), true);
  assert.strictEqual(subsetSum([5], 5), true);
  assert.strictEqual(subsetSum([5], 3), false);
  assert.strictEqual(subsetSum([4, 5, 2], 9), subsetSumRecursive([4, 5, 2], 9));
  console.log("PASS");
}
