/** Task 228 - Maximum sum increasing subsequence. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function maxSumIncreasingSubsequence(nums) {
  if (nums.length === 0) return 0;
  const dp = nums.slice();
  let best = dp[0];
  for (let i = 1; i < nums.length; i++) {
    for (let j = 0; j < i; j++) {
      if (nums[j] < nums[i] && dp[j] + nums[i] > dp[i]) dp[i] = dp[j] + nums[i];
    }
    best = Math.max(best, dp[i]);
  }
  return best;
}

module.exports = { maxSumIncreasingSubsequence };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(maxSumIncreasingSubsequence([1, 101, 2, 3, 100, 4]), 106);
  assert.strictEqual(maxSumIncreasingSubsequence([10, 1]), 10);
  assert.strictEqual(maxSumIncreasingSubsequence([5, 4, 3, 2, 1]), 5);
  assert.strictEqual(maxSumIncreasingSubsequence([1, 2, 3, 4]), 10);
  assert.strictEqual(maxSumIncreasingSubsequence([7]), 7);
  assert.strictEqual(maxSumIncreasingSubsequence([]), 0);
  assert.strictEqual(maxSumIncreasingSubsequence([3, 3]), 3);
  assert.strictEqual(maxSumIncreasingSubsequence([-2, -1, -3]), -1);
  console.log("PASS");
}
