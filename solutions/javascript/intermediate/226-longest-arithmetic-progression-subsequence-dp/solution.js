/** Task 226 - Longest arithmetic progression subsequence (DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function longestArithmeticSubsequence(nums) {
  if (nums.length === 0) return 0;
  const dp = nums.map(() => new Map());     // dp[i].get(d) = longest AP ending at i, step d
  let best = 1;
  for (let i = 1; i < nums.length; i++) {
    for (let j = 0; j < i; j++) {
      const d = nums[i] - nums[j];
      const length = (dp[j].get(d) || 1) + 1;
      if (length > (dp[i].get(d) || 0)) dp[i].set(d, length);
      best = Math.max(best, length);
    }
  }
  return best;
}

module.exports = { longestArithmeticSubsequence };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestArithmeticSubsequence([3, 6, 9, 12]), 4);
  assert.strictEqual(longestArithmeticSubsequence([9, 4, 7, 2, 10]), 3);
  assert.strictEqual(longestArithmeticSubsequence([1, 2]), 2);
  assert.strictEqual(longestArithmeticSubsequence([4]), 1);
  assert.strictEqual(longestArithmeticSubsequence([]), 0);
  assert.strictEqual(longestArithmeticSubsequence([1, 1, 1]), 3);
  assert.strictEqual(longestArithmeticSubsequence([2, 4, 6, 8, 10]), 5);
  assert.strictEqual(longestArithmeticSubsequence([1, 5, 9, 13, 2, 6, 10, 14]), 4);
  assert.strictEqual(longestArithmeticSubsequence([5, 5, 3, 1]), 3);
  console.log("PASS");
}
