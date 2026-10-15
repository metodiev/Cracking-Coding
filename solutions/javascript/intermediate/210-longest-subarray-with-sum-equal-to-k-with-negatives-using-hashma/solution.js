/** Task 210 - Longest subarray with sum equal to K (with negatives) using hashmap. (Tier: Intermediate, CTCI Ch 16) */
"use strict";

function longestSubarraySumK(nums, target) {
  const firstIndex = new Map([[0, -1]]);
  let prefix = 0, best = 0;
  for (let j = 0; j < nums.length; j++) {
    prefix += nums[j];
    if (firstIndex.has(prefix - target)) {
      best = Math.max(best, j - firstIndex.get(prefix - target));
    }
    if (!firstIndex.has(prefix)) firstIndex.set(prefix, j);
  }
  return best;
}

module.exports = { longestSubarraySumK };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestSubarraySumK([1, 2, 3, 1, 1, 1, 1], 6), 4);
  assert.strictEqual(longestSubarraySumK([-1, 1, 1], 1), 3);
  assert.strictEqual(longestSubarraySumK([1, -1, 5, -2, 3], 3), 4);
  assert.strictEqual(longestSubarraySumK([-2, -1, 2, 1], 1), 2);
  assert.strictEqual(longestSubarraySumK([0, 0, 0], 0), 3);
  console.log("PASS");
}
