/** Task 209 - Longest subarray with sum equal to K (array of positives). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function longestSubarraySumK(nums, target) {
  if (nums.some((x) => x <= 0)) throw new Error("requires positive values");
  let lo = 0, total = 0, best = 0;
  for (let hi = 0; hi < nums.length; hi++) {
    total += nums[hi];
    while (total > target && lo <= hi) {
      total -= nums[lo];
      lo++;
    }
    if (total === target) best = Math.max(best, hi - lo + 1);
  }
  return best;
}

module.exports = { longestSubarraySumK };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestSubarraySumK([1, 2, 3, 1, 1, 1, 1], 6), 4);
  assert.strictEqual(longestSubarraySumK([1, 1, 1, 1], 2), 2);
  assert.strictEqual(longestSubarraySumK([5], 5), 1);
  assert.strictEqual(longestSubarraySumK([1, 2, 3], 100), 0);
  assert.throws(() => longestSubarraySumK([1, -1, 1], 0), /positive/);
  console.log("PASS");
}
