/** Task 227 - Longest bitonic subarray. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function longestBitonicSubarray(nums) {
  const n = nums.length;
  if (n === 0) return 0;
  const inc = new Array(n).fill(1);
  for (let i = 1; i < n; i++) {
    if (nums[i] > nums[i - 1]) inc[i] = inc[i - 1] + 1;
  }
  const dec = new Array(n).fill(1);
  for (let i = n - 2; i >= 0; i--) {
    if (nums[i] > nums[i + 1]) dec[i] = dec[i + 1] + 1;
  }
  let best = 0;
  for (let i = 0; i < n; i++) best = Math.max(best, inc[i] + dec[i] - 1);
  return best;
}

module.exports = { longestBitonicSubarray };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestBitonicSubarray([2, 1, 4, 5, 6, 3, 2]), 6);
  assert.strictEqual(longestBitonicSubarray([1, 5, 1]), 3);
  assert.strictEqual(longestBitonicSubarray([1, 2, 3, 4]), 4);
  assert.strictEqual(longestBitonicSubarray([4, 3, 2, 1]), 4);
  assert.strictEqual(longestBitonicSubarray([7]), 1);
  assert.strictEqual(longestBitonicSubarray([]), 0);
  assert.strictEqual(longestBitonicSubarray([1, 1]), 1);
  assert.strictEqual(longestBitonicSubarray([12, 4, 78, 90, 45, 23, 100]), 5);
  console.log("PASS");
}
