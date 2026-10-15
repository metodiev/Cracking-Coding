/** Task 221 - Circular array maximum subarray (Kadane variant). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function maxCircularSubarray(nums) {
  if (nums.length === 0) throw new Error("empty array has no subarray");
  let best = nums[0], cur = nums[0];        // Kadane: maximum subarray
  let worst = nums[0], low = nums[0];       // Kadane: minimum subarray
  const total = nums.reduce((a, b) => a + b, 0);
  for (let i = 1; i < nums.length; i++) {
    cur = Math.max(nums[i], cur + nums[i]);
    best = Math.max(best, cur);
    low = Math.min(nums[i], low + nums[i]);
    worst = Math.min(worst, low);
  }
  if (best < 0) return best;                // all negative: wrapping picks nothing
  return Math.max(best, total - worst);
}

module.exports = { maxCircularSubarray };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(maxCircularSubarray([5, -3, 5]), 10);
  assert.strictEqual(maxCircularSubarray([-3, -2, -3]), -2);
  assert.strictEqual(maxCircularSubarray([1, -2, 3, -2]), 3);
  assert.strictEqual(maxCircularSubarray([1, 2, 3]), 6);
  assert.strictEqual(maxCircularSubarray([8, -8, 9]), 17);
  assert.strictEqual(maxCircularSubarray([11, -1, 11]), 22);
  assert.strictEqual(maxCircularSubarray([5]), 5);
  assert.strictEqual(maxCircularSubarray([-5]), -5);
  assert.strictEqual(maxCircularSubarray([0]), 0);
  assert.throws(() => maxCircularSubarray([]), /empty/);
  console.log("PASS");
}
