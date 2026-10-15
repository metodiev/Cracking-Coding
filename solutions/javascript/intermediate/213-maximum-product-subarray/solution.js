/** Task 213 - Maximum product subarray. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function maxProduct(nums) {
  if (nums.length === 0) throw new Error("empty array has no subarray");
  let best = nums[0], curMax = nums[0], curMin = nums[0];
  for (let i = 1; i < nums.length; i++) {
    const x = nums[i];
    const candidates = [x, curMax * x, curMin * x];
    curMax = Math.max(...candidates);
    curMin = Math.min(...candidates);
    best = Math.max(best, curMax);
  }
  return best;
}

module.exports = { maxProduct };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(maxProduct([2, 3, -2, 4]), 6);
  assert.strictEqual(maxProduct([-2, 0, -1]), 0);
  assert.strictEqual(maxProduct([-2, 3, -4]), 24);
  assert.strictEqual(maxProduct([-2]), -2);
  console.log("PASS");
}
