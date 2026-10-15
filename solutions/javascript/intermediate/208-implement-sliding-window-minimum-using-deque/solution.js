/** Task 208 - Implement sliding window minimum using deque. (Tier: Intermediate, CTCI Ch 3) */
"use strict";

function slidingWindowMinimum(nums, k) {
  if (k <= 0) throw new Error("k must be positive");
  const result = [];
  const dq = [];                 // indices, values increasing
  for (let i = 0; i < nums.length; i++) {
    while (dq.length && nums[dq[dq.length - 1]] >= nums[i]) dq.pop();
    dq.push(i);
    if (dq[0] <= i - k) dq.shift();
    if (i >= k - 1) result.push(nums[dq[0]]);
  }
  return result;
}

module.exports = { slidingWindowMinimum };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(slidingWindowMinimum([1, 3, -1, -3, 5, 3, 6, 7], 3), [-1, -3, -3, -3, 3, 3]);
  assert.deepStrictEqual(slidingWindowMinimum([1], 1), [1]);
  assert.deepStrictEqual(slidingWindowMinimum([1, 2], 5), []);
  assert.deepStrictEqual(slidingWindowMinimum([5, 5, 5], 2), [5, 5]);
  console.log("PASS");
}
