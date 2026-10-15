/** Task 091 - Max over sliding window naive. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function maxSlidingWindowNaive(nums, k) {
  const n = nums.length;
  if (k <= 0 || k > n) return [];
  const result = [];
  for (let start = 0; start + k <= n; start++) {
    let best = nums[start];
    for (let i = start + 1; i < start + k; i++) {
      if (nums[i] > best) best = nums[i];
    }
    result.push(best);
  }
  return result;
}

module.exports = { maxSlidingWindowNaive };

if (require.main === module) {
  const assert = require("node:assert");
  const nums = [1, 3, -1, -3, 5, 3, 6, 7];
  assert.deepStrictEqual(maxSlidingWindowNaive(nums, 3), [3, 3, 5, 5, 6, 7]);
  assert.deepStrictEqual(maxSlidingWindowNaive(nums, 1), nums);
  assert.deepStrictEqual(maxSlidingWindowNaive(nums, 8), [7]);
  assert.deepStrictEqual(maxSlidingWindowNaive([9, 9, 9], 2), [9, 9]);
  assert.deepStrictEqual(maxSlidingWindowNaive([-5, -2, -9], 2), [-2, -2]);
  assert.deepStrictEqual(maxSlidingWindowNaive([5], 1), [5]);
  assert.deepStrictEqual(maxSlidingWindowNaive([1, 2, 3], 0), []);
  assert.deepStrictEqual(maxSlidingWindowNaive([1, 2, 3], 4), []);
  assert.deepStrictEqual(maxSlidingWindowNaive([], 1), []);
  console.log("PASS");
}
