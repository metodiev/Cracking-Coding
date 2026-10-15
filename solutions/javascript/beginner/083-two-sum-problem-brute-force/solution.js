/** Task 083 - Two-sum problem (brute force). (Tier: Beginner, CTCI Ch 1)
 *
 * O(n^2) double loop. Returns the indices of the first pair found (smallest i,
 * then smallest j > i) whose values add to target, or [-1, -1] when no pair
 * exists.
 */
"use strict";

function twoSumBrute(nums, target) {
  for (let i = 0; i < nums.length; i++) {
    for (let j = i + 1; j < nums.length; j++) {
      if (nums[i] + nums[j] === target) return [i, j];
    }
  }
  return [-1, -1];
}

module.exports = { twoSumBrute };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(twoSumBrute([2, 7, 11, 15], 9), [0, 1]);
  assert.deepStrictEqual(twoSumBrute([3, 3], 6), [0, 1]);
  assert.deepStrictEqual(twoSumBrute([3, 2, 4], 6), [1, 2]);
  assert.deepStrictEqual(twoSumBrute([-3, 4, 3, 90], 0), [0, 2]);
  assert.deepStrictEqual(twoSumBrute([1, 5, 3, 4], 9), [1, 3]);
  assert.deepStrictEqual(twoSumBrute([2, 2, 2, 2], 4), [0, 1]);
  assert.deepStrictEqual(twoSumBrute([1, 2, 3], 7), [-1, -1]);
  assert.deepStrictEqual(twoSumBrute([], 0), [-1, -1]);
  assert.deepStrictEqual(twoSumBrute([5], 5), [-1, -1]);
  console.log("PASS");
}
