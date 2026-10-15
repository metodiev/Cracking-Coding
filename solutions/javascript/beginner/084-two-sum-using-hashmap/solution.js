/** Task 084 - Two-sum using hashmap. (Tier: Beginner, CTCI Ch 1)
 *
 * One pass, O(n): keep a map value -> earliest index seen so far and, for each
 * new index j, look up target - nums[j]. A single pass suffices because when
 * index j is reached, every earlier index has already been stored, so any pair
 * (i < j) is detected exactly when its later element arrives; each element is
 * visited once. Returns [i, j] with i < j, or [-1, -1] when no pair exists.
 */
"use strict";

function twoSumHash(nums, target) {
  const seen = new Map();
  for (let j = 0; j < nums.length; j++) {
    const complement = target - nums[j];
    if (seen.has(complement)) {
      return [seen.get(complement), j];
    }
    if (!seen.has(nums[j])) {
      seen.set(nums[j], j);
    }
  }
  return [-1, -1];
}

function twoSumBrute(nums, target) {
  for (let i = 0; i < nums.length; i++) {
    for (let j = i + 1; j < nums.length; j++) {
      if (nums[i] + nums[j] === target) return [i, j];
    }
  }
  return [-1, -1];
}

module.exports = { twoSumHash };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(twoSumHash([2, 7, 11, 15], 9), [0, 1]);
  assert.deepStrictEqual(twoSumHash([3, 2, 4], 6), [1, 2]);
  assert.deepStrictEqual(twoSumHash([3, 3], 6), [0, 1]);
  assert.deepStrictEqual(twoSumHash([-3, 4, 3, 90], 0), [0, 2]);
  assert.deepStrictEqual(twoSumHash([1, 2, 3], 7), [-1, -1]);
  assert.deepStrictEqual(twoSumHash([], 0), [-1, -1]);
  assert.deepStrictEqual(twoSumHash([5], 5), [-1, -1]);

  const cases = [
    [[2, 7, 11, 15], 9], [[3, 2, 4], 6], [[3, 3], 6], [[0, 0, 1], 0],
    [[-3, 4, 3, 90], 0], [[1, 2, 3], 7], [[], 0], [[5], 5], [[1, 5, 3, 4], 9],
  ];
  for (const [nums, target] of cases) {
    const got = twoSumHash(nums, target);
    const brute = twoSumBrute(nums, target);
    assert.strictEqual(got[0] !== -1, brute[0] !== -1, `agreement for ${nums} / ${target}`);
    if (got[0] !== -1) {
      assert.ok(got[0] >= 0 && got[0] < got[1] && got[1] < nums.length, "valid index order");
      assert.strictEqual(nums[got[0]] + nums[got[1]], target, "valid sum");
    }
  }
  console.log("PASS");
}
