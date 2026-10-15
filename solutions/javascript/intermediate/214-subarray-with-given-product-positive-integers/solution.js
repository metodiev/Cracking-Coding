/** Task 214 - Subarray with given product (positive integers). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function subarrayWithProduct(nums, target) {
  if (nums.some((x) => x <= 0)) throw new Error("requires positive values");
  if (target < 1) return null;
  let left = 0, product = 1;
  for (let right = 0; right < nums.length; right++) {
    product *= nums[right];
    while (left <= right && product > target) {
      product /= nums[left];
      left++;
    }
    if (left <= right && product === target) return [left, right];
  }
  return null;
}

module.exports = { subarrayWithProduct };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(subarrayWithProduct([2, 3, 1, 2, 4], 6), [0, 1]);
  assert.deepStrictEqual(subarrayWithProduct([1, 2, 3, 4], 6), [0, 2]);
  assert.deepStrictEqual(subarrayWithProduct([10], 10), [0, 0]);
  assert.strictEqual(subarrayWithProduct([3, 1, 1], 2), null);
  assert.deepStrictEqual(subarrayWithProduct([1, 1, 1], 1), [0, 0]);
  console.log("PASS");
}
