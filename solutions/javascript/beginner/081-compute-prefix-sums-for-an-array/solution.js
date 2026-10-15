/** Task 081 - Compute prefix sums for an array. (Tier: Beginner, CTCI Ch 1)
 *
 * prefix[i] = sum of the first i elements, so prefix.length === nums.length + 1.
 */
"use strict";

function prefixSums(nums) {
  const prefix = new Array(nums.length + 1).fill(0);
  for (let i = 0; i < nums.length; i++) {
    prefix[i + 1] = prefix[i] + nums[i];
  }
  return prefix;
}

module.exports = { prefixSums };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(prefixSums([]), [0]);
  assert.deepStrictEqual(prefixSums([5]), [0, 5]);
  assert.deepStrictEqual(prefixSums([1, 2, 3]), [0, 1, 3, 6]);
  assert.deepStrictEqual(prefixSums([-1, 1, -1]), [0, -1, 0, -1]);
  assert.deepStrictEqual(prefixSums([4, 4, 4, 4]), [0, 4, 8, 12, 16]);
  assert.strictEqual(prefixSums([1, 2, 3, 4, 5]).length, 6);
  console.log("PASS");
}
