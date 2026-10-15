/** Task 086 - Count inversion pairs using naive O(n^2). (Tier: Beginner, CTCI Ch 10)
 *
 * An inversion is a pair i < j with a[i] > a[j]; equal values are not inversions.
 */
"use strict";

function countInversionsNaive(nums) {
  let count = 0;
  for (let i = 0; i < nums.length; i++) {
    for (let j = i + 1; j < nums.length; j++) {
      if (nums[i] > nums[j]) count++;
    }
  }
  return count;
}

module.exports = { countInversionsNaive };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countInversionsNaive([]), 0);
  assert.strictEqual(countInversionsNaive([1]), 0);
  assert.strictEqual(countInversionsNaive([1, 2, 3]), 0);
  assert.strictEqual(countInversionsNaive([1, 1, 1]), 0);
  assert.strictEqual(countInversionsNaive([2, 2]), 0);
  assert.strictEqual(countInversionsNaive([1, 3, 2]), 1);
  assert.strictEqual(countInversionsNaive([3, 2, 1]), 3);
  assert.strictEqual(countInversionsNaive([2, 4, 1, 3, 5]), 3);
  assert.strictEqual(countInversionsNaive([-1, -2, -3]), 3);
  assert.strictEqual(countInversionsNaive([5, 4, 3, 2, 1]), 10);
  console.log("PASS");
}
