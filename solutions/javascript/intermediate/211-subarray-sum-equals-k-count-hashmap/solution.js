/** Task 211 - Subarray sum equals K count (hashmap). (Tier: Intermediate, CTCI Ch 16) */
"use strict";

function countSubarraysSumK(nums, target) {
  const counts = new Map([[0, 1]]);
  let prefix = 0, total = 0;
  for (const x of nums) {
    prefix += x;
    total += counts.get(prefix - target) ?? 0;
    counts.set(prefix, (counts.get(prefix) ?? 0) + 1);
  }
  return total;
}

module.exports = { countSubarraysSumK };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countSubarraysSumK([1, 1, 1], 2), 2);
  assert.strictEqual(countSubarraysSumK([1, 2, 3], 3), 2);
  assert.strictEqual(countSubarraysSumK([-1, -1, 1], 0), 1);
  assert.strictEqual(countSubarraysSumK([0, 0], 0), 3);
  console.log("PASS");
}
