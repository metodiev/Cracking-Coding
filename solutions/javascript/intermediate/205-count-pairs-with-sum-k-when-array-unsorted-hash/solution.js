/** Task 205 - Count pairs with sum K when array unsorted (hash). (Tier: Intermediate, CTCI Ch 16) */
"use strict";

function countPairsWithSum(nums, target) {
  const seen = new Map();
  let total = 0;
  for (const x of nums) {
    total += seen.get(target - x) ?? 0;
    seen.set(x, (seen.get(x) ?? 0) + 1);
  }
  return total;
}

module.exports = { countPairsWithSum };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countPairsWithSum([1, 5, 3, 3, 3], 6), 4);
  assert.strictEqual(countPairsWithSum([1, 2, 3, 4], 5), 2);
  assert.strictEqual(countPairsWithSum([1, 1, 1, 1], 2), 6);
  assert.strictEqual(countPairsWithSum([1, 2, 3], 100), 0);
  console.log("PASS");
}
