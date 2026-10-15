/** Task 204 - Count pairs with given difference K in array (hash). (Tier: Intermediate, CTCI Ch 16) */
"use strict";

function countPairsWithDifference(nums, k) {
  if (k < 0) throw new Error("k must be >= 0");
  const freq = new Map();
  for (const x of nums) freq.set(x, (freq.get(x) ?? 0) + 1);
  if (k === 0) {
    let total = 0;
    for (const c of freq.values()) total += (c * (c - 1)) / 2;
    return total;
  }
  let total = 0;
  for (const [value, count] of freq) total += count * (freq.get(value + k) ?? 0);
  return total;
}

module.exports = { countPairsWithDifference };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countPairsWithDifference([1, 5, 3, 4, 2], 2), 3);
  assert.strictEqual(countPairsWithDifference([1, 1, 1, 1], 0), 6);
  assert.strictEqual(countPairsWithDifference([1, 2, 3, 4], 10), 0);
  assert.strictEqual(countPairsWithDifference([5, 5, 5, 1], 4), 3);
  console.log("PASS");
}
