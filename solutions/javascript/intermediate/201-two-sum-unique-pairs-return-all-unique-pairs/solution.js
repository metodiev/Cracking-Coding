/** Task 201 - Two-sum unique pairs (return all unique pairs). (Tier: Intermediate, CTCI Ch 16) */
"use strict";

function twoSumUniquePairs(nums, target) {
  const seen = new Set();
  const pairs = new Map();          // key "a,b" -> [a, b] for dedup
  for (const x of nums) {
    const y = target - x;
    if (seen.has(y)) {
      const a = Math.min(x, y), b = Math.max(x, y);
      pairs.set(`${a},${b}`, [a, b]);
    }
    seen.add(x);
  }
  return [...pairs.values()].sort((p, q) => (p[0] - q[0]) || (p[1] - q[1]));
}

module.exports = { twoSumUniquePairs };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(twoSumUniquePairs([1, 1, 2, 3, 2], 4), [[1, 3], [2, 2]]);
  assert.deepStrictEqual(twoSumUniquePairs([1, 5, 3, 3, 5], 6), [[1, 5], [3, 3]]);
  assert.deepStrictEqual(twoSumUniquePairs([1, 2, 3], 100), []);
  assert.deepStrictEqual(twoSumUniquePairs([0, 0, 0], 0), [[0, 0]]);
  assert.deepStrictEqual(twoSumUniquePairs([-1, 1, 0], 0), [[-1, 1]]);
  console.log("PASS");
}
