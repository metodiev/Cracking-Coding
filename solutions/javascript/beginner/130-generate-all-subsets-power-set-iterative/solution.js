/** Task 130 - Generate all subsets (power set) iterative. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function subsetsBitmask(items) {
  const n = items.length;
  const out = [];
  for (let mask = 0; mask < (1 << n); mask++) {
    const subset = [];
    for (let i = 0; i < n; i++) if (mask & (1 << i)) subset.push(items[i]);
    out.push(subset);
  }
  return out;
}

function subsetsIncremental(items) {
  let out = [[]];
  for (const x of items) {
    out = out.concat(out.map((subset) => [...subset, x]));
  }
  return out;
}

module.exports = { subsetsBitmask, subsetsIncremental };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(subsetsBitmask([1, 2, 3]), [
    [], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3],
  ]);
  assert.deepStrictEqual(subsetsBitmask([]), [[]]);
  assert.deepStrictEqual(subsetsIncremental([1, 2]), [[], [1], [2], [1, 2]]);
  assert.strictEqual(subsetsBitmask([1, 2, 3, 4]).length, 16);
  console.log("PASS");
}
