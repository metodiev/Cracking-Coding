/** Task 007 - Compute sum and average of array elements. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function sumAndAverage(items) {
  if (items.length === 0) return [0, 0];
  const total = items.reduce((a, b) => a + b, 0);
  return [total, total / items.length];
}

module.exports = { sumAndAverage };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(sumAndAverage([1, 2, 3, 4]), [10, 2.5]);
  assert.deepStrictEqual(sumAndAverage([]), [0, 0]);
  console.log("PASS");
}
