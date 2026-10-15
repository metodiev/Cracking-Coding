/** Task 006 - Find minimum value in an int array. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function minValue(items) {
  if (items.length === 0) throw new Error("empty array");
  let best = items[0];
  for (const x of items) if (x < best) best = x;
  return best;
}

module.exports = { minValue };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(minValue([3, 7, 2, 9, 4]), 2);
  assert.strictEqual(minValue([-5, -1, -3]), -5);
  assert.strictEqual(minValue([42]), 42);
  console.log("PASS");
}
