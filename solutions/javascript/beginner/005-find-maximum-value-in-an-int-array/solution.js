/** Task 005 - Find maximum value in an int array. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function maxValue(items) {
  if (items.length === 0) throw new Error("empty array");
  let best = items[0];
  for (const x of items) if (x > best) best = x;
  return best;
}

module.exports = { maxValue };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(maxValue([3, 7, 2, 9, 4]), 9);
  assert.strictEqual(maxValue([-5, -1, -3]), -1);
  assert.strictEqual(maxValue([42]), 42);
  console.log("PASS");
}
