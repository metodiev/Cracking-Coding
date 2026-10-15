/** Task 019 - Check if array is sorted (ascending). (Tier: Beginner, CTCI Ch 1) */
"use strict";

function isSorted(items) {
  for (let i = 0; i + 1 < items.length; i++)
    if (items[i] > items[i + 1]) return false;
  return true;
}

module.exports = { isSorted };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isSorted([]), true);
  assert.strictEqual(isSorted([1, 2, 2, 3]), true);
  assert.strictEqual(isSorted([1, 3, 2]), false);
  console.log("PASS");
}
