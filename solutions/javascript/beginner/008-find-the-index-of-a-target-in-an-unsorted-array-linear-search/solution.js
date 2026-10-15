/** Task 008 - Find the index of a target in an unsorted array (linear search). (Tier: Beginner, CTCI Ch 1) */
"use strict";

function linearSearch(items, target) {
  for (let i = 0; i < items.length; i++) if (items[i] === target) return i;
  return -1;
}

module.exports = { linearSearch };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(linearSearch([4, 2, 7, 1], 7), 2);
  assert.strictEqual(linearSearch([4, 2, 7, 1], 9), -1);
  assert.strictEqual(linearSearch([], 1), -1);
  console.log("PASS");
}
