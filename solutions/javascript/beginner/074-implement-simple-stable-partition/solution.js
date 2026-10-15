/** Task 074 - Implement simple stable partition. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function stablePartition(items, predicate) {
  const yes = items.filter(predicate);
  const no = items.filter(x => !predicate(x));
  return [...yes, ...no];
}

module.exports = { stablePartition };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(stablePartition([1, 2, 3, 4, 5, 6], x => x % 2 === 0), [2, 4, 6, 1, 3, 5]);
  assert.deepStrictEqual(stablePartition([], () => true), []);
  assert.deepStrictEqual(stablePartition([4, 1, 4, 2], x => x === 4), [4, 4, 1, 2]);
  console.log("PASS");
}
