/** Task 004 - Remove duplicates from an array (create new array). (Tier: Beginner, CTCI Ch 1) */
"use strict";

function removeDuplicates(items) {
  const seen = new Set();
  const out = [];
  for (const x of items) {
    if (!seen.has(x)) {
      seen.add(x);
      out.push(x);
    }
  }
  return out;
}

module.exports = { removeDuplicates };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(removeDuplicates([1, 2, 2, 3, 1]), [1, 2, 3]);
  assert.deepStrictEqual(removeDuplicates([]), []);
  assert.deepStrictEqual(removeDuplicates([5, 5, 5]), [5]);
  console.log("PASS");
}
