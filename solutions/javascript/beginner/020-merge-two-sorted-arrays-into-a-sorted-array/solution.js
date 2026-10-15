/** Task 020 - Merge two sorted arrays into a sorted array. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function mergeSorted(a, b) {
  const out = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length)
    out.push(a[i] <= b[j] ? a[i++] : b[j++]);
  while (i < a.length) out.push(a[i++]);
  while (j < b.length) out.push(b[j++]);
  return out;
}

module.exports = { mergeSorted };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(mergeSorted([1, 3, 5], [2, 4, 6]), [1, 2, 3, 4, 5, 6]);
  assert.deepStrictEqual(mergeSorted([], [1, 2]), [1, 2]);
  assert.deepStrictEqual(mergeSorted([2, 2], [2]), [2, 2, 2]);
  console.log("PASS");
}
