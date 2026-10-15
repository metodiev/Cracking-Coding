/** Task 072 - Count sort for small integer range. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function countingSort(a) {
  if (a.length === 0) return [];
  const lo = Math.min(...a), hi = Math.max(...a);
  const counts = new Array(hi - lo + 1).fill(0);
  for (const x of a) counts[x - lo]++;
  const out = [];
  for (let offset = 0; offset < counts.length; offset++) {
    for (let c = 0; c < counts[offset]; c++) out.push(lo + offset);
  }
  return out;
}

module.exports = { countingSort };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(countingSort([4, 2, 2, 8, 3, 3, 1]), [1, 2, 2, 3, 3, 4, 8]);
  assert.deepStrictEqual(countingSort([]), []);
  assert.deepStrictEqual(countingSort([-3, -1, -3]), [-3, -3, -1]);
  console.log("PASS");
}
