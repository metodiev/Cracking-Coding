/** Task 073 - Radix sort for non-negative integers. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function radixSort(a) {
  if (a.length === 0) return [];
  const max = Math.max(...a);
  let out = a.slice();
  for (let exp = 1; Math.floor(max / exp) > 0; exp *= 10) {
    const buckets = Array.from({ length: 10 }, () => []);
    for (const x of out) buckets[Math.floor(x / exp) % 10].push(x);
    out = buckets.flat();
  }
  return out;
}

module.exports = { radixSort };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(radixSort([170, 45, 75, 90, 802, 24, 2, 66]), [2, 24, 45, 66, 75, 90, 170, 802]);
  assert.deepStrictEqual(radixSort([]), []);
  assert.deepStrictEqual(radixSort([0, 10, 100]), [0, 10, 100]);
  console.log("PASS");
}
