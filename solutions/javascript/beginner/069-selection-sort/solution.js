/** Task 069 - Selection sort. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function selectionSort(a) {
  const n = a.length;
  for (let i = 0; i < n - 1; i++) {
    let minIdx = i;
    for (let j = i + 1; j < n; j++) if (a[j] < a[minIdx]) minIdx = j;
    if (minIdx !== i) [a[i], a[minIdx]] = [a[minIdx], a[i]];
  }
  return a;
}

module.exports = { selectionSort };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(selectionSort([5, 3, 8, 1, 9, 2]), [1, 2, 3, 5, 8, 9]);
  assert.deepStrictEqual(selectionSort([]), []);
  assert.deepStrictEqual(selectionSort([2, 1]), [1, 2]);
  console.log("PASS");
}
