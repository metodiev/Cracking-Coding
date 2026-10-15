/** Task 070 - Insertion sort. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function insertionSort(a) {
  for (let i = 1; i < a.length; i++) {
    const key = a[i];
    let j = i - 1;
    while (j >= 0 && a[j] > key) {
      a[j + 1] = a[j];
      j--;
    }
    a[j + 1] = key;
  }
  return a;
}

module.exports = { insertionSort };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(insertionSort([5, 3, 8, 1, 9, 2]), [1, 2, 3, 5, 8, 9]);
  assert.deepStrictEqual(insertionSort([1, 2, 3]), [1, 2, 3]);
  assert.deepStrictEqual(insertionSort([]), []);
  console.log("PASS");
}
