/** Task 068 - Heapsort using heap operations. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function siftDown(a, i, size) {
  while (true) {
    let largest = i;
    const left = 2 * i + 1, right = 2 * i + 2;
    if (left < size && a[left] > a[largest]) largest = left;
    if (right < size && a[right] > a[largest]) largest = right;
    if (largest === i) return;
    [a[i], a[largest]] = [a[largest], a[i]];
    i = largest;
  }
}

function heapSort(a) {
  const n = a.length;
  for (let i = (n >> 1) - 1; i >= 0; i--) siftDown(a, i, n);
  for (let end = n - 1; end > 0; end--) {
    [a[0], a[end]] = [a[end], a[0]];
    siftDown(a, 0, end);
  }
  return a;
}

module.exports = { heapSort };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(heapSort([5, 3, 8, 1, 9, 2]), [1, 2, 3, 5, 8, 9]);
  assert.deepStrictEqual(heapSort([]), []);
  assert.deepStrictEqual(heapSort([7]), [7]);
  assert.deepStrictEqual(heapSort([3, 3, 1]), [1, 3, 3]);
  console.log("PASS");
}
