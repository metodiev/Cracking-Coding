/** Task 075 - Binary search iterative on sorted array. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function binarySearch(a, target) {
  let lo = 0, hi = a.length - 1;
  while (lo <= hi) {
    const mid = lo + ((hi - lo) >> 1);
    if (a[mid] === target) return mid;
    a[mid] < target ? (lo = mid + 1) : (hi = mid - 1);
  }
  return -1;
}

module.exports = { binarySearch };

if (require.main === module) {
  const assert = require("node:assert");
  const a = [1, 3, 5, 7, 9, 11];
  assert.strictEqual(binarySearch(a, 7), 3);
  assert.strictEqual(binarySearch(a, 1), 0);
  assert.strictEqual(binarySearch(a, 11), 5);
  assert.strictEqual(binarySearch(a, 4), -1);
  assert.strictEqual(binarySearch([], 5), -1);
  console.log("PASS");
}
