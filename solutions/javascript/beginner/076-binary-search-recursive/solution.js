/** Task 076 - Binary search recursive. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function binarySearchRecursive(a, target, lo = 0, hi = a.length - 1) {
  if (lo > hi) return -1;
  const mid = lo + ((hi - lo) >> 1);
  if (a[mid] === target) return mid;
  if (a[mid] < target) return binarySearchRecursive(a, target, mid + 1, hi);
  return binarySearchRecursive(a, target, lo, mid - 1);
}

module.exports = { binarySearchRecursive };

if (require.main === module) {
  const assert = require("node:assert");
  const a = [2, 4, 6, 8, 10];
  assert.strictEqual(binarySearchRecursive(a, 8), 3);
  assert.strictEqual(binarySearchRecursive(a, 1), -1);
  assert.strictEqual(binarySearchRecursive([], 1), -1);
  console.log("PASS");
}
