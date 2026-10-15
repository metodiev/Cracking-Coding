/** Task 077 - Find first occurrence of target in sorted array with duplicates. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function firstOccurrence(a, target) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo < a.length && a[lo] === target ? lo : -1;
}

module.exports = { firstOccurrence };

if (require.main === module) {
  const assert = require("node:assert");
  const a = [1, 2, 2, 2, 3, 4, 4];
  assert.strictEqual(firstOccurrence(a, 2), 1);
  assert.strictEqual(firstOccurrence(a, 4), 5);
  assert.strictEqual(firstOccurrence(a, 9), -1);
  console.log("PASS");
}
