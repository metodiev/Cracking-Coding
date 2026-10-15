/** Task 078 - Find last occurrence of target in sorted array with duplicates. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function lastOccurrence(a, target) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] <= target) lo = mid + 1;
    else hi = mid;
  }
  const idx = lo - 1;
  return idx >= 0 && a[idx] === target ? idx : -1;
}

module.exports = { lastOccurrence };

if (require.main === module) {
  const assert = require("node:assert");
  const a = [1, 2, 2, 2, 3, 4, 4];
  assert.strictEqual(lastOccurrence(a, 2), 3);
  assert.strictEqual(lastOccurrence(a, 4), 6);
  assert.strictEqual(lastOccurrence(a, 9), -1);
  console.log("PASS");
}
