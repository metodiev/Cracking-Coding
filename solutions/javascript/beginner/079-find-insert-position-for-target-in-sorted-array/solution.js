/** Task 079 - Find insert position for target in sorted array. (Tier: Beginner, CTCI Ch 10) */
"use strict";

function searchInsert(a, target) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

module.exports = { searchInsert };

if (require.main === module) {
  const assert = require("node:assert");
  const a = [1, 3, 5, 6];
  assert.strictEqual(searchInsert(a, 5), 2);
  assert.strictEqual(searchInsert(a, 2), 1);
  assert.strictEqual(searchInsert(a, 7), 4);
  assert.strictEqual(searchInsert(a, 0), 0);
  console.log("PASS");
}
