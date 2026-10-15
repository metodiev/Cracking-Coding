/** Task 202 - 3-sum all unique triplets. (Tier: Intermediate, CTCI Ch 16) */
"use strict";

function threeSum(nums) {
  const a = [...nums].sort((x, y) => x - y);
  const out = [];
  for (let i = 0; i + 2 < a.length; i++) {
    if (i > 0 && a[i] === a[i - 1]) continue;
    if (a[i] > 0) break;
    let lo = i + 1, hi = a.length - 1;
    while (lo < hi) {
      const total = a[i] + a[lo] + a[hi];
      if (total === 0) {
        out.push([a[i], a[lo], a[hi]]);
        while (lo < hi && a[lo] === a[lo + 1]) lo++;
        while (lo < hi && a[hi] === a[hi - 1]) hi--;
        lo++;
        hi--;
      } else if (total < 0) {
        lo++;
      } else {
        hi--;
      }
    }
  }
  return out;
}

module.exports = { threeSum };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(threeSum([-1, 0, 1, 2, -1, -4]), [[-1, -1, 2], [-1, 0, 1]]);
  assert.deepStrictEqual(threeSum([0, 1, 1]), []);
  assert.deepStrictEqual(threeSum([0, 0, 0]), [[0, 0, 0]]);
  assert.deepStrictEqual(threeSum([-2, 0, 0, 2, 2]), [[-2, 0, 2]]);
  console.log("PASS");
}
