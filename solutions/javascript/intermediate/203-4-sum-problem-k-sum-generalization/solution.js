/** Task 203 - 4-sum problem (k-sum generalization). (Tier: Intermediate, CTCI Ch 16) */
"use strict";

function kSum(nums, target, k) {
  const a = [...nums].sort((x, y) => x - y);
  const results = [];

  function search(start, remaining, kLeft, current) {
    if (kLeft === 2) {
      let lo = start, hi = a.length - 1;
      while (lo < hi) {
        const total = a[lo] + a[hi];
        if (total === remaining) {
          results.push([...current, a[lo], a[hi]]);
          while (lo < hi && a[lo] === a[lo + 1]) lo++;
          while (lo < hi && a[hi] === a[hi - 1]) hi--;
          lo++;
          hi--;
        } else if (total < remaining) {
          lo++;
        } else {
          hi--;
        }
      }
      return;
    }
    for (let i = start; i + kLeft <= a.length; i++) {
      if (i > start && a[i] === a[i - 1]) continue;
      if (a[i] * kLeft > remaining) break;
      search(i + 1, remaining - a[i], kLeft - 1, [...current, a[i]]);
    }
  }

  search(0, target, k, []);
  return results;
}

function fourSum(nums, target) {
  return kSum(nums, target, 4);
}

module.exports = { kSum, fourSum };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(fourSum([1, 0, -1, 0, -2, 2], 0), [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]);
  assert.deepStrictEqual(fourSum([2, 2, 2, 2, 2], 8), [[2, 2, 2, 2]]);
  assert.deepStrictEqual(fourSum([1, 2, 3, 4], 100), []);
  assert.deepStrictEqual(kSum([1, 2, 3, 4], 5, 2), [[1, 4], [2, 3]]);
  assert.deepStrictEqual(kSum([-1, 0, 1, 2, -1, -4], 0, 3), [[-1, -1, 2], [-1, 0, 1]]);
  console.log("PASS");
}
