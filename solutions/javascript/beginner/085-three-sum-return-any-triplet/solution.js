/** Task 085 - Three-sum (return any triplet). (Tier: Beginner, CTCI Ch 1)
 *
 * Sort a copy, then fix the smallest value and use two pointers for the rest:
 * O(n^2) time. Returns any triplet of values (v1 <= v2 <= v3) drawn from three
 * distinct positions that sums to 0, or [] when none exists.
 */
"use strict";

function threeSum(nums) {
  if (nums.length < 3) return [];
  const values = nums.slice().sort((a, b) => a - b);
  const n = values.length;
  for (let i = 0; i < n - 2; i++) {
    if (i > 0 && values[i] === values[i - 1]) continue;
    let lo = i + 1;
    let hi = n - 1;
    while (lo < hi) {
      const total = values[i] + values[lo] + values[hi];
      if (total === 0) return [values[i], values[lo], values[hi]];
      if (total < 0) lo++;
      else hi--;
    }
  }
  return [];
}

function isSorted(a) {
  for (let i = 1; i < a.length; i++) if (a[i - 1] > a[i]) return false;
  return true;
}

function containsTriplet(nums, triplet) {
  if (triplet.length !== 3) return false;
  const used = new Array(nums.length).fill(false);
  for (const v of triplet) {
    let found = false;
    for (let i = 0; i < nums.length; i++) {
      if (!used[i] && nums[i] === v) {
        used[i] = true;
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

module.exports = { threeSum };

if (require.main === module) {
  const assert = require("node:assert");
  const nums = [-1, 0, 1, 2, -1, -4];
  const t = threeSum(nums);
  assert.strictEqual(t.length, 3);
  assert.strictEqual(t[0] + t[1] + t[2], 0);
  assert.ok(isSorted(t), "sorted output");
  assert.ok(containsTriplet(nums, t), "three distinct positions");

  assert.deepStrictEqual(threeSum([0, 0, 0]), [0, 0, 0]);
  assert.deepStrictEqual(threeSum([-3, 1, 2]), [-3, 1, 2]);
  assert.deepStrictEqual(threeSum([1, -1, 0]), [-1, 0, 1]);
  assert.deepStrictEqual(threeSum([-2, 0, 0, 2, 2]), [-2, 0, 2]);
  assert.deepStrictEqual(threeSum([1, 2, 3]), []);
  assert.deepStrictEqual(threeSum([1, 1, 1]), []);
  assert.deepStrictEqual(threeSum([0, 0]), []);
  assert.deepStrictEqual(threeSum([]), []);

  const nums2 = [8, -4, -2, 3, 1, -5, 7];
  const t2 = threeSum(nums2);
  assert.strictEqual(t2.length, 3);
  assert.strictEqual(t2[0] + t2[1] + t2[2], 0);
  assert.ok(isSorted(t2) && containsTriplet(nums2, t2));
  console.log("PASS");
}
