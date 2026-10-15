/** Task 087 - Implement merge and count inversions using merge sort. (Tier: Beginner, CTCI Ch 10)
 *
 * Augmented merge sort: when an element from the right half wins the merge, every
 * remaining element of the left half is larger than it, so the running count
 * grows by the number of those remaining elements. O(n log n) time.
 */
"use strict";

function countInversions(nums) {
  const a = nums.slice();
  return mergeSortCount(a, 0, a.length);
}

function mergeSortCount(a, lo, hi) {
  if (hi - lo <= 1) return 0;
  const mid = lo + Math.floor((hi - lo) / 2);
  let count = mergeSortCount(a, lo, mid) + mergeSortCount(a, mid, hi);
  const merged = [];
  let i = lo;
  let j = mid;
  while (i < mid && j < hi) {
    if (a[i] <= a[j]) {
      merged.push(a[i++]);
    } else {
      merged.push(a[j++]);
      count += mid - i;
    }
  }
  while (i < mid) merged.push(a[i++]);
  while (j < hi) merged.push(a[j++]);
  for (let k = 0; k < merged.length; k++) a[lo + k] = merged[k];
  return count;
}

function countInversionsNaive(nums) {
  let count = 0;
  for (let i = 0; i < nums.length; i++) {
    for (let j = i + 1; j < nums.length; j++) {
      if (nums[i] > nums[j]) count++;
    }
  }
  return count;
}

function pseudoRandomList(size, seed = 7) {
  const values = [];
  let state = seed;
  for (let i = 0; i < size; i++) {
    state = (state * 48271) % 2147483647;
    values.push(state % 1000);
  }
  return values;
}

module.exports = { countInversions };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countInversions([]), 0);
  assert.strictEqual(countInversions([1]), 0);
  assert.strictEqual(countInversions([1, 2, 3, 4]), 0);
  assert.strictEqual(countInversions([3, 2, 1]), 3);
  assert.strictEqual(countInversions([5, 4, 3, 2, 1]), 10);
  assert.strictEqual(countInversions([2, 1, 2, 1]), 3);
  assert.strictEqual(countInversions([2, 4, 1, 3, 5]), 3);
  assert.strictEqual(countInversions([1, 1, 1]), 0);

  const reversed = [];
  for (let v = 300; v >= 1; v--) reversed.push(v);
  assert.strictEqual(countInversions(reversed), 44850);

  for (const size of [2, 7, 50, 200]) {
    const nums = pseudoRandomList(size);
    assert.strictEqual(countInversions(nums), countInversionsNaive(nums), `agrees with naive, n=${size}`);
  }
  console.log("PASS");
}
