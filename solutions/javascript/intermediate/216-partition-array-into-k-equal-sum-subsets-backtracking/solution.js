/** Task 216 - Partition array into k equal-sum subsets (backtracking). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function canPartitionK(nums, k) {
  if (k <= 0) throw new Error("k must be positive");
  const total = nums.reduce((a, b) => a + b, 0);
  if (total % k !== 0) return false;
  const target = total / k;
  const a = [...nums].sort((x, y) => y - x);
  if (a.length && a[0] > target) return false;
  const buckets = new Array(k).fill(0);

  function place(i) {
    if (i === a.length) return true;
    const seen = new Set();
    for (let b = 0; b < k; b++) {
      if (seen.has(buckets[b])) continue;
      if (buckets[b] + a[i] <= target) {
        seen.add(buckets[b]);
        buckets[b] += a[i];
        if (place(i + 1)) return true;
        buckets[b] -= a[i];
      }
      if (buckets[b] === 0) break;
    }
    return false;
  }

  return place(0);
}

module.exports = { canPartitionK };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(canPartitionK([4, 3, 2, 3, 5, 2, 1], 4), true);
  assert.strictEqual(canPartitionK([1, 2, 3, 4], 2), true);
  assert.strictEqual(canPartitionK([1, 1, 1, 1], 2), true);
  assert.strictEqual(canPartitionK([2, 2, 2, 2, 3], 3), false);
  assert.strictEqual(canPartitionK([2, 2, 2, 2, 2, 2], 4), false);
  assert.strictEqual(canPartitionK([10, 10, 10, 10, 10, 10], 3), true);
  console.log("PASS");
}
