/** Task 090 - Implement sliding window sum of size k. (Tier: Beginner, CTCI Ch 1)
 *
 * O(n) running-sum window: subtract the element leaving the window and add the
 * one entering it. Returns the sums of every contiguous window of size k in
 * left-to-right order; k === 0 or k > n returns [] (no windows); k < 0 throws.
 */
"use strict";

function windowSums(nums, k) {
  if (k < 0) throw new Error("k must be >= 0");
  const n = nums.length;
  if (k === 0 || k > n) return [];
  const result = [];
  let window = 0;
  for (let i = 0; i < k; i++) window += nums[i];
  result.push(window);
  for (let i = k; i < n; i++) {
    window += nums[i] - nums[i - k];
    result.push(window);
  }
  return result;
}

module.exports = { windowSums };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(windowSums([1, 2, 3, 4, 5], 3), [6, 9, 12]);
  assert.deepStrictEqual(windowSums([1, 2, 3, 4, 5], 1), [1, 2, 3, 4, 5]);
  assert.deepStrictEqual(windowSums([1, 2, 3, 4, 5], 5), [15]);
  assert.deepStrictEqual(windowSums([1, 2, 3, 4, 5], 6), []);
  assert.deepStrictEqual(windowSums([1, 2, 3, 4, 5], 0), []);
  assert.deepStrictEqual(windowSums([], 1), []);
  assert.deepStrictEqual(windowSums([], 0), []);
  assert.deepStrictEqual(windowSums([4], 1), [4]);
  assert.deepStrictEqual(windowSums([-1, -2, -3, -4], 2), [-3, -5, -7]);
  assert.deepStrictEqual(windowSums([2, 2, 2, 2], 2), [4, 4, 4]);
  assert.throws(() => windowSums([1, 2], -1), "k < 0");

  const n = 20000;
  const k = 100;
  const nums = Array.from({ length: n }, (_, i) => i);
  const sums = windowSums(nums, k);
  assert.strictEqual(sums.length, n - k + 1);
  assert.strictEqual(sums[0], (k * (k - 1)) / 2);
  assert.strictEqual(sums[sums.length - 1], (n - k) * k + (k * (k - 1)) / 2);
  console.log("PASS");
}
