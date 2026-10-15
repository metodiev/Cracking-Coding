/** Task 092 - Max over sliding window using deque. (Tier: Beginner, CTCI Ch 3) */
"use strict";

function maxSlidingWindowNaive(nums, k) {
  const n = nums.length;
  if (k <= 0 || k > n) return [];
  const result = [];
  for (let start = 0; start + k <= n; start++) {
    let best = nums[start];
    for (let i = start + 1; i < start + k; i++) {
      if (nums[i] > best) best = nums[i];
    }
    result.push(best);
  }
  return result;
}

function maxSlidingWindowDeque(nums, k) {
  const n = nums.length;
  if (k <= 0 || k > n) return [];
  const dq = [];   // indices, values decreasing; head pointer keeps this O(n)
  let head = 0;
  const result = [];
  for (let i = 0; i < n; i++) {
    while (dq.length > head && nums[dq[dq.length - 1]] <= nums[i]) dq.pop();
    dq.push(i);
    if (dq[head] <= i - k) head++;
    if (i >= k - 1) result.push(nums[dq[head]]);
  }
  return result;
}

module.exports = { maxSlidingWindowNaive, maxSlidingWindowDeque };

if (require.main === module) {
  const assert = require("node:assert");
  const nums = [1, 3, -1, -3, 5, 3, 6, 7];
  assert.deepStrictEqual(maxSlidingWindowDeque(nums, 3), [3, 3, 5, 5, 6, 7]);
  assert.deepStrictEqual(maxSlidingWindowDeque(nums, 1), nums);
  assert.deepStrictEqual(maxSlidingWindowDeque(nums, 8), [7]);
  assert.deepStrictEqual(maxSlidingWindowDeque([9, 9, 9], 2), [9, 9]);
  assert.deepStrictEqual(maxSlidingWindowDeque([-5, -2, -9], 2), [-2, -2]);
  assert.deepStrictEqual(maxSlidingWindowDeque([5], 1), [5]);
  assert.deepStrictEqual(maxSlidingWindowDeque([1, 2, 3], 0), []);
  assert.deepStrictEqual(maxSlidingWindowDeque([1, 2, 3], 4), []);
  assert.deepStrictEqual(maxSlidingWindowDeque([], 1), []);

  const arr = [9, -4, 7, 7, 2, -1, 8, 0, 5, 3];
  for (let k = 0; k <= arr.length + 1; k++) {
    assert.deepStrictEqual(
      maxSlidingWindowDeque(arr, k),
      maxSlidingWindowNaive(arr, k),
      `cross-check k=${k}`
    );
  }

  const samples = [
    [[4, 2, 12, 3, 8, 7, 5, 9, 1, 6], 4],
    [[-1, -1, -2, -3], 2],
    [[1, 1, 1, 1], 3],
    [[10, -10, 5, -5, 0], 5],
    [[7, 2], 2],
  ];
  for (const [sample, k] of samples) {
    assert.deepStrictEqual(maxSlidingWindowDeque(sample, k), maxSlidingWindowNaive(sample, k));
  }
  console.log("PASS");
}
