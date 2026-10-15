/** Task 212 - Find subarray with maximum sum (Kadane's algorithm). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function maxSubarray(nums) {
  if (nums.length === 0) throw new Error("empty array has no subarray");
  let best = nums[0], cur = nums[0];
  let bestStart = 0, bestEnd = 0, start = 0;
  for (let i = 1; i < nums.length; i++) {
    if (cur + nums[i] < nums[i]) {
      start = i;
      cur = nums[i];
    } else {
      cur += nums[i];
    }
    if (cur > best) {
      best = cur;
      bestStart = start;
      bestEnd = i;
    }
  }
  return { sum: best, start: bestStart, end: bestEnd + 1 };
}

module.exports = { maxSubarray };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(maxSubarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), { sum: 6, start: 3, end: 7 });
  assert.strictEqual(maxSubarray([-3, -1, -2]).sum, -1);
  assert.strictEqual(maxSubarray([5, -1, 5]).sum, 9);
  assert.throws(() => maxSubarray([]), /empty/);
  console.log("PASS");
}
