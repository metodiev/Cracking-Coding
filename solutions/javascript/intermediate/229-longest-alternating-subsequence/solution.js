/** Task 229 - Longest alternating subsequence. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function longestAlternatingSubsequence(nums) {
  if (nums.length === 0) return 0;
  let up = 1, down = 1;
  for (let i = 1; i < nums.length; i++) {
    if (nums[i] > nums[i - 1]) up = down + 1;
    else if (nums[i] < nums[i - 1]) down = up + 1;
  }
  return Math.max(up, down);
}

module.exports = { longestAlternatingSubsequence };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestAlternatingSubsequence([1, 7, 4, 9, 2, 5]), 6);
  assert.strictEqual(longestAlternatingSubsequence([1, 17, 5, 10, 13, 15, 10, 5, 16, 8]), 7);
  assert.strictEqual(longestAlternatingSubsequence([8]), 1);
  assert.strictEqual(longestAlternatingSubsequence([]), 0);
  assert.strictEqual(longestAlternatingSubsequence([1, 2, 3, 4]), 2);
  assert.strictEqual(longestAlternatingSubsequence([2, 1]), 2);
  assert.strictEqual(longestAlternatingSubsequence([3, 3]), 1);
  assert.strictEqual(longestAlternatingSubsequence([3, 3, 3, 2]), 2);
  console.log("PASS");
}
