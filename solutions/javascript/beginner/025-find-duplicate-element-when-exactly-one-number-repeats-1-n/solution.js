/** Task 025 - Find duplicate element when exactly one number repeats (1..n). (Tier: Beginner, CTCI Ch 1) */
function findDuplicate(nums) {
  let slow = nums[0], fast = nums[0];
  do {
    slow = nums[slow];
    fast = nums[nums[fast]];
  } while (slow !== fast);
  slow = nums[0];
  while (slow !== fast) { slow = nums[slow]; fast = nums[fast]; }
  return slow;
}

module.exports = { findDuplicate };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(findDuplicate([1, 3, 4, 2, 2]), 2);
  assert.strictEqual(findDuplicate([3, 1, 3, 4, 2]), 3);
  console.log("PASS");
}
