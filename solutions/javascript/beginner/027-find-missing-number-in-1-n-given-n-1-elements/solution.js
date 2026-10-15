/** Task 027 - Find missing number in 1..n given n-1 elements. (Tier: Beginner, CTCI Ch 1) */
function missingNumber(nums, n) {
  const expected = (n * (n + 1)) / 2;
  const actual = nums.reduce((a, b) => a + b, 0);
  return expected - actual;
}

module.exports = { missingNumber };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(missingNumber([1, 2, 4, 5], 5), 3);
  assert.strictEqual(missingNumber([2], 2), 1);
  console.log("PASS");
}
