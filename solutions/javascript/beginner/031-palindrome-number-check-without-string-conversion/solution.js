/** Task 031 - Palindrome number check (without string conversion). (Tier: Beginner, CTCI Ch 6) */
function isPalindromeNumber(n) {
  if (n < 0) return false;
  const original = n;
  let reversed = 0;
  while (n > 0) {
    reversed = reversed * 10 + (n % 10);
    n = Math.floor(n / 10);
  }
  return original === reversed;
}

module.exports = { isPalindromeNumber };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isPalindromeNumber(121), true);
  assert.strictEqual(isPalindromeNumber(-121), false);
  assert.strictEqual(isPalindromeNumber(123), false);
  console.log("PASS");
}
