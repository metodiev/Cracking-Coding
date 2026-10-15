/** Task 030 - Reverse digits of an integer. (Tier: Beginner, CTCI Ch 6) */
function reverseDigits(n) {
  const sign = n < 0 ? -1 : 1;
  let value = Math.abs(n);
  let result = 0;
  while (value > 0) {
    result = result * 10 + (value % 10);
    value = Math.floor(value / 10);
  }
  result *= sign;
  return result < -(2 ** 31) || result > 2 ** 31 - 1 ? 0 : result;
}

module.exports = { reverseDigits };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(reverseDigits(123), 321);
  assert.strictEqual(reverseDigits(-123), -321);
  assert.strictEqual(reverseDigits(0), 0);
  console.log("PASS");
}
