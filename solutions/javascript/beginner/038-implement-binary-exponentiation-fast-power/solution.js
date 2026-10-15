/** Task 038 - Implement binary exponentiation (fast power). (Tier: Beginner, CTCI Ch 6) */
function powerFast(base, exp) {
  let result = 1n;
  let b = BigInt(base);
  let e = BigInt(exp);
  while (e > 0n) {
    if (e & 1n) result *= b;
    b *= b;
    e >>= 1n;
  }
  return result;
}

module.exports = { powerFast };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(powerFast(3, 13), 1594323n);
  assert.strictEqual(powerFast(2, 10), 1024n);
  assert.strictEqual(powerFast(7, 0), 1n);
  console.log("PASS");
}
