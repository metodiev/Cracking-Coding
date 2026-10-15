/** Task 037 - Compute power(x, n) iterative. (Tier: Beginner, CTCI Ch 6) */
function power(x, n) {
  const exp = Math.abs(n);
  let result = 1.0;
  for (let i = 0; i < exp; i++) result *= x;
  return n >= 0 ? result : 1.0 / result;
}

module.exports = { power };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(power(2, 10), 1024);
  assert.strictEqual(power(2, -2), 0.25);
  assert.strictEqual(power(5, 0), 1);
  console.log("PASS");
}
