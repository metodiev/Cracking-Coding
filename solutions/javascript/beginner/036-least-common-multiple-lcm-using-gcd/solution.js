/** Task 036 - Least common multiple (LCM) using GCD. (Tier: Beginner, CTCI Ch 6) */
function gcd(a, b) {
  a = Math.abs(a); b = Math.abs(b);
  while (b) [a, b] = [b, a % b];
  return a;
}

function lcm(a, b) {
  if (a === 0 || b === 0) return 0;
  return Math.abs((a / gcd(a, b)) * b);
}

module.exports = { lcm };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(lcm(4, 6), 12);
  assert.strictEqual(lcm(21, 6), 42);
  assert.strictEqual(lcm(0, 5), 0);
  console.log("PASS");
}
