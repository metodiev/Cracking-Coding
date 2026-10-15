/** Task 035 - Greatest common divisor (GCD) Euclid's algorithm. (Tier: Beginner, CTCI Ch 6) */
function gcd(a, b) {
  a = Math.abs(a); b = Math.abs(b);
  while (b) [a, b] = [b, a % b];
  return a;
}

module.exports = { gcd };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(gcd(48, 18), 6);
  assert.strictEqual(gcd(17, 5), 1);
  assert.strictEqual(gcd(0, 9), 9);
  assert.strictEqual(gcd(-24, 36), 12);
  console.log("PASS");
}
