/** Task 032 - Check prime number naive. (Tier: Beginner, CTCI Ch 6) */
function isPrime(n) {
  if (n < 2) return false;
  for (let i = 2; i * i <= n; i++) if (n % i === 0) return false;
  return true;
}

module.exports = { isPrime };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isPrime(0), false);
  assert.strictEqual(isPrime(97), true);
  assert.strictEqual(isPrime(100), false);
  console.log("PASS");
}
