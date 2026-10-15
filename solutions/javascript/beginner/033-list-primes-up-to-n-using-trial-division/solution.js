/** Task 033 - List primes up to N using trial division. (Tier: Beginner, CTCI Ch 6) */
function primesUpto(n) {
  const isPrime = x => {
    if (x < 2) return false;
    for (let i = 2; i * i <= x; i++) if (x % i === 0) return false;
    return true;
  };
  const out = [];
  for (let x = 2; x <= n; x++) if (isPrime(x)) out.push(x);
  return out;
}

module.exports = { primesUpto };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(primesUpto(1), []);
  assert.deepStrictEqual(primesUpto(10), [2, 3, 5, 7]);
  assert.deepStrictEqual(primesUpto(20), [2, 3, 5, 7, 11, 13, 17, 19]);
  console.log("PASS");
}
