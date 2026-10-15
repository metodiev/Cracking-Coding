/** Task 034 - Sieve of Eratosthenes to list primes up to N. (Tier: Beginner, CTCI Ch 6) */
function sieve(n) {
  if (n < 2) return [];
  const isPrime = new Array(n + 1).fill(true);
  isPrime[0] = isPrime[1] = false;
  for (let p = 2; p * p <= n; p++)
    if (isPrime[p])
      for (let m = p * p; m <= n; m += p) isPrime[m] = false;
  const out = [];
  for (let x = 2; x <= n; x++) if (isPrime[x]) out.push(x);
  return out;
}

module.exports = { sieve };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(sieve(1), []);
  assert.deepStrictEqual(sieve(30), [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
  console.log("PASS");
}
