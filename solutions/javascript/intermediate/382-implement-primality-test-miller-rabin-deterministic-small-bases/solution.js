/** Task 382 - Miller-Rabin primality test with deterministic small bases. (Tier: Intermediate, CTCI Ch 6) */
"use strict";

const DETERMINISTIC_BASES = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];

function modPow(base, exponent, modulus) {
  let result = 1n % modulus;
  let b = base % modulus;
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % modulus;
    b = (b * b) % modulus;
    e >>= 1n;
  }
  return result;
}

/** One-base strong test: true when the base fails to prove n composite. */
function isStrongProbablePrime(n, base) {
  if (n < 2n) return false;
  let d = n - 1n;
  let s = 0;
  while ((d & 1n) === 0n) {
    d >>= 1n;
    s++;
  }
  let x = modPow(base, d, n);
  if (x === 1n || x === n - 1n) return true;
  for (let r = 1; r < s; r++) {
    x = (x * x) % n;
    if (x === n - 1n) return true;
  }
  return false;
}

/** Deterministic Miller-Rabin for n < 3.3e24 using the twelve fixed bases. */
function isPrime(n) {
  if (n < 2n) return false;
  for (const p of DETERMINISTIC_BASES) {
    if (n % p === 0n) return n === p;
  }
  for (const a of DETERMINISTIC_BASES) {
    if (!isStrongProbablePrime(n, a)) return false;
  }
  return true;
}

/** Reference oracle for the cross-check; O(sqrt(n)). */
function isPrimeTrialDivision(n) {
  if (n < 2n) return false;
  for (let i = 2n; i * i <= n; i++) {
    if (n % i === 0n) return false;
  }
  return true;
}

module.exports = { DETERMINISTIC_BASES, modPow, isStrongProbablePrime, isPrime, isPrimeTrialDivision };

if (require.main === module) {
  const assert = require("node:assert");

  for (const p of DETERMINISTIC_BASES) assert.ok(isPrime(p), `base is prime ${p}`);

  const knownPrimes = [41n, 97n, 541n, 7919n, 104729n, 1299709n, 2147483647n,
    2305843009213693951n, 67280421310721n];
  for (const p of knownPrimes) assert.ok(isPrime(p), `known prime ${p}`);

  const knownComposites = [0n, 1n, 4n, 9n, 15n, 21n, 25n, 49n, 91n, 121n, 341n, 561n, 1105n,
    1729n, 4294967297n];
  for (const c of knownComposites) assert.ok(!isPrime(c), `known composite ${c}`);
  assert.strictEqual(11n * 31n, 341n);
  assert.strictEqual(3n * 11n * 17n, 561n);
  assert.strictEqual(5n * 13n * 17n, 1105n);
  assert.strictEqual(7n * 13n * 19n, 1729n);
  assert.strictEqual(641n * 6700417n, 4294967297n);

  // 2047 = 23*89: the base-2 test alone would call it prime; extra bases expose it
  assert.strictEqual(23n * 89n, 2047n);
  assert.ok(isStrongProbablePrime(2047n, 2n));
  assert.ok(!isStrongProbablePrime(2047n, 3n));
  assert.ok(!isPrime(2047n));
  assert.ok(!isPrimeTrialDivision(2047n));

  // 3215031751 = 151*751*28351 passes the first four bases but not the whole set
  assert.strictEqual(151n * 751n * 28351n, 3215031751n);
  for (const b of [2n, 3n, 5n, 7n]) {
    assert.ok(isStrongProbablePrime(3215031751n, b), `3215031751 passes base ${b}`);
  }
  assert.ok(!isStrongProbablePrime(3215031751n, 11n));
  assert.ok(!isPrime(3215031751n));

  // cross-check against trial division for every n <= 10000
  let primeCount = 0;
  for (let n = 0n; n <= 10000n; n++) {
    assert.strictEqual(isPrime(n), isPrimeTrialDivision(n), `trial division agrees at ${n}`);
    if (n > 1n && isPrime(n)) primeCount++;
  }
  assert.strictEqual(primeCount, 1229, "pi(10000) = 1229");

  console.log("PASS");
}
