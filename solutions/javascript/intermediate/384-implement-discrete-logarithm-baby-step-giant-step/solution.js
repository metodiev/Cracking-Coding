/** Task 384 - Implement discrete logarithm baby-step giant-step. (Tier: Intermediate, CTCI Ch 6) */
"use strict";

function modPower(base, exponent, modulus) {
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

function extendedGcd(a, b) {
  let oldR = a, r = b;
  let oldS = 1n, s = 0n;
  while (r !== 0n) {
    const q = oldR / r;
    [oldR, r] = [r, oldR - q * r];
    [oldS, s] = [s, oldS - q * s];
  }
  return [oldR, oldS];
}

function modInverse(a, modulus) {
  const [g, x] = extendedGcd(((a % modulus) + modulus) % modulus, modulus);
  if (g !== 1n) throw new Error("modular inverse does not exist");
  return ((x % modulus) + modulus) % modulus;
}

/**
 * Smallest x >= 0 with g^x = h (mod p); -1 when no such x exists.
 * Baby steps list g^j -> j for j in [0, m), m = ceil(sqrt(p)); giant steps walk
 * h * (g^(-m))^i. The first hit with the smallest i and j is the minimal solution.
 */
function discreteLog(g, h, p) {
  if (p <= 1n) throw new Error("modulus must exceed 1");
  const target = ((h % p) + p) % p;
  let m = 1n;
  while (m * m < p) m++;
  const baby = new Map();
  let value = 1n % p;
  for (let j = 0n; j < m; j++) {
    if (!baby.has(value)) baby.set(value, j);
    value = (value * g) % p;
  }
  const giantFactor = modPower(modInverse(g, p), m, p);
  let gamma = target;
  for (let i = 0n; i <= m; i++) {
    if (baby.has(gamma)) return i * m + baby.get(gamma);
    gamma = (gamma * giantFactor) % p;
  }
  return -1n;
}

/** Reference oracle: minimal x in [0, p), or -1; O(p). */
function bruteForceDiscreteLog(g, h, p) {
  const target = ((h % p) + p) % p;
  let value = 1n % p;
  for (let x = 0n; x < p; x++) {
    if (value === target) return x;
    value = (value * g) % p;
  }
  return -1n;
}

module.exports = { modPower, modInverse, discreteLog, bruteForceDiscreteLog };

if (require.main === module) {
  const assert = require("node:assert");

  // hand-verified small problems
  assert.strictEqual(discreteLog(2n, 8n, 13n), 3n);
  assert.strictEqual(modPower(2n, 3n, 13n), 8n);
  assert.strictEqual(discreteLog(3n, 13n, 17n), 4n);   // 3, 9, 27=10, 30=13 mod 17
  assert.strictEqual(modPower(3n, 4n, 17n), 13n);
  assert.strictEqual(discreteLog(2n, 10n, 19n), 17n);
  assert.strictEqual(modPower(2n, 17n, 19n), 10n);
  assert.strictEqual(discreteLog(2n, 5n, 11n), 4n);
  assert.strictEqual(modPower(2n, 4n, 11n), 5n);
  assert.strictEqual(discreteLog(2n, 1n, 13n), 0n);    // g^0 = 1 always

  // documented failure sentinel: no solution when h is outside the subgroup <g>
  assert.strictEqual(discreteLog(2n, 3n, 7n), -1n);    // <2> = {1,2,4} (7) misses 3
  assert.strictEqual(bruteForceDiscreteLog(2n, 3n, 7n), -1n);
  assert.strictEqual(discreteLog(3n, 2n, 11n), -1n);   // <3> = {1,3,4,5,9} (11) misses 2
  assert.strictEqual(bruteForceDiscreteLog(3n, 2n, 11n), -1n);
  assert.strictEqual(discreteLog(1n, 2n, 7n), -1n);    // powers of 1 are all 1
  assert.strictEqual(discreteLog(2n, 0n, 7n), -1n);    // 0 is never a power of g

  // cross-check against brute force for small primes
  const primes = [13n, 17n, 19n, 97n, 101n, 103n, 211n];
  const targets = [1n, 2n, 3n, 5n, 10n, 42n, 123n];
  for (const p of primes) {
    for (let g = 2n; g < (p < 6n ? p : 6n); g++) {
      for (const h of targets) {
        const expected = bruteForceDiscreteLog(g, h, p);
        assert.strictEqual(discreteLog(g, h, p), expected, `cross-check ${g},${h},${p}`);
      }
    }
  }

  // LCG triples: h = g^x is solvable and the answer must satisfy g^answer = h
  let seed = 9871;
  const lcgPrimes = [13n, 17n, 19n, 97n, 101n, 103n, 211n, 1009n];
  for (let trial = 0; trial < 60; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const p = lcgPrimes[seed % lcgPrimes.length];
    seed = (seed * 75 + 74) % 65537;
    const g = 2n + BigInt(seed % Number(p - 2n));
    seed = (seed * 75 + 74) % 65537;
    const x = BigInt(seed % Number(p));
    const h = modPower(g, x, p);
    const found = discreteLog(g, h, p);
    assert.ok(found !== -1n && modPower(g, found, p) === h, `LCG solvable ${trial}`);
    assert.strictEqual(found, bruteForceDiscreteLog(g, h, p), `LCG minimal ${trial}`);
  }

  console.log("PASS");
}
