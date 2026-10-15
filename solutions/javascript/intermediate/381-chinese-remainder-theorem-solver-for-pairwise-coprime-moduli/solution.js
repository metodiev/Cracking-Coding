/** Task 381 - Chinese remainder theorem solver for pairwise coprime moduli. (Tier: Intermediate, CTCI Ch 6) */
"use strict";

/** Returns [g, x, y] with a*x + b*y == g == gcd(a, b). */
function extendedGcd(a, b) {
  let oldR = a, r = b;
  let oldS = 1n, s = 0n;
  let oldT = 0n, t = 1n;
  while (r !== 0n) {
    const q = oldR / r;
    [oldR, r] = [r, oldR - q * r];
    [oldS, s] = [s, oldS - q * s];
    [oldT, t] = [t, oldT - q * t];
  }
  return [oldR, oldS, oldT];
}

function modInverse(a, modulus) {
  const m = ((a % modulus) + modulus) % modulus;
  const [g, x] = extendedGcd(m, modulus);
  if (g !== 1n) throw new Error("modular inverse does not exist");
  return ((x % modulus) + modulus) % modulus;
}

/** Merge two congruences; throws when the moduli are not coprime. */
function crtTwo(a1, m1, a2, m2) {
  if (extendedGcd(m1, m2)[0] !== 1n)
    throw new Error(`moduli ${m1} and ${m2} are not coprime`);
  const t = (((a2 - a1) % m2) * modInverse(m1, m2)) % m2;
  const modulus = m1 * m2;
  return [(((a1 + m1 * t) % modulus) + modulus) % modulus, modulus];
}

/**
 * Solve a system of congruences with pairwise coprime moduli.
 * Returns [x, m]: x the smallest non-negative solution, m the product of the moduli.
 * Throws for an empty system or non-coprime moduli.
 */
function crt(congruences) {
  if (!congruences.length) throw new Error("empty congruence system");
  let a = 0n, m = 1n;
  for (const [residue, modulus] of congruences) {
    if (modulus <= 0n) throw new Error("modulus must be positive");
    const r = ((residue % modulus) + modulus) % modulus;
    [a, m] = crtTwo(a, m, r, modulus);
  }
  return [a, m];
}

module.exports = { extendedGcd, modInverse, crtTwo, crt };

if (require.main === module) {
  const assert = require("node:assert");

  // classic worked example: 23 = 2 (3), 23 = 3 (5), 23 = 2 (7)
  assert.deepStrictEqual(crt([[2n, 3n], [3n, 5n], [2n, 7n]]), [23n, 105n]);
  const [x, m] = crt([[2n, 3n], [3n, 5n], [2n, 7n]]);
  assert.deepStrictEqual([x % 3n, x % 5n, x % 7n], [2n, 3n, 2n]);
  assert.strictEqual(m, 105n);

  // a single congruence is the identity; merge order must not matter
  assert.deepStrictEqual(crt([[4n, 9n]]), [4n, 9n]);
  assert.deepStrictEqual(crt([[23n, 105n]]), [23n, 105n]);
  assert.deepStrictEqual(crt([[2n, 7n], [3n, 5n], [2n, 3n]]), [23n, 105n]);
  assert.deepStrictEqual(crt([[0n, 2n], [1n, 3n]]), [4n, 6n]);

  // non-coprime moduli are rejected (documented error), even when consistent
  assert.throws(() => crt([[1n, 4n], [2n, 6n]]));
  assert.throws(() => crt([[0n, 2n], [0n, 4n]]));
  assert.throws(() => crt([[1n, 9n], [1n, 3n]]));

  // LCG-driven systems: cross-check by substitution and by brute force where feasible
  let seed = 4212;
  const primes = [3, 5, 7, 11, 13, 17, 19, 23];
  let checked = 0;
  for (let trial = 0; trial < 40; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const count = 2 + (seed % 3);
    const chosen = [];
    while (chosen.length < count) {
      seed = (seed * 75 + 74) % 65537;
      const p = primes[seed % primes.length];
      if (!chosen.includes(p)) chosen.push(p);
    }
    const system = chosen.map((p) => {
      seed = (seed * 75 + 74) % 65537;
      return [BigInt(seed % p), BigInt(p)];
    });
    const [solved, modulus] = crt(system);
    for (const [r, p] of system) assert.strictEqual(solved % p, r, `substitution ${trial}`);
    let product = 1n;
    for (const [, p] of system) product *= p;
    assert.strictEqual(modulus, product, `product ${trial}`);
    assert.ok(solved < modulus, `range ${trial}`);
    if (modulus <= 100000n) {
      let brute = 0n;
      while (brute < modulus) {
        if (system.every(([r, p]) => brute % p === r)) break;
        brute++;
      }
      assert.strictEqual(brute, solved, `brute force ${trial}`);
      checked++;
    }
  }
  assert.ok(checked >= 10, "enough brute-force checks");

  console.log("PASS");
}
