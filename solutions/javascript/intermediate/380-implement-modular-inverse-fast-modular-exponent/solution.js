/**
 * Task 380 - Modular inverse (extended Euclid and Fermat) + fast modular exponentiation. (Tier: Intermediate, CTCI Ch 6)
 *
 * Two independent inverse implementations:
 *   * extended Euclid: g = gcd(a, m) and Bezout coefficients x, y with a*x + m*y = g;
 *     when g == 1, x mod m is the inverse.  Works for ANY modulus m;
 *   * Fermat: for PRIME p and a not divisible by p, inv = a^(p-2) mod p.
 * Both must agree on every coprime pair, and a * inv(a, m) % m == 1 on LCG values.
 *
 * Documented failure: when gcd(a, m) != 1 no inverse exists; the API returns null (the
 * documented sentinel) instead of silently returning garbage.  The m = 1 case is
 * documented as out of scope (every residue is 0 mod 1) and rejected.
 *
 * All modular arithmetic is BigInt: products of two residues near 1e9 would exceed the
 * exact Number range, so mixing Number literals into the modular expressions would be a
 * correctness bug (BigInt-only expressions below).
 *
 * Fast modular exponentiation is binary (square-and-multiply) with the usual contract
 * powMod(a, 0, m) = 1 mod m; hand-checked: 2^10 = 1024, 3^0 = 1, cross-checked against a
 * naive O(e) loop for small exponents, checked against the identity
 * pow(a, x+y) = pow(a, x) * pow(a, y) mod m, and Fermat's little theorem a^(p-1) = 1
 * mod p is asserted for several primes.
 */
"use strict";

class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/** Fast binary modular exponentiation; exponents >= 0 only (documented). */
function powMod(base, exponent, modulus) {
  if (exponent < 0n) throw new Error("negative exponents are not supported");
  if (modulus <= 0n) throw new Error("modulus must be positive");
  let result = 1n % modulus;
  base = ((base % modulus) + modulus) % modulus;
  while (exponent > 0n) {
    if ((exponent & 1n) === 1n) result = (result * base) % modulus;
    base = (base * base) % modulus;
    exponent >>= 1n;
  }
  return result;
}

function gcd(a, b) {
  while (b !== 0n) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a < 0n ? -a : a;
}

/** Iterative extended Euclid: returns [g, x, y] with a*x + b*y = g >= 0. */
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
  if (oldR < 0n) return [-oldR, -oldS, -oldT];
  return [oldR, oldS, oldT];
}

/** Inverse of a modulo m via extended Euclid, or null when gcd(a, m) != 1. */
function modInverseEuclid(a, m) {
  if (m <= 1n) throw new Error("m >= 2 required (m = 1 is out of scope)");
  const value = ((a % m) + m) % m;
  const [g, x] = extendedGcd(value, m);
  if (g !== 1n) return null;
  return ((x % m) + m) % m;
}

/** Inverse of a modulo the PRIME p via a^(p-2) mod p; null when p divides a. */
function modInverseFermat(a, p) {
  if (p <= 1n) throw new Error("p > 1 required");
  const value = ((a % p) + p) % p;
  if (value === 0n) return null;
  return powMod(value, p - 2n, p);
}

/** Independent O(exponent) oracle for small exponents. */
function powNaive(base, exponent, modulus) {
  let result = 1n % modulus;
  for (let i = 0n; i < exponent; i++) result = (result * (base % modulus)) % modulus;
  return result;
}

module.exports = { powMod, gcd, extendedGcd, modInverseEuclid, modInverseFermat, powNaive };

if (require.main === module) {
  const assert = require("node:assert");

  // --- hand-checked small cases -----------------------------------------
  assert.strictEqual(modInverseEuclid(3n, 11n), 4n, "inv(3, 11) = 4");
  assert.strictEqual(modInverseEuclid(1n, 11n), 1n, "inv(1, 11) = 1");
  assert.strictEqual(modInverseEuclid(1n, 7n), 1n, "inv(1, 7) = 1");
  assert.strictEqual(modInverseEuclid(1n, 1000000000n), 1n, "inv(1, 1e9) = 1");
  assert.strictEqual(modInverseEuclid(2n, 5n), 3n, "inv(2, 5) = 3");
  assert.strictEqual(modInverseEuclid(10n, 17n), 12n, "inv(10, 17) = 12");
  assert.strictEqual(modInverseEuclid(4n, 8n), null, "gcd(4, 8) != 1 -> null");
  assert.strictEqual(modInverseEuclid(6n, 9n), null, "gcd(6, 9) != 1 -> null");
  assert.strictEqual(modInverseEuclid(0n, 5n), null, "gcd(0, 5) != 1 -> null");

  assert.strictEqual(modInverseFermat(3n, 11n), 4n, "fermat inv(3, 11) = 4");
  assert.strictEqual(modInverseFermat(5n, 101n), 81n, "fermat inv(5, 101) = 81");
  assert.strictEqual(modInverseFermat(11n, 11n), null, "p divides a -> null");

  // --- fast exponentiation, hand-checked ---------------------------------
  assert.strictEqual(powMod(2n, 10n, 1000000007n), 1024n, "2^10 = 1024");
  assert.strictEqual(powMod(3n, 0n, 1000000007n), 1n, "3^0 = 1");
  assert.strictEqual(powMod(0n, 0n, 7n), 1n, "0^0 = 1 mod 7");
  assert.strictEqual(powMod(7n, 1n, 5n), 2n, "7 mod 5");
  assert.strictEqual(powMod(5n, 3n, 13n), 8n, "5^3 mod 13");
  assert.strictEqual(powMod(2n, 1000n, 1000000007n), 688423210n, "2^1000 mod p (recomputed)");
  assert.strictEqual(powMod(3n, 200n, 1000000007n), 136318165n, "3^200 mod p (recomputed)");

  // binary vs naive loop for small exponents
  const lcg = new Lcg(38013);
  for (let trial = 0; trial < 60; trial++) {
    const base = BigInt(lcg.next(1000));
    const exponent = BigInt(lcg.next(40));
    const modulus = BigInt(lcg.next(999) + 2);
    assert.strictEqual(powMod(base, exponent, modulus), powNaive(base, exponent, modulus),
      `binary vs naive ${trial}`);
  }

  // pow(a, x + y) = pow(a, x) * pow(a, y) mod m
  for (let trial = 0; trial < 40; trial++) {
    const base = BigInt(lcg.next(1000000006) + 1);
    const x = BigInt(lcg.next(60));
    const y = BigInt(lcg.next(60));
    const modulus = 1000000007n;
    assert.strictEqual(powMod(base, x + y, modulus), (powMod(base, x, modulus) * powMod(base, y, modulus)) % modulus,
      `pow additive identity ${trial}`);
  }

  // --- extended Euclid identity a*inv = 1 (mod m) on LCG values ----------
  for (let trial = 0; trial < 60; trial++) {
    const m = BigInt(lcg.next(998) + 2);
    const a = BigInt(lcg.next(1000));
    const inverse = modInverseEuclid(a, m);
    if (gcd(a, m) === 1n) {
      assert.notStrictEqual(inverse, null, `coprime pair has inverse ${a} ${m}`);
      assert.strictEqual((a * inverse) % m, 1n, `a * inv = 1 mod m ${a} ${m}`);
    } else {
      assert.strictEqual(inverse, null, `non-coprime pair has no inverse ${a} ${m}`);
    }
  }

  // --- Euclid and Fermat agree for primes --------------------------------
  const primes = [7n, 11n, 13n, 101n, 65537n, 1000000007n];
  for (const prime of primes) {
    for (let trial = 0; trial < 12; trial++) {
      const a = BigInt(lcg.next(Number(prime - 1n < 65536n ? prime - 1n : 65536n)) + 1);
      if (a % prime === 0n) continue;
      const euclid = modInverseEuclid(a, prime);
      const fermat = modInverseFermat(a, prime);
      assert.ok(euclid !== null && euclid === fermat, `euclid == fermat ${a} ${prime}`);
      assert.strictEqual((a * euclid) % prime, 1n, `inverse verifies ${a} ${prime}`);
    }
  }

  // Fermat's little theorem: a^(p-1) = 1 mod p for p prime, p not dividing a
  for (const prime of primes) {
    for (const a of [1n, 2n, 3n, prime - 1n, prime - 2n]) {
      if (a % prime === 0n) continue;
      assert.strictEqual(powMod(a, prime - 1n, prime), 1n, `fermat little theorem ${a} ${prime}`);
    }
  }

  // Wilson-flavoured sanity for a small prime: (19-1)! = -1 mod 19
  let factorial = 1n;
  for (let i = 1n; i <= 18n; i++) factorial = (factorial * i) % 19n;
  assert.strictEqual(factorial, 18n, "Wilson 18! = -1 mod 19");
  console.log("PASS");
}
