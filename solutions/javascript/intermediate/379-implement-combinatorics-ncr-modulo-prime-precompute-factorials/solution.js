/**
 * Task 379 - Binomial coefficients C(n, k) mod prime p via precomputed factorials. (Tier: Intermediate, CTCI Ch 6)
 *
 * For a prime p = 1_000_000_007 factorials and inverse factorials are precomputed once:
 * invFact[n] = fact[n]^(p-2) mod p (Fermat), then downward invFact[i] = invFact[i+1]*(i+1).
 * Then C(n, k) = fact[n] * invFact[k] * invFact[n-k] mod p in O(1) per query, valid for
 * 0 <= k <= n < p (Lucas is not needed below p).
 *
 * All modular arithmetic is done in BigInt: products of two residues near 1e9 exceed the
 * exact Number range, so mixing Number literals into the modular expressions would be a
 * correctness bug.  n and k themselves stay as plain Numbers (they are small indices).
 *
 * Hand-verified assertions: C(10,3) = 120, C(20,10) = 184756, C(n,0) = C(n,n) = 1,
 * k > n -> 0 (the convention used everywhere in this repo), k < 0 -> 0.
 *
 * Independent verification paths:
 *   * a multiplicative loop C(n,k) = prod_{i=1..k} (n-k+i) * inv(i) mod p that never
 *     touches the factorial tables;
 *   * additive Pascal rows built by repeated addition mod p (no modular inverses at
 *     all) for n <= 140, cross-checked on LCG (n, k) pairs;
 *   * the known constants C(100,50) mod p = 538992043 and C(1000,500) mod p = 159835829,
 *     each obtained independently (multiplicative loop + exact big-integer reduction)
 *     and asserted here.
 */
"use strict";

const MOD = 1000000007n;

class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/** Fast modular exponentiation (binary), all arguments BigInt. */
function modPow(base, exponent, modulus) {
  let result = 1n % modulus;
  base %= modulus;
  while (exponent > 0n) {
    if ((exponent & 1n) === 1n) result = (result * base) % modulus;
    base = (base * base) % modulus;
    exponent >>= 1n;
  }
  return result;
}

/** C(n, k) mod MOD with precomputed tables. */
class Binomial {
  constructor(limit) {
    this.limit = limit;
    this.fact = new Array(limit + 1).fill(1n);
    for (let i = 1; i <= limit; i++) this.fact[i] = (this.fact[i - 1] * BigInt(i)) % MOD;
    this.invFact = new Array(limit + 1).fill(1n);
    this.invFact[limit] = modPow(this.fact[limit], MOD - 2n, MOD);
    for (let i = limit; i > 0; i--) this.invFact[i - 1] = (this.invFact[i] * BigInt(i)) % MOD;
  }

  choose(n, k) {
    if (k < 0 || n < 0 || k > n) return 0n;
    if (n > this.limit) throw new Error("n above the precomputed limit");
    return (((this.fact[n] * this.invFact[k]) % MOD) * this.invFact[n - k]) % MOD;
  }
}

/** Independent path: running product with per-step modular inverses. */
function chooseMultiplicative(n, k) {
  if (k < 0 || n < 0 || k > n) return 0n;
  if (k > n - k) k = n - k;
  let result = 1n;
  for (let i = 1; i <= k; i++) {
    result = (result * (BigInt(n - k + i) % MOD)) % MOD;
    result = (result * modPow(BigInt(i), MOD - 2n, MOD)) % MOD;
  }
  return result;
}

/** Additive Pascal row mod MOD - no division or inversion used at all. */
function pascalRow(n) {
  let row = [1n];
  for (let step = 0; step < n; step++) {
    const next = new Array(row.length + 1).fill(1n);
    for (let i = 1; i < row.length; i++) next[i] = (row[i - 1] + row[i]) % MOD;
    row = next;
  }
  return row;
}

module.exports = { MOD, modPow, Binomial, chooseMultiplicative, pascalRow };

if (require.main === module) {
  const assert = require("node:assert");

  const table = new Binomial(1000);

  assert.strictEqual(table.choose(10, 3), 120n, "C(10,3) = 120");
  assert.strictEqual(table.choose(20, 10), 184756n, "C(20,10) = 184756");
  assert.ok(table.choose(5, 0) === 1n && table.choose(5, 5) === 1n, "C(n,0) = C(n,n) = 1");
  assert.strictEqual(table.choose(0, 0), 1n, "C(0,0) = 1");
  assert.strictEqual(table.choose(3, 5), 0n, "k > n -> 0");
  assert.strictEqual(table.choose(3, -1), 0n, "k < 0 -> 0");
  assert.strictEqual(table.choose(0, 1), 0n, "C(0,1) = 0");

  // symmetry C(n, k) == C(n, n-k)
  for (const n of [1, 2, 7, 40, 999, 1000]) {
    for (const k of [0, 1, Math.floor(n / 2), n - 1, n]) {
      assert.strictEqual(table.choose(n, k), table.choose(n, n - k), `symmetry ${n} ${k}`);
    }
  }

  // real (non-modular) checks where the exact value is known
  assert.strictEqual(chooseMultiplicative(10, 3), 120n, "multiplicative C(10,3)");
  assert.strictEqual(chooseMultiplicative(20, 10), 184756n, "multiplicative C(20,10)");
  assert.strictEqual(chooseMultiplicative(52, 5), 2598960n, "poker hands C(52,5)");

  // known constants, each verified by the independent multiplicative path
  assert.strictEqual(table.choose(100, 50), 538992043n, "C(100,50) mod p");
  assert.strictEqual(chooseMultiplicative(100, 50), 538992043n, "multiplicative C(100,50)");
  assert.strictEqual(table.choose(1000, 500), 159835829n, "C(1000,500) mod p");
  assert.strictEqual(chooseMultiplicative(1000, 500), 159835829n, "multiplicative C(1000,500)");

  // additive Pascal rows: completely different arithmetic path for smaller n
  const lcg = new Lcg(37913);
  for (let trial = 0; trial < 60; trial++) {
    const n = lcg.next(141);
    const k = lcg.next(n + 2) - 1;              // includes -1, 0, n and n+1
    const expected = (k < 0 || k > n) ? 0n : pascalRow(n)[k];
    assert.strictEqual(table.choose(n, k), expected, `pascal cross-check ${n} ${k}`);
    assert.strictEqual(chooseMultiplicative(n, k), expected, `pascal multiplicative ${n} ${k}`);
  }
  for (const n of [1, 2, 3, 10, 50, 140]) {
    const row = pascalRow(n);
    for (let k = 0; k <= n; k++) {
      assert.strictEqual(table.choose(n, k), row[k], `full pascal row ${n} ${k}`);
      assert.strictEqual(chooseMultiplicative(n, k), row[k], `full pascal multiplicative ${n} ${k}`);
    }
  }

  // Pascal identity C(n, k) = C(n-1, k-1) + C(n-1, k) on the modular tables
  for (let trial = 0; trial < 40; trial++) {
    const n = lcg.next(1001);
    const k = lcg.next(n + 1);
    if (k < 1 || k > n) continue;
    assert.strictEqual(table.choose(n, k),
      (table.choose(n - 1, k - 1) + table.choose(n - 1, k)) % MOD, `pascal identity ${n} ${k}`);
  }

  // binomial theorem sanity: sum_k C(n, k) = 2^n mod p
  for (const n of [5, 30, 200, 1000]) {
    let total = 0n;
    for (let k = 0; k <= n; k++) total = (total + table.choose(n, k)) % MOD;
    assert.strictEqual(total, modPow(2n, BigInt(n), MOD), `binomial theorem ${n}`);
  }
  console.log("PASS");
}
