/** Task 383 - Factor a large integer with Pollard's rho (outline / small tests). (Tier: Intermediate, CTCI Ch 6) */
"use strict";

const MILLER_RABIN_BASES = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];

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

function gcd(a, b) {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y !== 0n) {
    [x, y] = [y, x % y];
  }
  return x;
}

/** Deterministic Miller-Rabin (same twelve-base set as task 382). */
function isPrime(n) {
  if (n < 2n) return false;
  for (const p of MILLER_RABIN_BASES) {
    if (n % p === 0n) return n === p;
  }
  let d = n - 1n;
  let s = 0;
  while ((d & 1n) === 0n) {
    d >>= 1n;
    s++;
  }
  for (const a of MILLER_RABIN_BASES) {
    let x = modPow(a, d, n);
    if (x === 1n || x === n - 1n) continue;
    let witnessed = true;
    for (let r = 1; r < s; r++) {
      x = (x * x) % n;
      if (x === n - 1n) { witnessed = false; break; }
    }
    if (witnessed) return false;
  }
  return true;
}

function f(x, c, n) {
  return ((x * x) % n + c) % n;
}

/** Floyd cycle-finding variant; retries with the next c when the gcd is n. */
function pollardRho(n) {
  if (n % 2n === 0n) return 2n;
  if (n % 3n === 0n) return 3n;
  for (let c = 1n; c < 100n; c++) {
    let x = 2n, y = 2n, d = 1n;
    while (d === 1n) {
      x = f(x, c, n);
      y = f(f(y, c, n), c, n);
      d = gcd(x - y, n);
    }
    if (d !== n) return d;
  }
  throw new Error("Pollard's rho failed to find a factor");
}

function split(value, out) {
  if (value === 1n) return;
  if (isPrime(value)) {
    out.push(value);
    return;
  }
  const factor = pollardRho(value);
  split(factor, out);
  split(value / factor, out);
}

/** Complete prime factorisation in ascending order ([] for 1). */
function factorize(n) {
  const out = [];
  split(n, out);
  out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  return out;
}

module.exports = { modPow, gcd, isPrime, pollardRho, factorize };

if (require.main === module) {
  const assert = require("node:assert");

  // hand-checked factorisations
  assert.deepStrictEqual(factorize(8051n), [83n, 97n]);
  assert.deepStrictEqual(factorize(10403n), [101n, 103n]);
  assert.deepStrictEqual(factorize(4294967297n), [641n, 6700417n]);
  assert.deepStrictEqual(factorize(123456789n), [3n, 3n, 3607n, 3803n]);
  assert.deepStrictEqual(factorize(600851475143n), [71n, 839n, 1471n, 6857n]);
  assert.deepStrictEqual(factorize(1099511627777n), [257n, 4278255361n]);
  assert.strictEqual(83n * 97n, 8051n);
  assert.strictEqual(101n * 103n, 10403n);
  assert.strictEqual(641n * 6700417n, 4294967297n);
  assert.strictEqual(9n * 13717421n, 123456789n);
  assert.strictEqual(71n * 839n * 1471n * 6857n, 600851475143n);
  assert.strictEqual(257n * 4278255361n, 1099511627777n);

  // primes and 1 are handled without recursion
  assert.deepStrictEqual(factorize(1n), []);
  assert.deepStrictEqual(factorize(2n), [2n]);
  assert.deepStrictEqual(factorize(97n), [97n]);
  assert.deepStrictEqual(factorize(97n * 97n), [97n, 97n]);

  // cross-check: product of factors == n, every factor prime, list ascending
  const samples = [8051n, 10403n, 4294967297n, 123456789n, 600851475143n, 1099511627777n,
    999983n, 1n, 4n, 51n, 65536n, 1048577n];
  for (const n of samples) {
    const factors = factorize(n);
    let product = 1n;
    for (const factor of factors) {
      assert.ok(isPrime(factor), `factor prime ${n}/${factor}`);
      product *= factor;
    }
    assert.strictEqual(product, n, `product ${n}`);
    for (let i = 1; i < factors.length; i++) {
      assert.ok(factors[i - 1] <= factors[i], `ascending ${n}`);
    }
  }
  assert.deepStrictEqual(factorize(999983n), [999983n]);
  const powers = factorize(65536n);
  assert.strictEqual(powers.length, 16);
  for (const p of powers) assert.strictEqual(p, 2n);
  assert.deepStrictEqual(factorize(1048577n), [17n, 61681n]);

  console.log("PASS");
}
