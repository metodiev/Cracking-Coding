/** Task 302 - Convolution via number-theoretic transform (outline). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

const MOD = 998244353n;
const PRIMITIVE_ROOT = 3n;

function power(base, exponent, modulus) {
  let result = 1n;
  base %= modulus;
  let exp = exponent;
  while (exp > 0n) {
    if (exp & 1n) result = (result * base) % modulus;
    base = (base * base) % modulus;
    exp >>= 1n;
  }
  return result;
}

function ntt(values, invert) {
  const n = values.length;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j |= bit;
    if (i < j) {
      [values[i], values[j]] = [values[j], values[i]];
    }
  }

  for (let length = 2; length <= n; length <<= 1) {
    let root = power(PRIMITIVE_ROOT, BigInt((Number(MOD) - 1) / length), MOD);
    if (invert) root = power(root, MOD - 2n, MOD);
    const half = length >> 1;
    for (let start = 0; start < n; start += length) {
      let w = 1n;
      for (let k = 0; k < half; k++) {
        const u = values[start + k];
        const v = (values[start + k + half] * w) % MOD;
        values[start + k] = (u + v) % MOD;
        values[start + k + half] = ((u - v) % MOD + MOD) % MOD;
        w = (w * root) % MOD;
      }
    }
  }

  if (invert) {
    const inverseN = power(BigInt(n), MOD - 2n, MOD);
    for (let i = 0; i < n; i++) values[i] = (values[i] * inverseN) % MOD;
  }
}

function convolutionNtt(a, b) {
  if (!a.length || !b.length) return [];
  let size = 1;
  while (size < a.length + b.length - 1) size <<= 1;
  const left = new Array(size).fill(0n);
  const right = new Array(size).fill(0n);
  for (let i = 0; i < a.length; i++) left[i] = BigInt(a[i]) % MOD;
  for (let i = 0; i < b.length; i++) right[i] = BigInt(b[i]) % MOD;
  ntt(left, false);
  ntt(right, false);
  for (let i = 0; i < size; i++) left[i] = (left[i] * right[i]) % MOD;
  ntt(left, true);
  return left.slice(0, a.length + b.length - 1);
}

function convolutionNaiveMod(a, b) {
  if (!a.length || !b.length) return [];
  const out = new Array(a.length + b.length - 1).fill(0n);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      out[i + j] = (out[i + j] + BigInt(a[i]) * BigInt(b[j])) % MOD;
    }
  }
  return out;
}

function lcgVectors(count, seed = 7788) {
  const vectors = [];
  for (let k = 0; k < count; k++) {
    const size = 1 + (seed % 7);
    const vector = [];
    for (let i = 0; i < size; i++) {
      seed = (seed * 75 + 74) % 65537;
      vector.push(seed % 1000);
    }
    vectors.push(vector);
  }
  return vectors;
}

module.exports = { MOD, PRIMITIVE_ROOT, power, ntt, convolutionNtt, convolutionNaiveMod, lcgVectors };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(power(PRIMITIVE_ROOT, MOD - 1n, MOD), 1n);
  assert.strictEqual((MOD - 1n) % (1n << 23n), 0n);
  assert.strictEqual((MOD - 1n) / (1n << 23n), 119n);

  assert.deepStrictEqual(convolutionNtt([1, 2, 3], [0, 1, 1]), [0n, 1n, 3n, 5n, 3n]);
  assert.deepStrictEqual(convolutionNtt([1, 1], [1, 1]), [1n, 2n, 1n]);
  assert.deepStrictEqual(convolutionNtt([7], [9]), [63n]);
  assert.deepStrictEqual(convolutionNtt([], [1]), []);
  assert.deepStrictEqual(convolutionNtt([MOD - 1n], [2]), [MOD - 2n]);

  const vector = [3n, 1n, 4n, 1n, 5n, 9n, 2n, 6n];
  const copy = vector.slice();
  ntt(copy, false);
  assert.notDeepStrictEqual(copy, vector);
  ntt(copy, true);
  assert.deepStrictEqual(copy, vector);

  const vectors = lcgVectors(40);
  for (let i = 0; i + 1 < vectors.length; i += 2) {
    const a = vectors[i];
    const b = vectors[i + 1];
    assert.deepStrictEqual(convolutionNtt(a, b), convolutionNaiveMod(a, b), `forward ${i}`);
    assert.deepStrictEqual(convolutionNtt(b, a), convolutionNaiveMod(a, b), `swapped ${i}`);
  }
  console.log("PASS");
}
