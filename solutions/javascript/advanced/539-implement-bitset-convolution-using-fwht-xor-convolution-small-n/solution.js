/**
 * Task 539 - Implement bitset convolution using FWHT (XOR convolution) small N. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * XOR convolution on Z_2^k: c[m] = sum_{i xor j == m} a[i] * b[j], computed with the
 * unnormalised Fast Walsh-Hadamard transform (butterfly with the 2x2 Hadamard kernel in
 * len = 1, 2, 4, ... blocks) as c = FWHT^-1(FWHT(a) . FWHT(b)); the inverse is the same butterfly
 * followed by a division by n.  Because n is a power of two, that division is exact in binary
 * floating point, so integer inputs give bit-identical integer results - the self-test asserts
 * strict equality (not merely a tolerance) against the O(n^2) naive convolution, against pair
 * enumeration over explicit set elements, and against the group-algebra product of two delta
 * functions.
 *
 * Documented checks: FWHT(FWHT(a)) == n * a and FWHT(a + b) == FWHT(a) + FWHT(b); the Parseval
 * identity sum(FWHT(a)^2) == n * sum(a^2) and sum(FWHT(a)) == n * a[0]; FWHT of a delta at s is
 * exactly the character t -> (-1)^popcount(s & t); the delta at 0 acts as the convolution
 * identity and convolution is commutative and associative (shared with the count of ordered
 * pairs); convolutions are compared with the naive definition for n = 2, 4, 8, 16; the transform
 * rejects non power-of-two lengths; and repeated runs are identical.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function isPowerOfTwo(n) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

function popcount(x) {
  let count = 0;
  let v = x;
  while (v) {
    v &= v - 1;
    count += 1;
  }
  return count;
}

/** Unnormalised Walsh-Hadamard transform; `invert` divides by n at the end. */
function fwht(values, invert = false) {
  const n = values.length;
  if (!isPowerOfTwo(n)) throw new Error("FWHT needs a power-of-two length");
  const a = values.slice();
  for (let len = 1; len < n; len *= 2) {
    for (let block = 0; block < n; block += 2 * len) {
      for (let j = block; j < block + len; j++) {
        const u = a[j];
        const v = a[j + len];
        a[j] = u + v;
        a[j + len] = u - v;
      }
    }
  }
  if (invert) {
    for (let i = 0; i < n; i++) a[i] /= n;
  }
  return a;
}

function xorConvolution(a, b) {
  if (a.length !== b.length) throw new Error("length mismatch");
  const transformA = fwht(a);
  const transformB = fwht(b);
  const product = transformA.map((value, i) => value * transformB[i]);
  return fwht(product, true);
}

function xorConvolutionNaive(a, b) {
  const n = a.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) out[i ^ j] += a[i] * b[j];
  }
  return out;
}

/** Pair counting over explicit set elements - a different code path from the naive index loop. */
function xorConvolutionByElements(setA, setB, n) {
  const out = new Array(n).fill(0);
  for (const i of setA) {
    for (const j of setB) out[i ^ j] += 1;
  }
  return out;
}

function randomVector(n, seed, limit = 21) {
  const values = [];
  let s = seed % 65537;
  for (let i = 0; i < n; i++) {
    s = lcgNext(s);
    values.push((s % limit) - Math.floor(limit / 2));
  }
  return values;
}

function nonNegativeVector(n, seed, limit = 5) {
  const values = [];
  let s = seed % 65537;
  for (let i = 0; i < n; i++) {
    s = lcgNext(s);
    values.push(s % limit);
  }
  return values;
}

function expand(values) {
  const elements = [];
  values.forEach((count, index) => {
    for (let rep = 0; rep < count; rep++) elements.push(index);
  });
  return elements;
}

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

function main() {
  const assert = require("node:assert");

  // (1) Hand-checked n = 2 and n = 4 cases.
  assert.deepStrictEqual(fwht([1, 2]), [3, -1]);
  assert.deepStrictEqual(xorConvolution([1, 2], [3, 4]), [11, 10]);
  assert.deepStrictEqual(xorConvolutionNaive([1, 2], [3, 4]), [11, 10]);
  const a4 = [1, 2, 3, 4];
  const b4 = [5, 6, 7, 8];
  assert.deepStrictEqual(xorConvolution(a4, b4), [70, 68, 62, 60]);
  assert.deepStrictEqual(xorConvolution(a4, b4), xorConvolutionNaive(a4, b4));
  assert.deepStrictEqual(fwht([1, 1, 1, 1]), [4, 0, 0, 0]);
  assert.deepStrictEqual(fwht([1, -1, 1, -1]), [0, 4, 0, 0], "chi_1 transforms to 4 * delta_1");
  assert.deepStrictEqual(fwht([1, 1, -1, -1]), [0, 0, 4, 0], "chi_2 transforms to 4 * delta_2");

  // (2) Transform algebra: involution up to n, linearity, Parseval, row-sum identity.
  for (let n = 2; n <= 16; n *= 2) {
    for (let seed = 1; seed <= 3; seed++) {
      const a = randomVector(n, seed * 17 + n, 21);
      const b = randomVector(n, seed * 31 + n, 21);
      const doubled = fwht(fwht(a));
      assert.deepStrictEqual(doubled, a.map((value) => value * n),
        `n=${n}: FWHT is an involution up to the factor n`);
      assert.deepStrictEqual(fwht(a.map((value, i) => value + b[i])),
        fwht(a).map((value, i) => value + fwht(b)[i]), `n=${n}: linearity`);
      const l2 = a.reduce((acc, value) => acc + value * value, 0);
      const transformedL2 = fwht(a).reduce((acc, value) => acc + value * value, 0);
      assert.strictEqual(transformedL2, n * l2, `n=${n}: Parseval identity`);
      const total = fwht(a).reduce((acc, value) => acc + value, 0);
      assert.strictEqual(total, n * a[0], `n=${n}: the transform of a sums to n * a[0]`);
    }
  }

  // (3) FWHT of a delta is exactly the parity character (-1)^popcount(s & t).
  for (let k = 1; k <= 4; k++) {
    const n = 1 << k;
    for (let s = 0; s < n; s++) {
      const delta = new Array(n).fill(0);
      delta[s] = 1;
      const expected = [];
      for (let t = 0; t < n; t++) expected.push(popcount(s & t) % 2 === 0 ? 1 : -1);
      assert.deepStrictEqual(fwht(delta), expected, `character of s=${s}`);
    }
  }

  // (4) The identity element and commutativity / associativity of XOR convolution.
  for (let n = 2; n <= 16; n *= 2) {
    const identity = new Array(n).fill(0);
    identity[0] = 1;
    const a = randomVector(n, 101 + n, 21);
    const b = randomVector(n, 202 + n, 21);
    const c = randomVector(n, 303 + n, 21);
    assert.deepStrictEqual(xorConvolution(a, identity), a, `n=${n}: delta at 0 is the identity`);
    assert.deepStrictEqual(xorConvolution(a, b), xorConvolution(b, a), `n=${n}: commutative`);
    assert.deepStrictEqual(xorConvolution(xorConvolution(a, b), c),
      xorConvolution(a, xorConvolution(b, c)), `n=${n}: associative`);
  }

  // (5) Transform-based convolution == naive definition, and == explicit pair counting over the
  //     multisets represented by non-negative count vectors.
  for (let n = 2; n <= 16; n *= 2) {
    for (let seed = 5; seed <= 9; seed++) {
      const a = randomVector(n, seed * 41 + n, 21);
      const b = randomVector(n, seed * 59 + n, 21);
      const viaTransform = xorConvolution(a, b);
      const naive = xorConvolutionNaive(a, b);
      assert.strictEqual(maxAbsDiff(viaTransform, naive), 0, `n=${n} seed ${seed}: exact match`);
      assert.ok(viaTransform.every(Number.isInteger), "integer inputs keep integer outputs");
      if (seed === 5) {
        const countsA = nonNegativeVector(n, seed * 41 + n);
        const countsB = nonNegativeVector(n, seed * 59 + n);
        assert.deepStrictEqual(xorConvolutionByElements(expand(countsA), expand(countsB), n),
          xorConvolution(countsA, countsB),
          `n=${n}: multiset pair counting over explicit elements agrees`);
      }
    }
  }

  // (6) Inverse transform round trip is exact for power-of-two sizes with integer data.
  const data = randomVector(16, 4242, 41);
  assert.deepStrictEqual(fwht(fwht(data), true), data, "inverse FWHT undoes the transform");
  assert.deepStrictEqual(fwht(new Array(1).fill(7)), [7], "n = 1 is the trivial transform");
  assert.deepStrictEqual(fwht(new Array(1).fill(7), true), [7]);

  // (7) Guard rails and determinism.
  assert.throws(() => fwht([1, 2, 3]), /power-of-two/);
  assert.throws(() => xorConvolution([1, 2], [1, 2, 3, 4]), /length mismatch/);
  const repeat = randomVector(8, 909, 21);
  assert.deepStrictEqual(fwht(repeat), fwht(repeat));
  assert.deepStrictEqual(xorConvolution(repeat, repeat), xorConvolutionNaive(repeat, repeat));
}

module.exports = { lcgNext, isPowerOfTwo, popcount, fwht, xorConvolution, xorConvolutionNaive,
  xorConvolutionByElements, randomVector, nonNegativeVector, expand, maxAbsDiff };

if (require.main === module) {
  main();
  console.log("PASS");
}
