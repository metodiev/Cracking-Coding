/** Task 300 - Implement convolution (naive) for polynomials. (Tier: Intermediate, CTCI Beyond — Numerical & Scientific Computing) */
"use strict";

function convolve(a, b) {
  if (a.length === 0 || b.length === 0) return [];
  const result = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      result[i + j] += a[i] * b[j];
    }
  }
  return result;
}

function evaluate(polynomial, x) {
  let value = 0;
  for (let i = polynomial.length - 1; i >= 0; i--) value = value * x + polynomial[i];
  return value;
}

function lcgVector(size, seed) {
  const values = [];
  for (let i = 0; i < size; i++) {
    seed = (seed * 75 + 74) % 65537;
    values.push(seed % 10);
  }
  return [values, seed];
}

module.exports = { convolve, evaluate, lcgVector };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(convolve([1, 2, 3], [0, 1, 1]), [0, 1, 3, 5, 3]);
  assert.deepStrictEqual(convolve([5], [7]), [35]);
  assert.deepStrictEqual(convolve([5], [0]), [0]);
  assert.deepStrictEqual(convolve([], [1, 2]), []);
  assert.deepStrictEqual(convolve([1, 2], []), []);

  const a = [4, 1, 2, 3];
  const b = [2, 0, 5];
  const c = [1, 1, 0, 2];
  assert.deepStrictEqual(convolve(a, b), convolve(b, a));
  assert.deepStrictEqual(convolve(convolve(a, b), c), convolve(a, convolve(b, c)));

  assert.deepStrictEqual(convolve(a, [1]), a);
  assert.deepStrictEqual(convolve([1], a), a);
  assert.deepStrictEqual(convolve([0, 0, 1], [1, 2, 3]), [0, 0, 1, 2, 3]);
  assert.deepStrictEqual(convolve([1, 0, 0], [3, 4]), [3, 4, 0, 0]);

  let seed = 70707;
  for (let trial = 0; trial < 120; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 1 + (seed % 6);
    seed = (seed * 75 + 74) % 65537;
    const m = 1 + (seed % 6);
    let left;
    let right;
    [left, seed] = lcgVector(n, seed);
    [right, seed] = lcgVector(m, seed);
    const product = convolve(left, right);
    assert.strictEqual(product.length, n + m - 1, `length ${trial}`);
    assert.deepStrictEqual(product, convolve(right, left), `commutativity ${trial}`);
    for (const x of [2, 3, 5]) {
      assert.strictEqual(
        evaluate(product, x),
        evaluate(left, x) * evaluate(right, x),
        `evaluation ${trial}`
      );
    }
  }
  console.log("PASS");
}
