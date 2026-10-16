/**
 * Task 559 - Implement fast exponentiation of linear recurrences using matrix power. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A linear recurrence a_n = c_1 a_{n-1} + ... + c_k a_{n-k} is an entry of a matrix power: the state
 * vector (a_{k-1}, ..., a_0) is advanced by the companion matrix M (coefficients in the first row, ones
 * on the subdiagonal) to (a_k, ..., a_1) and in general M^n s = (a_{n+k-1}, ..., a_n), so a_n is the
 * bottom component, and one binary exponentiation gives a_n in O(k^3 log n) modular
 * operations rather than the O(k n) of the naive iteration.  The file implements three independent
 * routes to the same number and asserts they agree: naive iteration (the definition), companion-matrix
 * exponentiation with an operation counter, and Kitamasa's polynomial exponentiation x^n mod the
 * characteristic polynomial x^k - c_1 x^(k-1) - ... - c_k, which is O(k^2 log n) and therefore the one
 * that scales.  Exact BigInt arithmetic is used whenever the modulus is null, so the modular and the
 * exact runs can be compared directly, and the modulus is the Mersenne prime 2^61 - 1.
 *
 * Documented checks: Fibonacci agrees with an independent BigInt iteration up to n = 300, with known
 * values F(50) and F(100), with Binet's closed form to a relative tolerance of 1e-9 for n <= 70 (where
 * the floating-point formula is still meaningful) and with Cassini's identity
 * a_{n+1} a_{n-1} - a_n^2 = (-1)^n; Tribonacci and twelve LCG recurrences of orders 3 and 5 are
 * checked for n up to 120 against the naive iteration; modular and exact runs agree modulo 2^61 - 1;
 * a_1000000 is computed by both fast routes and, for Fibonacci, against an independent fast-doubling
 * recursion in log n BigInt multiplications; the characteristic polynomial is computed symbolically as
 * det(xI - M) by cofactor expansion over polynomial coefficients, asserted to be monic with coefficients
 * -c_i, cross-checked against det(tI - M) evaluated at integer points, and used to verify
 * Cayley-Hamilton (the polynomial annihilates M); the reduction rules are pinned by hand
 * (x^2 = x + 1, x^3 = 2x + 1 for Fibonacci, x^0 = 1, x^1 = x); index boundaries n = 0, 1, k-1 are the
 * initial conditions, k = 1 is the geometric case with the closed form c^n, and all-zero coefficients
 * give the zero sequence; the operation counters show exponentiation needs O(log n) matrix products
 * while Kitamasa needs O(log n) polynomial products, both well below the n - 1 products of repeated
 * multiplication; and reruns are bit-identical.
 */
"use strict";

const MOD = 2305843009213693951n;
const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

function lcg64Int(state, bound) {
  return Number((state >> 13n) % BigInt(bound));
}

function mod(value, modulus) {
  const reduced = value % modulus;
  return reduced < 0n ? reduced + modulus : reduced;
}

/** Reduce only when a modulus is given; with null the BigInt arithmetic stays exact. */
function reduce(value, modulus) {
  return modulus === null ? value : mod(value, modulus);
}

/** Naive iteration a_n = sum c_i a_{n-i}; returns the whole prefix [a_0 .. a_n]. */
function naiveSequence(coeffs, initial, n, modulus = null) {
  const k = coeffs.length;
  if (k === 0) return [0n];
  if (initial.length < k) throw new Error("too few initial values");
  const values = initial.slice(0, k).map((value) => BigInt(value));
  for (let index = k; index <= n; index++) {
    let total = 0n;
    for (let i = 0; i < k; i++) total += BigInt(coeffs[i]) * values[index - 1 - i];
    values.push(reduce(total, modulus));
  }
  return values.slice(0, n + 1);
}

function zeros(n, m) {
  return Array.from({ length: n }, () => new Array(m).fill(0n));
}

function matMul(a, b, modulus) {
  const n = a.length;
  const out = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < n; l++) {
      if (a[i][l] === 0n) continue;
      for (let j = 0; j < n; j++) {
        if (b[l][j] === 0n) continue;
        out[i][j] = reduce(out[i][j] + a[i][l] * b[l][j], modulus);
      }
    }
  }
  return out;
}

function matIdentity(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1n : 0n)));
}

function matPower(matrix, exponent, modulus, counter = null) {
  let base = matrix;
  let result = matIdentity(matrix.length);
  let e = BigInt(exponent);
  while (e > 0n) {
    if (e & 1n) {
      if (counter) counter.multiplications += 1;
      result = matMul(result, base, modulus);
    }
    e >>= 1n;
    if (e > 0n) {
      if (counter) counter.multiplications += 1;
      base = matMul(base, base, modulus);
    }
  }
  return result;
}

/** Companion matrix of the recurrence, coefficients in the first row, ones on the subdiagonal. */
function companionMatrix(coeffs, modulus = MOD) {
  const k = coeffs.length;
  const matrix = zeros(k, k);
  for (let i = 1; i < k; i++) matrix[i][i - 1] = 1n;
  for (let j = 0; j < k; j++) matrix[0][j] = reduce(BigInt(coeffs[j]), modulus);
  return matrix;
}

/**
 * a_n = last component of M^n applied to the state s = (a_{k-1}, ..., a_0): the first row of M answers
 * c_1 s_1 + ... + c_k s_k = a_k, the subdiagonal shifts, so M^n s = (a_{n+k-1}, ..., a_n) and a_n is
 * the bottom entry.
 */
function termViaCompanion(coeffs, initial, n, modulus, counter = null) {
  const k = coeffs.length;
  if (k === 0) return 0n;
  const power = matPower(companionMatrix(coeffs, modulus), n, modulus, counter);
  let total = 0n;
  for (let j = 0; j < k; j++) total += power[k - 1][j] * BigInt(initial[k - 1 - j]);
  return reduce(total, modulus);
}

/** Polynomial product modulo x^k - c_1 x^(k-1) - ... - c_k (coefficients lowest degree first). */
function polyMultiply(a, b, coeffs, modulus) {
  const k = coeffs.length;
  const product = new Array(2 * k - 1).fill(0n);
  for (let i = 0; i < a.length; i++) {
    if (a[i] === 0n) continue;
    for (let j = 0; j < b.length; j++) {
      if (b[j] === 0n) continue;
      product[i + j] = reduce(product[i + j] + a[i] * b[j], modulus);
    }
  }
  for (let m = product.length - 1; m >= k; m--) {
    const value = product[m];
    if (value === 0n) continue;
    product[m] = 0n;
    for (let i = 0; i < k; i++) {
      product[m - 1 - i] = reduce(product[m - 1 - i] + value * BigInt(coeffs[i]), modulus);
    }
  }
  return product.slice(0, k);
}

/** x^n modulo the characteristic polynomial, by binary exponentiation over polynomials. */
function polyPower(coeffs, n, modulus, counter = null) {
  const k = coeffs.length;
  let result = new Array(k).fill(0n);
  result[0] = 1n;
  const base = new Array(k).fill(0n);
  if (k === 1) base[0] = reduce(BigInt(coeffs[0]), modulus);
  else base[1] = 1n; // the polynomial x itself
  let e = BigInt(n);
  let accumulator = base;
  while (e > 0n) {
    if (e & 1n) {
      if (counter) counter.multiplications += 1;
      result = polyMultiply(result, accumulator, coeffs, modulus);
    }
    e >>= 1n;
    if (e > 0n) {
      if (counter) counter.multiplications += 1;
      accumulator = polyMultiply(accumulator, accumulator, coeffs, modulus);
    }
  }
  return result;
}

/** Kitamasa: a_n = sum_j (x^n mod char.poly)_j * a_j. */
function termViaKitamasa(coeffs, initial, n, modulus, counter = null) {
  const k = coeffs.length;
  if (k === 0) return 0n;
  const reduced = polyPower(coeffs, n, modulus, counter);
  let total = 0n;
  for (let j = 0; j < k; j++) total += reduced[j] * BigInt(initial[j]);
  return reduce(total, modulus);
}

/* ---------------------------------------------------------------- polynomial arithmetic ---- */

function polyAdd(a, b) {
  const length = Math.max(a.length, b.length);
  const out = new Array(length).fill(0n);
  for (let i = 0; i < length; i++) out[i] = (a[i] ?? 0n) + (b[i] ?? 0n);
  return out;
}

function polySub(a, b) {
  const length = Math.max(a.length, b.length);
  const out = new Array(length).fill(0n);
  for (let i = 0; i < length; i++) out[i] = (a[i] ?? 0n) - (b[i] ?? 0n);
  return out;
}

function polyTimes(a, b) {
  const out = new Array(a.length + b.length - 1).fill(0n);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) out[i + j] += a[i] * b[j];
  }
  return out;
}

function polyIsZero(a) {
  return a.every((value) => value === 0n);
}

/** Symbolic determinant by cofactor expansion over Z[x]; entries are coefficient arrays. */
function polyDeterminant(matrix) {
  const n = matrix.length;
  if (n === 0) return [1n];
  if (n === 1) return matrix[0][0];
  if (n === 2) {
    return polySub(polyTimes(matrix[0][0], matrix[1][1]),
      polyTimes(matrix[0][1], matrix[1][0]));
  }
  let total = [0n];
  for (let j = 0; j < n; j++) {
    if (polyIsZero(matrix[0][j])) continue;
    const minor = matrix.slice(1).map((row) => row.filter((_, c) => c !== j));
    const term = polyTimes(matrix[0][j], polyDeterminant(minor));
    total = j % 2 === 0 ? polyAdd(total, term) : polySub(total, term);
  }
  return total;
}

/** det(xI - A) with the highest-degree coefficient first: [1, -c_1, ..., -c_k]. */
function characteristicPolynomial(matrix) {
  const n = matrix.length;
  const symbolic = matrix.map((row, i) =>
    row.map((value, j) => (i === j ? [-BigInt(value), 1n] : [-BigInt(value)])));
  return polyDeterminant(symbolic).reverse();
}

/** Numeric det(tI - A) at an integer t, by cofactor expansion (the interpolation cross-check). */
function numericDeterminant(matrix) {
  const n = matrix.length;
  if (n === 0) return 1n;
  if (n === 1) return matrix[0][0];
  let total = 0n;
  for (let j = 0; j < n; j++) {
    if (matrix[0][j] === 0n) continue;
    const minor = matrix.slice(1).map((row) => row.filter((_, c) => c !== j));
    const term = matrix[0][j] * numericDeterminant(minor);
    total = j % 2 === 0 ? total + term : total - term;
  }
  return total;
}

function evaluatePolynomial(coefficients, t) {
  let value = 0n;
  for (const coefficient of coefficients) value = value * BigInt(t) + coefficient;
  return value;
}

function randomRecurrence(k, seedState, maxCoeff = 9) {
  const coeffs = [];
  let state = seedState % TWO_64;
  for (let i = 0; i < k; i++) {
    state = lcg64Next(state);
    coeffs.push(1 + lcg64Int(state, maxCoeff));
  }
  const initial = [];
  for (let i = 0; i < k; i++) {
    state = lcg64Next(state);
    initial.push(lcg64Int(state, 100));
  }
  return { coeffs, initial };
}

/** Fast doubling for Fibonacci, the third independent route used at n = 10^6. */
function fibDoubling(n) {
  const rec = (index) => {
    if (index === 0) return [0n, 1n];
    const [a, b] = rec(Math.floor(index / 2));
    const c = mod(a * (2n * b - a), MOD);
    const d = mod(a * a + b * b, MOD);
    return index % 2 === 0 ? [c, d] : [d, mod(c + d, MOD)];
  };
  return rec(n);
}

function main() {
  const assert = require("node:assert");

  // (1) Fibonacci against the independent BigInt iteration, known values and both fast routes.
  const fib = [1, 1];
  const exactFib = naiveSequence(fib, [0, 1], 300);
  for (const n of [0, 1, 2, 3, 5, 10, 50, 100, 199, 300]) {
    assert.strictEqual(termViaCompanion(fib, [0, 1], n, null), exactFib[n],
      `Fibonacci a_${n} by companion matrix`);
    assert.strictEqual(termViaKitamasa(fib, [0, 1], n, null), exactFib[n],
      `Fibonacci a_${n} by Kitamasa`);
  }
  assert.strictEqual(exactFib[10], 55n);
  assert.strictEqual(exactFib[20], 6765n);
  assert.strictEqual(exactFib[30], 832040n);
  assert.strictEqual(exactFib[50], 12586269025n, "a known Fibonacci value");
  assert.strictEqual(exactFib[100], 354224848179261915075n, "F(100)");

  // (2) Binet's closed form and Cassini's identity.
  const phi = (1 + Math.sqrt(5)) / 2;
  const psi = (1 - Math.sqrt(5)) / 2;
  for (let n = 0; n <= 70; n++) {
    const binet = (Math.pow(phi, n) - Math.pow(psi, n)) / Math.sqrt(5);
    const exact = Number(exactFib[n]);
    assert.ok(Math.abs(binet - exact) <= 1e-9 * Math.max(1, exact), `Binet at n = ${n}`);
  }
  for (let n = 1; n <= 30; n++) {
    assert.strictEqual(exactFib[n + 1] * exactFib[n - 1] - exactFib[n] * exactFib[n],
      n % 2 === 0 ? 1n : -1n, `Cassini at n = ${n}`);
  }

  // (3) Tribonacci and twelve LCG recurrences against the naive iteration.
  const tribCoeffs = [1, 1, 1];
  const trib = naiveSequence(tribCoeffs, [0, 0, 1], 200);
  assert.strictEqual(trib[4], 2n, "A000073: 0, 0, 1, 1, 2, 4, 7, 13, 24, 44, ...");
  assert.strictEqual(trib[10], 81n);
  assert.strictEqual(trib[11], 149n, "the Tribonacci numbers");
  for (const n of [0, 1, 3, 7, 25, 99, 200]) {
    assert.strictEqual(termViaCompanion(tribCoeffs, [0, 0, 1], n, null), trib[n]);
    assert.strictEqual(termViaKitamasa(tribCoeffs, [0, 0, 1], n, null), trib[n]);
  }
  for (let seed = 1; seed <= 6; seed++) {
    for (const k of [3, 5]) {
      const { coeffs, initial } = randomRecurrence(k, BigInt(seed) * 7919n + BigInt(k));
      const sequence = naiveSequence(coeffs, initial, 120);
      for (const n of [0, 1, k - 1, k, k + 1, 20, 57, 120]) {
        assert.strictEqual(termViaCompanion(coeffs, initial, n, null), sequence[n],
          `seed ${seed}, k = ${k}, n = ${n}: companion route`);
        assert.strictEqual(termViaKitamasa(coeffs, initial, n, null), sequence[n],
          `seed ${seed}, k = ${k}, n = ${n}: Kitamasa route`);
      }
    }
  }

  // (4) Modular runs agree with the exact runs, and a_1000000 needs no iteration at all.
  for (const k of [2, 3, 4, 6]) {
    const { coeffs, initial } = randomRecurrence(k, BigInt(k) * 104729n);
    for (const n of [0, 1, 5, 40]) {
      assert.strictEqual(termViaCompanion(coeffs, initial, n, MOD),
        mod(termViaCompanion(coeffs, initial, n, null), MOD), `k = ${k}, n = ${n}: companion mod p`);
      assert.strictEqual(termViaKitamasa(coeffs, initial, n, MOD),
        mod(termViaKitamasa(coeffs, initial, n, null), MOD), `k = ${k}, n = ${n}: Kitamasa mod p`);
    }
    const huge = 1000000;
    assert.strictEqual(termViaKitamasa(coeffs, initial, huge, MOD),
      termViaCompanion(coeffs, initial, huge, MOD), `k = ${k}: a_1000000 by both fast routes`);
  }
  assert.strictEqual(termViaCompanion(fib, [0, 1], 1000000, MOD), fibDoubling(1000000)[0],
    "F(10^6) by matrix power vs fast doubling");
  assert.strictEqual(termViaKitamasa(fib, [0, 1], 1000000, MOD), fibDoubling(1000000)[0],
    "F(10^6) by Kitamasa vs fast doubling");

  // (5) Characteristic polynomial: monic, coefficients -c_i, matching det(tI - M) numerically, and
  //     Cayley-Hamilton.
  for (const coeffs of [[1, 1], [1, 1, 1], [2, 3, 5], [1, 2, 3, 4]]) {
    const k = coeffs.length;
    const matrix = companionMatrix(coeffs, null);
    const characteristic = characteristicPolynomial(matrix);
    assert.strictEqual(characteristic.length, k + 1);
    assert.strictEqual(characteristic[0], 1n, "the polynomial is monic");
    for (let i = 0; i < k; i++) {
      assert.strictEqual(characteristic[i + 1], -BigInt(coeffs[i]), `coefficient ${i + 1} = -c_${i + 1}`);
    }
    for (const t of [0, 1, 2, 3]) {
      const shifted = matrix.map((row, i) =>
        row.map((value, j) => BigInt(t) * (i === j ? 1n : 0n) - value));
      assert.strictEqual(evaluatePolynomial(characteristic, t), numericDeterminant(shifted),
        `det(${t}I - M) must equal the symbolic polynomial at t = ${t}`);
    }
    // Cayley-Hamilton: P(M) = 0.
    let accumulator = zeros(k, k);
    let power = matIdentity(k);
    for (let i = k; i >= 0; i--) {
      const coefficient = i === 0 ? 1n : -BigInt(coeffs[i - 1]);
      for (let r = 0; r < k; r++) {
        for (let c = 0; c < k; c++) {
          accumulator[r][c] = mod(accumulator[r][c] + coefficient * power[r][c], MOD);
        }
      }
      if (i > 0) power = matMul(power, matrix, MOD);
    }
    assert.deepStrictEqual(accumulator, zeros(k, k), `Cayley-Hamilton for coeffs ${coeffs}`);
  }
  assert.deepStrictEqual(characteristicPolynomial(companionMatrix([1, 1], null)), [1n, -1n, -1n],
    "x^2 - x - 1 for Fibonacci");
  assert.deepStrictEqual(characteristicPolynomial(companionMatrix([1, 1, 1], null)),
    [1n, -1n, -1n, -1n], "x^3 - x^2 - x - 1 for Tribonacci");

  // (6) Reduction rules pinned by hand, companion structure, index boundaries and the geometric case.
  const matrix = companionMatrix([2, 3, 5], null);
  assert.deepStrictEqual(matrix.map((row) => row.map(Number)), [[2, 3, 5], [1, 0, 0], [0, 1, 0]]);
  assert.deepStrictEqual(polyPower([1, 1], 0, MOD), [1n, 0n], "x^0 = 1");
  assert.deepStrictEqual(polyPower([1, 1], 1, MOD), [0n, 1n], "x^1 = x");
  assert.deepStrictEqual(polyPower([1, 1], 2, MOD), [1n, 1n], "x^2 = x + 1");
  assert.deepStrictEqual(polyPower([1, 1], 3, MOD), [1n, 2n], "x^3 = x(x + 1) = 2x + 1");
  assert.deepStrictEqual(polyPower([1, 1], 4, MOD), [2n, 3n], "x^4 = x(2x + 1) = 3x + 2");
  assert.strictEqual(termViaCompanion([2, 3, 5], [1, 0, 0], 0, null), 1n, "a_0 = initial[0]");
  assert.strictEqual(termViaCompanion([2, 3, 5], [1, 0, 0], 1, null), 0n, "a_1 = initial[1]");
  assert.strictEqual(termViaCompanion([2, 3, 5], [1, 0, 0], 2, null), 0n, "a_2 = initial[2]");
  assert.strictEqual(termViaCompanion([2, 3, 5], [1, 0, 0], 3, null), 5n, "a_3 = c_3 a_0");
  assert.strictEqual(termViaKitamasa([2, 3, 5], [1, 0, 0], 3, null), 5n);
  for (const c of [2, 3, 7]) {
    assert.strictEqual(naiveSequence([c], [1], 20)[20], BigInt(c) ** 20n,
      "k = 1 is the geometric sequence");
    assert.strictEqual(termViaCompanion([c], [1], 20, null), BigInt(c) ** 20n);
    assert.strictEqual(termViaKitamasa([c], [1], 20, null), BigInt(c) ** 20n);
    assert.deepStrictEqual(polyPower([c], 20, MOD), [mod(BigInt(c) ** 20n, MOD)],
      "x^20 mod (x - c) = c^20");
  }
  assert.strictEqual(termViaCompanion([0, 0], [5, 7], 10, null), 0n, "all-zero coefficients vanish");
  assert.strictEqual(termViaKitamasa([0, 0], [5, 7], 10, null), 0n);
  assert.strictEqual(naiveSequence([0, 0], [5, 7], 10)[10], 0n);
  assert.throws(() => naiveSequence([1, 1], [0], 5), /too few initial values/);
  assert.strictEqual(termViaCompanion([], [], 5, null), 0n, "no coefficients, no terms");

  // (7) Work metrics: O(log n) products instead of O(n).
  for (const n of [64, 4096, 1000000]) {
    const matrixCounter = { multiplications: 0 };
    termViaCompanion(fib, [0, 1], n, MOD, matrixCounter);
    assert.ok(matrixCounter.multiplications <= 2 * Math.floor(Math.log2(n)) + 1,
      `n = ${n}: ${matrixCounter.multiplications} matrix products`);
    const polyCounter = { multiplications: 0 };
    termViaKitamasa(fib, [0, 1], n, MOD, polyCounter);
    assert.ok(polyCounter.multiplications <= 2 * Math.floor(Math.log2(n)) + 1,
      `n = ${n}: ${polyCounter.multiplications} polynomial products`);
    assert.ok(polyCounter.multiplications >= Math.log2(n) - 1, "and at least log2 n of them");
  }
  const baseline = { multiplications: 0 };
  matPower(companionMatrix(fib, MOD), 63, MOD, baseline);
  assert.ok(baseline.multiplications < 63,
    `exponentiation beats 63 products of repeated multiplication (${baseline.multiplications})`);
  assert.ok(matMul(matIdentity(2), companionMatrix(fib, MOD), MOD).length === 2);

  // (8) Determinism.
  assert.strictEqual(termViaCompanion([2, 3, 5], [1, 0, 0], 77, MOD),
    termViaCompanion([2, 3, 5], [1, 0, 0], 77, MOD));
  assert.strictEqual(termViaKitamasa([2, 3, 5], [1, 0, 0], 77, MOD),
    termViaKitamasa([2, 3, 5], [1, 0, 0], 77, MOD));
  assert.deepStrictEqual(naiveSequence([2, 3, 5], [1, 0, 0], 30),
    naiveSequence([2, 3, 5], [1, 0, 0], 30));
  assert.deepStrictEqual(characteristicPolynomial(companionMatrix([2, 3, 5], null)),
    characteristicPolynomial(companionMatrix([2, 3, 5], null)));
}

module.exports = { MOD, lcg64Next, lcg64Int, mod, reduce, naiveSequence, zeros, matMul, matIdentity,
  matPower, companionMatrix, termViaCompanion, polyMultiply, polyPower, termViaKitamasa, polyAdd,
  polySub, polyTimes, polyDeterminant, characteristicPolynomial, numericDeterminant,
  evaluatePolynomial, randomRecurrence, fibDoubling };

if (require.main === module) {
  main();
  console.log("PASS");
}
