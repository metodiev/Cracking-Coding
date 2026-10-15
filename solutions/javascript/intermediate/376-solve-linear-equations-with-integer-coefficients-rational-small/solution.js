/**
 * Task 376 - Exact linear systems over rationals (integer coefficients). (Tier: Intermediate, CTCI Ch 6)
 *
 * The same Gaussian elimination as task 375, but every entry is an exact rational
 * number: a small Fraction type on top of gcd (implemented here per language) keeps
 * num/den normalised with den > 0.  All arithmetic is exact, so:
 *   * the systems of task 375 return exact rational solutions - every assertion is an
 *     EQUALITY of rationals, NOT a tolerance check (contrast with 375);
 *   * no pivoting strategy is required: the first non-zero pivot can be taken because
 *     exact arithmetic never introduces rounding, so a zero pivot really is zero
 *     (documented; 375's partial pivoting exists to limit rounding, which cannot
 *     happen here);
 *   * sentinels: status "unique"/"infinite"/"inconsistent", x = null unless unique.
 *
 * Cross-check with floats: a compact double-precision Gaussian implemented HERE (same
 * file, duplicating 375's algorithm) must agree with the exact result to 1e-9 - this is
 * the documented link between the two tasks; with small integer inputs the doubles
 * round the rationals faithfully.
 *
 * Inputs and expected solutions include genuine fractions (1/3, 1/6) whose float images
 * cannot be represented exactly - those are asserted exactly on the rationals.
 * All intermediate values stay far inside the safe integer range (entries <= ~1e6).
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

function gcd(a, b) {
  while (b !== 0) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a;
}

/** Exact fraction with normalised sign and gcd-reduced terms. */
class Rational {
  constructor(num, den = 1) {
    if (den === 0) throw new Error("rational with zero denominator");
    if (den < 0) {
      num = -num;
      den = -den;
    }
    const g = gcd(Math.abs(num), den);
    if (g === 0) {
      this.num = 0;
      this.den = 1;
    } else {
      this.num = num / g;
      this.den = den / g;
    }
  }

  add(other) {
    return new Rational(this.num * other.den + other.num * this.den, this.den * other.den);
  }

  sub(other) {
    return new Rational(this.num * other.den - other.num * this.den, this.den * other.den);
  }

  mul(other) {
    return new Rational(this.num * other.num, this.den * other.den);
  }

  div(other) {
    if (other.num === 0) throw new Error("division by zero rational");
    return new Rational(this.num * other.den, this.den * other.num);
  }

  isZero() {
    return this.num === 0;
  }

  equals(other) {
    return this.num === other.num && this.den === other.den;
  }

  toString() {
    return this.den === 1 ? String(this.num) : `${this.num}/${this.den}`;
  }
}

/** Exact Gaussian elimination; returns {status, x} with x = null unless unique. */
function solveRational(matrix, rhs) {
  const n = matrix.length;
  const augmented = matrix.map((row, i) => [...row.map((v) => new Rational(v)), new Rational(rhs[i])]);

  let rank = 0;
  for (let col = 0; col < n; col++) {
    let pivotRow = -1;
    for (let row = rank; row < n; row++) {
      if (!augmented[row][col].isZero()) {         // exact zero test
        pivotRow = row;
        break;
      }
    }
    if (pivotRow === -1) continue;                 // genuinely free column
    [augmented[rank], augmented[pivotRow]] = [augmented[pivotRow], augmented[rank]];
    for (let row = rank + 1; row < n; row++) {
      if (augmented[row][col].isZero()) continue;
      const factor = augmented[row][col].div(augmented[rank][col]);
      for (let k = col; k <= n; k++) {
        augmented[row][k] = augmented[row][k].sub(factor.mul(augmented[rank][k]));
      }
    }
    rank++;
  }

  for (let row = rank; row < n; row++) {
    if (!augmented[row][n].isZero()) return { status: "inconsistent", x: null };
  }
  if (rank < n) return { status: "infinite", x: null };

  const x = new Array(n).fill(null).map(() => new Rational(0));
  for (let row = n - 1; row >= 0; row--) {
    let total = augmented[row][n];
    for (let col = row + 1; col < n; col++) total = total.sub(augmented[row][col].mul(x[col]));
    x[row] = total.div(augmented[row][row]);
  }
  return { status: "unique", x };
}

/** Compact double-precision copy of 375's elimination, used only as a cross-check. */
function solveDouble(matrix, rhs) {
  const eps = 1e-12;
  const n = matrix.length;
  const augmented = matrix.map((row, i) => [...row.map(Number), Number(rhs[i])]);
  let rank = 0;
  for (let col = 0; col < n; col++) {
    let pivotRow = -1;
    let pivotSize = 0;
    for (let row = rank; row < n; row++) {
      if (Math.abs(augmented[row][col]) > pivotSize) {
        pivotSize = Math.abs(augmented[row][col]);
        pivotRow = row;
      }
    }
    if (pivotRow === -1 || pivotSize <= eps) continue;
    [augmented[rank], augmented[pivotRow]] = [augmented[pivotRow], augmented[rank]];
    for (let row = rank + 1; row < n; row++) {
      const factor = augmented[row][col] / augmented[rank][col];
      for (let k = col; k <= n; k++) augmented[row][k] -= factor * augmented[rank][k];
    }
    rank++;
  }
  if (rank < n) return null;
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let total = augmented[row][n];
    for (let col = row + 1; col < n; col++) total -= augmented[row][col] * x[col];
    x[row] = total / augmented[row][row];
  }
  return x;
}

function sameRationals(got, expectedNum, expectedDen) {
  if (got.length !== expectedNum.length) return false;
  return got.every((value, i) => value.equals(new Rational(expectedNum[i], expectedDen[i])));
}

function lcgSystem(lcg) {
  const size = 2 + lcg.next(3);
  const matrix = [];
  for (let i = 0; i < size; i++) {
    const row = [];
    for (let j = 0; j < size; j++) row.push(lcg.next(11) - 5);
    matrix.push(row);
  }
  const planted = [];
  for (let i = 0; i < size; i++) planted.push(lcg.next(11) - 5);
  const rhs = matrix.map((row) => row.reduce((sum, value, j) => sum + value * planted[j], 0));
  return { matrix, rhs, planted };
}

module.exports = { Rational, solveRational, solveDouble };

if (require.main === module) {
  const assert = require("node:assert");

  // 2x2 integer system: x = 2, y = 1, exactly
  let result = solveRational([[1, 1], [1, -1]], [3, 1]);
  assert.strictEqual(result.status, "unique", "2x2 unique");
  assert.ok(sameRationals(result.x, [2, 1], [1, 1]), "2x2 exact (2, 1)");

  // pivot swap: leading zero, x = (3, 2)
  result = solveRational([[0, 1], [1, 0]], [2, 3]);
  assert.strictEqual(result.status, "unique", "pivot swap unique");
  assert.ok(sameRationals(result.x, [3, 2], [1, 1]), "pivot swap (3, 2)");

  // classic 3x3, solution (5, 3, -2)
  result = solveRational([[1, 1, 1], [0, 2, 5], [2, 5, -1]], [6, -4, 27]);
  assert.strictEqual(result.status, "unique", "3x3 unique");
  assert.ok(sameRationals(result.x, [5, 3, -2], [1, 1, 1]), "3x3 exact (5, 3, -2)");

  // 4x4 diagonally dominant, solution (1, 2, 3, 4)
  result = solveRational([[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]], [11, 12, 13, 14]);
  assert.strictEqual(result.status, "unique", "4x4 unique");
  assert.ok(sameRationals(result.x, [1, 2, 3, 4], [1, 1, 1, 1]), "4x4 exact");

  // genuine fractions: 3x = 1 -> x = 1/3
  result = solveRational([[3]], [1]);
  assert.strictEqual(result.status, "unique", "1x1 unique");
  assert.ok(sameRationals(result.x, [1], [3]), "1x1 exact 1/3");

  // 2x + 4y = 1, x - y = 0 -> x = y = 1/6
  result = solveRational([[2, 4], [1, -1]], [1, 0]);
  assert.strictEqual(result.status, "unique", "1/6 system unique");
  assert.ok(sameRationals(result.x, [1, 1], [6, 6]), "1/6 exact");

  // 5x + 2y = 1, x + y = 0 -> x = 1/3, y = -1/3
  result = solveRational([[5, 2], [1, 1]], [1, 0]);
  assert.strictEqual(result.status, "unique", "third system unique");
  assert.ok(sameRationals(result.x, [1, -1], [3, 3]), "1/3, -1/3 exact");

  // sentinels behave exactly like 375's
  result = solveRational([[1, 1], [2, 2]], [1, 2]);
  assert.ok(result.status === "infinite" && result.x === null, "infinite sentinel");
  result = solveRational([[1, 1], [2, 2]], [1, 5]);
  assert.ok(result.status === "inconsistent" && result.x === null, "inconsistent sentinel");
  result = solveRational([[1, 0, 0], [0, 1, 0], [1, 1, 0]], [1, 2, 3]);
  assert.ok(result.status === "infinite" && result.x === null, "3x3 dependent infinite");
  result = solveRational([[1, 0, 0], [0, 1, 0], [1, 1, 0]], [1, 2, 4]);
  assert.ok(result.status === "inconsistent" && result.x === null, "3x3 dependent inconsistent");

  // --- float cross-check: the double solver agrees to 1e-9 ---------------
  const matrices = [
    [[1, 1], [1, -1]], [[0, 1], [1, 0]], [[1, 1, 1], [0, 2, 5], [2, 5, -1]],
    [[3]], [[2, 4], [1, -1]], [[5, 2], [1, 1]],
  ];
  const rhss = [[3, 1], [2, 3], [6, -4, 27], [1], [1, 0], [1, 0]];
  const expectedNums = [[2, 1], [3, 2], [5, 3, -2], [1], [1, 1], [1, -1]];
  const expectedDens = [[1, 1], [1, 1], [1, 1, 1], [3], [6, 6], [3, 3]];
  for (let s = 0; s < matrices.length; s++) {
    const approx = solveDouble(matrices[s], rhss[s]);
    assert.notStrictEqual(approx, null, `float cross-check solvable ${s}`);
    for (let i = 0; i < approx.length; i++) {
      const exact = expectedNums[s][i] / expectedDens[s][i];
      assert.ok(Math.abs(approx[i] - exact) < 1e-9, `float cross-check ${s} index ${i}`);
    }
  }

  // --- LCG cross-checks: exact recovery of the planted solution ----------
  const lcg = new Lcg(37613);
  let uniqueCount = 0;
  for (let trial = 0; trial < 80; trial++) {
    const { matrix, rhs, planted } = lcgSystem(lcg);
    result = solveRational(matrix, rhs);
    if (result.status !== "unique") continue;        // singular; skipped
    uniqueCount++;
    assert.ok(sameRationals(result.x, planted, planted.map(() => 1)), `planted solution ${trial}`);
    // exact substitution A x == b, no tolerance involved
    for (let i = 0; i < matrix.length; i++) {
      let total = new Rational(0);
      for (let j = 0; j < matrix[i].length; j++) total = total.add(new Rational(matrix[i][j]).mul(result.x[j]));
      assert.ok(total.equals(new Rational(rhs[i])), `exact substitution ${trial}`);
    }
    const approx = solveDouble(matrix, rhs);
    assert.notStrictEqual(approx, null, `float path solvable ${trial}`);
    for (let i = 0; i < planted.length; i++) {
      assert.ok(Math.abs(approx[i] - planted[i]) < 1e-9, `float agrees ${trial}`);
    }
  }
  assert.ok(uniqueCount >= 40, `enough unique systems, got ${uniqueCount}`);
  console.log("PASS");
}
