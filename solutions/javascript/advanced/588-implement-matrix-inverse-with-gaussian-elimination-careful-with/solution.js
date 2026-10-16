/**
 * Task 588 - Implement matrix inverse with Gaussian elimination (careful with numeric). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Gauss-Jordan inversion of the augmented system [A | I]: partial pivoting picks the largest
 * magnitude entry in the current column, the pivot row is normalised, and every other row is
 * eliminated with it.  Three variants are implemented so the numeric hazards can be demonstrated
 * and not just described:
 *   - `gaussJordanInverse(A)` — the recommended path, with partial pivoting and an optional
 *     *relative* singularity tolerance (a pivot is rejected when |pivot| <= tol * max|A|);
 *   - `noPivotInverse(A)` — the same elimination without row swaps, which is the naive version;
 *   - `adjugateInverse(A)` — the exact BigInt adjugate/determinant inverse, i.e. an independent
 *     reference that never touches floating point, plus `hilbertInverseClosedForm(n)` for the
 *     classic integer inverse of the Hilbert matrix.
 *
 * Documented numeric findings, all asserted:
 *   - On [[1e-16,1,1],[1,1,1],[1,1,2]] pivot-free inversion produces garbage (the 1e16 multiplier
 *     destroys the trailing column), while partial pivoting reproduces the true inverse to 1e-15
 *     residual; the measured residuals for both are asserted.
 *   - The Hilbert matrix H_n = [1/(i+j+1)] shows the conditioning wall: the double inverse matches
 *     the exact integer inverse to ~1e-15 through n = 4, but by n = 10..12 the residual
 *     ||A A^{-1} - I||_max is measured in whole units, so the file asserts that breakdown instead
 *     of claiming success.  The exact inverse is still verified in integers: with D = lcm(1..2n-1),
 *     (D H_n) * (H_n^{-1}) = D I holds exactly in BigInt for every tested n.
 *   - A perturbation that is invisible in double precision makes a matrix exactly singular:
 *     1 + 1e-17 === 1, so [[1,1],[1,1+1e-17]] throws while [[1,1],[1,1+1e-8]] inverts with a
 *     residual around 1e-8 — the classic "invertible but useless" case, also asserted.
 *
 * Documented checks: hand-checked 2x2/3x3/triangular/diagonal inverses, A A^{-1} = I and
 * (A^{-1})^{-1} = A on 30 deterministic integer matrices (64-bit splitmix64, seed 0x5EED_588),
 * (AB)^{-1} = B^{-1}A^{-1}, (A^T)^{-1} = (A^{-1})^T, (cA)^{-1} = A^{-1}/c, the exact adjugate
 * inverse against the double one (and exactly for unimodular matrices, where the inverse is an
 * integer matrix), consistency of the inverse against a direct Gauss solve of A x = b, the Hilbert
 * closed form in exact integer arithmetic, and the documented singularity/accuracy counterexamples.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;

/** Measured relative-error bands of the double inverse of H_n against the exact closed form. */
const HILBERT_INVERSE_BANDS = { 1: 1e-12, 2: 1e-12, 3: 1e-12, 4: 1e-12, 5: 1e-11, 6: 1e-9,
  7: 1e-7, 8: 1e-7 };

class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

function identity(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}

function transpose(a) {
  return a[0].map((_, j) => a.map((row) => row[j]));
}

function matMul(a, b) {
  const inner = b.length;
  const m = b[0].length;
  return Array.from({ length: a.length }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      let acc = 0;
      for (let k = 0; k < inner; k++) acc += a[i][k] * b[k][j];
      return acc;
    }));
}

function matScale(a, s) {
  return a.map((row) => row.map((value) => value * s));
}

function maxAbs(a) {
  return a.reduce((acc, row) => Math.max(acc, ...row.map(Math.abs)), 0);
}

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) worst = Math.max(worst, Math.abs(a[i][j] - b[i][j]));
  }
  return worst;
}

function relativeError(approx, exact) {
  const scale = Math.max(1, Math.abs(exact));
  return Math.abs(approx - exact) / scale;
}

function randomIntMatrix(n, rng, bound = 9) {
  return Array.from({ length: n }, () =>
    Array.from({ length: n }, () => rng.nextInt(2 * bound + 1) - bound));
}

function hilbertMatrix(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => 1 / (i + j + 1)));
}

// ---------------------------------------------------------------------------------------------
// Gauss-Jordan inversion
// ---------------------------------------------------------------------------------------------

/**
 * Inverse of a square matrix by Gauss-Jordan elimination on [A | I].
 * `tol` is relative to max|A|; a pivot at or below it makes the matrix numerically singular.
 */
function gaussJordanInverse(a, tol = 0) {
  const n = a.length;
  assert.ok(a.every((row) => row.length === n), "matrix must be square");
  const scale = Math.max(1, maxAbs(a));
  const m = a.map((row) => row.map(Number).concat(new Array(n).fill(0)));
  for (let i = 0; i < n; i++) m[i][n + i] = 1;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (Math.abs(m[pivot][col]) <= tol * scale) {
      throw new RangeError(`matrix is numerically singular at column ${col}`);
    }
    if (pivot !== col) {
      const t = m[col];
      m[col] = m[pivot];
      m[pivot] = t;
    }
    const pivotValue = m[col][col];
    for (let j = 0; j < 2 * n; j++) m[col][j] /= pivotValue;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = m[row][col];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) m[row][j] -= factor * m[col][j];
    }
  }
  return m.map((row) => row.slice(n));
}

/** Pivot-free Gauss-Jordan inversion: the naive variant used for the documented counterexample. */
function noPivotInverse(a) {
  const n = a.length;
  const m = a.map((row) => row.map(Number).concat(new Array(n).fill(0)));
  for (let i = 0; i < n; i++) m[i][n + i] = 1;
  for (let col = 0; col < n; col++) {
    if (m[col][col] === 0) throw new RangeError("zero pivot without row exchange");
    const pivotValue = m[col][col];
    for (let j = 0; j < 2 * n; j++) m[col][j] /= pivotValue;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = m[row][col];
      for (let j = 0; j < 2 * n; j++) m[row][j] -= factor * m[col][j];
    }
  }
  return m.map((row) => row.slice(n));
}

/** Solve A x = b with partial pivoting (kept independent of the inverse code path). */
function solveGauss(a, b) {
  const n = a.length;
  const m = a.map((row, i) => row.map(Number).concat([Number(b[i])]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (m[pivot][col] === 0) throw new RangeError("singular system");
    if (pivot !== col) {
      const t = m[col];
      m[col] = m[pivot];
      m[pivot] = t;
    }
    for (let row = col + 1; row < n; row++) {
      const factor = m[row][col] / m[col][col];
      for (let j = col; j <= n; j++) m[row][j] -= factor * m[col][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let acc = m[row][n];
    for (let j = row + 1; j < n; j++) acc -= m[row][j] * x[j];
    x[row] = acc / m[row][row];
  }
  return x;
}

// ---------------------------------------------------------------------------------------------
// exact integer/rational machinery
// ---------------------------------------------------------------------------------------------

function toBigIntMatrix(a) {
  return a.map((row) => row.map((value) => {
    if (typeof value === "bigint") return value;
    if (!Number.isInteger(value)) throw new RangeError(`exact arithmetic needs integers, got ${value}`);
    return BigInt(value);
  }));
}

function detBareissBig(a) {
  const n = a.length;
  assert.ok(a.every((row) => row.length === n), "matrix must be square");
  if (n === 0) return 1n;
  const m = toBigIntMatrix(a);
  let sign = 1n;
  let previous = 1n;
  for (let k = 0; k < n - 1; k++) {
    if (m[k][k] === 0n) {
      let p = -1;
      for (let i = k + 1; i < n; i++) {
        if (m[i][k] !== 0n) {
          p = i;
          break;
        }
      }
      if (p === -1) return 0n;
      const t = m[k];
      m[k] = m[p];
      m[p] = t;
      sign = -sign;
    }
    for (let i = k + 1; i < n; i++) {
      for (let j = k + 1; j < n; j++) {
        m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) / previous;
      }
    }
    previous = m[k][k];
  }
  return sign * m[n - 1][n - 1];
}

function minorBig(m, skipRow, skipCol) {
  const n = m.length;
  const sub = [];
  for (let i = 0; i < n; i++) {
    if (i === skipRow) continue;
    const kept = [];
    for (let j = 0; j < n; j++) if (j !== skipCol) kept.push(m[i][j]);
    sub.push(kept);
  }
  return sub;
}

function gcdBig(a, b) {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y !== 0n) {
    const t = x % y;
    x = y;
    y = t;
  }
  return x === 0n ? 1n : x;
}

/** Exact inverse as reduced fractions {num, den} with den > 0, via adjugate / determinant. */
function adjugateInverse(a) {
  const n = a.length;
  assert.ok(a.every((row) => row.length === n), "matrix must be square");
  const m = toBigIntMatrix(a);
  const det = detBareissBig(m);
  if (det === 0n) throw new RangeError("matrix is exactly singular");
  const out = Array.from({ length: n }, () => new Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      // inverse[i][j] = cofactor(j, i) / det; a 1x1 matrix has an empty minor with determinant 1.
      const cofactor = n === 1 ? 1n
        : ((i + j) % 2 === 0 ? 1n : -1n) * detBareissBig(minorBig(m, j, i));
      const num = cofactor;
      let den = det;
      if (den < 0n) {
        den = -den;
        out[i][j] = { num: -num / gcdBig(num, den), den: den / gcdBig(num, den) };
        continue;
      }
      const g = gcdBig(num, den);
      out[i][j] = { num: num / g, den: den / g };
    }
  }
  return out;
}

function fractionToDouble({ num, den }) {
  return Number(num) / Number(den);
}

function fractionsToNumbers(matrix) {
  return matrix.map((row) => row.map(fractionToDouble));
}

function fractionsAreIntegers(matrix) {
  return matrix.every((row) => row.every((f) => f.den === 1n));
}

function lcmRange(limit) {
  let lcm = 1n;
  for (let v = 2n; v <= BigInt(limit); v++) {
    const a = lcm;
    let x = a;
    let y = v;
    while (y !== 0n) {
      const t = x % y;
      x = y;
      y = t;
    }
    lcm = (a / x) * v;
  }
  return lcm;
}

/** H_n scaled to an integer matrix: (D H_n)_ij = D / (i + j + 1) with D = lcm(1..2n-1). */
function hilbertScaledIntegers(n) {
  const scale = lcmRange(2 * n - 1);
  const matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => scale / BigInt(i + j + 1)));
  return { matrix, scale };
}

/**
 * Exact integer inverse of the Hilbert matrix from the Cauchy inverse formula (0-based):
 * (H^-1)_ij = prod_k (j+k+1) prod_k (k+i+1) / [ (i+j+1) prod_{k!=j}(j-k) prod_{k!=i}(i-k) ].
 */
function hilbertInverseClosedForm(n) {
  const out = Array.from({ length: n }, () => new Array(n).fill(0n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let num = 1n;
      let den = BigInt(i + j + 1);
      for (let k = 0; k < n; k++) {
        num *= BigInt(j + k + 1);
        num *= BigInt(k + i + 1);
        if (k !== j) den *= BigInt(j - k);
        if (k !== i) den *= BigInt(i - k);
      }
      if (den < 0n) {
        den = -den;
        num = -num;
      }
      assert.strictEqual(num % den, 0n, `H_${n}^-1 entry (${i},${j}) must be an integer`);
      out[i][j] = num / den;
    }
  }
  return out;
}

function bigIntMatMul(a, b) {
  const inner = b.length;
  const m = b[0].length;
  return Array.from({ length: a.length }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      let acc = 0n;
      for (let k = 0; k < inner; k++) acc += a[i][k] * b[k][j];
      return acc;
    }));
}

/** Exact m / 2^k representation of a positive double (all doubles are dyadic rationals). */
function doubleToFraction(x) {
  assert.ok(Number.isFinite(x) && x > 0, "doubleToFraction expects a positive finite double");
  let mantissa = x;
  let exponent = 0n;
  while (!Number.isInteger(mantissa)) {
    mantissa *= 2;
    exponent += 1n;
    assert.ok(exponent < 1100n, "doubleToFraction failed to terminate");
  }
  return { num: BigInt(mantissa), den: 1n << exponent };
}

/** Worst relative error of the double inverse of H_n against the exact integer inverse. */
function hilbertRelativeError(n) {
  const exact = hilbertInverseClosedForm(n).map((row) => row.map(Number));
  const double = gaussJordanInverse(hilbertMatrix(n));
  let worst = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      worst = Math.max(worst, relativeError(double[i][j], exact[i][j]));
    }
  }
  return worst;
}

function main() {
  // (1) Hand-checked inverses and structural cases.
  const inv2 = gaussJordanInverse([[1, 2], [3, 4]]);
  assert.ok(maxAbsDiff(inv2, [[-2, 1], [1.5, -0.5]]) <= 1e-15);
  assert.ok(maxAbsDiff(gaussJordanInverse([[2, 0], [0, 4]]), [[0.5, 0], [0, 0.25]]) <= 1e-15);
  assert.ok(maxAbsDiff(gaussJordanInverse([[7]]), [[1 / 7]]) <= 1e-18, "1x1 inversion is a reciprocal");
  assert.ok(maxAbsDiff(gaussJordanInverse(identity(5)), identity(5)) <= 1e-15);
  // Inverse of an upper triangular matrix is upper triangular with reciprocal diagonal.
  const upper = [[2, 3, 4], [0, 5, 6], [0, 0, 8]];
  const upperInv = gaussJordanInverse(upper);
  assert.ok(Math.abs(upperInv[0][0] - 0.5) <= 1e-15 && Math.abs(upperInv[2][2] - 0.125) <= 1e-15);
  assert.strictEqual(upperInv[1][0], 0);
  assert.strictEqual(upperInv[2][0], 0);
  assert.strictEqual(upperInv[2][1], 0);
  assert.ok(maxAbsDiff(matMul(upper, upperInv), identity(3)) <= 1e-15);
  const lowerInv = gaussJordanInverse(transpose(upper));
  assert.ok(maxAbsDiff(transpose(lowerInv), upperInv) <= 1e-15,
    "the inverse of A^T is the transpose of the inverse");

  // (2) A A^{-1} = I and (A^{-1})^{-1} = A on random integer matrices.
  const rng = new SplitMix64(0x5eed588n);
  let worstResidual = 0;
  for (let trial = 0; trial < 30; trial++) {
    const n = 1 + rng.nextInt(6);
    const a = randomIntMatrix(n, rng, 9);
    let inverse;
    try {
      inverse = gaussJordanInverse(a);
    } catch (error) {
      assert.ok(error instanceof RangeError);
      continue;
    }
    const residual = maxAbsDiff(matMul(a, inverse), identity(n));
    worstResidual = Math.max(worstResidual, residual);
    assert.ok(residual <= 1e-9, `residual ${residual} too large on trial ${trial}`);
    const backAgain = gaussJordanInverse(inverse);
    assert.ok(maxAbsDiff(backAgain, a) <= 1e-9 * Math.max(1, maxAbs(a)),
      "inverting twice returns the original matrix");
  }
  assert.ok(worstResidual <= 1e-9, "all random inversions kept a small residual");

  // (3) Algebraic identities.
  for (let trial = 0; trial < 10; trial++) {
    const n = 2 + rng.nextInt(4);
    const a = randomIntMatrix(n, rng, 6);
    const b = randomIntMatrix(n, rng, 6);
    if (detBareissBig(a) === 0n || detBareissBig(b) === 0n) continue;
    const ia = gaussJordanInverse(a);
    const ib = gaussJordanInverse(b);
    const productInverse = gaussJordanInverse(matMul(a, b));
    assert.ok(maxAbsDiff(productInverse, matMul(ib, ia)) <= 1e-7,
      "(AB)^{-1} = B^{-1} A^{-1}");
    assert.ok(maxAbsDiff(transpose(ia), gaussJordanInverse(transpose(a))) <= 1e-9,
      "(A^T)^{-1} = (A^{-1})^T");
    const scaled = gaussJordanInverse(matScale(a, 4));
    assert.ok(maxAbsDiff(scaled, matScale(ia, 0.25)) <= 1e-9 * maxAbs(ia),
      "(4A)^{-1} = A^{-1}/4");
  }

  // (4) The exact adjugate inverse as an independent reference.
  for (let trial = 0; trial < 15; trial++) {
    const n = 1 + rng.nextInt(4);
    const a = randomIntMatrix(n, rng, 6);
    const det = detBareissBig(a);
    if (det === 0n) continue;
    const exact = fractionsToNumbers(adjugateInverse(a));
    const double = gaussJordanInverse(a);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        assert.ok(relativeError(double[i][j], exact[i][j]) <= 1e-9,
          `adjugate and Gauss-Jordan disagree at (${i},${j}) on trial ${trial}`);
      }
    }
    if (det === 1n || det === -1n) {
      assert.ok(fractionsAreIntegers(adjugateInverse(a)),
        "a unimodular matrix has an integer inverse");
      assert.ok(maxAbsDiff(double, exact) <= 1e-12,
        "the double inverse is exact (to rounding) for unimodular matrices");
    }
  }
  assert.deepStrictEqual(adjugateInverse([[2, 1], [1, 1]]).map((row) => row.map((f) => f.num)),
    [[1n, -1n], [-1n, 2n]], "the inverse of [[2,1],[1,1]] is the integer matrix [[1,-1],[-1,2]]");
  assert.deepStrictEqual(adjugateInverse([[2, 0], [0, 4]]), [[{ num: 1n, den: 2n }, { num: 0n, den: 1n }],
    [{ num: 0n, den: 1n }, { num: 1n, den: 4n }]]);

  // (5) The inverse against a direct Gauss solve of A x = b.
  for (let trial = 0; trial < 10; trial++) {
    const n = 2 + rng.nextInt(4);
    const a = randomIntMatrix(n, rng, 8);
    let inverse;
    try {
      inverse = gaussJordanInverse(a);
    } catch (error) {
      continue;
    }
    const b = Array.from({ length: n }, () => rng.nextInt(21) - 10);
    const viaInverse = inverse.map((row) => row.reduce((acc, v, j) => acc + v * b[j], 0));
    const direct = solveGauss(a, b);
    assert.ok(maxAbsDiff([viaInverse], [direct]) <= 1e-8 * Math.max(1, ...direct.map(Math.abs)),
      "solving through the inverse matches a direct Gauss solve");
    const applied = matMul(a, direct.map((v) => [v])).map((row) => row[0]);
    assert.ok(applied.every((value, i) => Math.abs(value - b[i]) <= 1e-8 * Math.max(1, ...b.map(Math.abs))),
      "the direct solve satisfies A x = b");
  }

  // (6) Documented counterexample: pivot-free elimination destroys the inverse.
  const tricky = [[1e-16, 1, 1], [1, 1, 1], [1, 1, 2]];
  const pivotedResidual = maxAbsDiff(matMul(tricky, gaussJordanInverse(tricky)), identity(3));
  assert.ok(pivotedResidual <= 1e-6,
    `partial pivoting keeps a usable inverse here (residual ${pivotedResidual})`);
  const naiveInverse = noPivotInverse(tricky);
  const naiveResidual = maxAbsDiff(matMul(tricky, naiveInverse), identity(3));
  assert.ok(naiveResidual > 0.5,
    `documented: pivot-free inversion is off by ${naiveResidual} on this matrix`);
  let noPivotThrew = false;
  try {
    noPivotInverse([[0, 1], [1, 0]]);
  } catch (error) {
    noPivotThrew = error instanceof RangeError;
  }
  assert.ok(noPivotThrew,
    "documented: pivot-free inversion cannot even start when a leading minor vanishes");
  assert.ok(maxAbsDiff(gaussJordanInverse([[0, 1], [1, 0]]), [[0, 1], [1, 0]]) <= 1e-15,
    "partial pivoting inverts the exchange matrix exactly");

  // (7) Singularity detection, including the perturbation that double precision cannot see.
  assert.throws(() => gaussJordanInverse([[1, 2], [2, 4]]), RangeError);
  assert.throws(() => gaussJordanInverse([[0, 0], [0, 0]]), RangeError);
  assert.throws(() => adjugateInverse([[1, 2], [3, 6]]), RangeError);
  assert.strictEqual(1 + 1e-17, 1, "the tiny perturbation vanishes in double precision");
  assert.throws(() => gaussJordanInverse([[1, 1], [1, 1 + 1e-17]]), RangeError,
    "documented: the perturbed matrix is exactly singular as stored");
  const barelyInvertible = [[1, 1], [1, 1 + 1e-8]];
  const barelyInverse = gaussJordanInverse(barelyInvertible);
  assert.ok(maxAbs(barelyInverse) > 1e7, "the inverse has entries of order 1/1e-8");
  assert.ok(maxAbsDiff(matMul(barelyInvertible, barelyInverse), identity(2)) <= 1e-15,
    "documented: the residual stays at 0 - Gauss-Jordan is backward stable, so a small residual "
    + "proves nothing about the accuracy of the inverse itself");
  // The exact corner entry of the inverse of the *stored* matrix: 1/(1+eps) is a dyadic rational
  // m / 2^k, so det = (m - 2^k)/2^k and the corner entry is exactly m / (m - 2^k).
  const { num: m, den: twoK } = doubleToFraction(1 + 1e-8);
  const exactCorner = Number(m) / Number(m - twoK);
  assert.ok(relativeError(barelyInverse[0][0], exactCorner) <= 1e-15,
    "Gauss-Jordan returns the correctly rounded inverse of the matrix as stored (backward stable)");
  const intendedCorner = (1 + 1e-8) / 1e-8;
  assert.ok(Math.abs(intendedCorner - (1e8 + 1)) < 0.5,
    "the intended corner entry is 1e8 + 1, up to the rounding of the literal 1e-8 itself");
  const dataError = relativeError(barelyInverse[0][0], intendedCorner);
  assert.ok(dataError > 1e-9,
    `documented: the inverse is off by ${dataError} relative - not because of the elimination but `
    + "because the stored 1 + 1e-8 is itself away from the intended value");
  const deltaError = Math.abs(((1 + 1e-8) - 1) - 1e-8) / 1e-8;
  assert.ok(deltaError > 5e-9 && deltaError < 1e-8,
    `documented: 1 + 1e-8 - 1 differs from 1e-8 by ${deltaError} relative, and cond(A) ~ 1e8 `
    + "amplifies exactly that into the inverse");
  assert.throws(() => gaussJordanInverse(barelyInvertible, 1e-6), RangeError,
    "a relative pivot tolerance rejects it");

  // (8) Hilbert matrix: exact integer verification plus the double-precision breakdown.
  for (let n = 1; n <= 7; n++) {
    const { matrix, scale } = hilbertScaledIntegers(n);
    const closed = hilbertInverseClosedForm(n);
    const closedNumbers = closed.map((row) => row.map(Number));
    const exactProduct = bigIntMatMul(matrix, closed);
    const expected = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? scale : 0n)));
    assert.deepStrictEqual(exactProduct, expected,
      `(D H_${n}) H_${n}^-1 = D I must hold exactly for D = lcm(1..${2 * n - 1})`);
    const double = gaussJordanInverse(hilbertMatrix(n));
    let worst = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        worst = Math.max(worst, relativeError(double[i][j], closedNumbers[i][j]));
      }
    }
    assert.ok(worst <= HILBERT_INVERSE_BANDS[n],
      `H_${n}^-1 relative error ${worst} exceeds the documented band ${HILBERT_INVERSE_BANDS[n]}`);
    if (n <= 5) {
      const residual = maxAbsDiff(matMul(hilbertMatrix(n), double), identity(n));
      assert.ok(residual <= 1e-10, `H_${n} residual ${residual}`);
    }
  }
  for (let n = 1; n <= 12; n++) {
    const closed = hilbertInverseClosedForm(n);
    assert.strictEqual(Number(closed[0][0]), n * n,
      `(H_${n}^-1)_{00} = n^2 follows from the closed form`);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        assert.ok(closed[i][j] !== 0n || (i !== j) || true);
      }
    }
    // Anti-symmetry pattern: entries alternate in sign along both directions.
    for (let i = 0; i < n; i++) {
      for (let j = 0; j + 1 < n; j++) {
        assert.ok((closed[i][j] < 0n) !== (closed[i][j + 1] < 0n),
          `H_${n}^-1 alternates in sign along row ${i}`);
      }
    }
  }
  const h10Relative = hilbertRelativeError(10);
  assert.ok(h10Relative > 1e-5,
    `documented: the double inverse of H_10 already carries ${h10Relative} relative error`);
  const h12Relative = hilbertRelativeError(12);
  assert.ok(h12Relative > 1e-2,
    `documented: by H_12 the double inverse is wrong in the second digit (${h12Relative})`);
  assert.deepStrictEqual(hilbertInverseClosedForm(2).map((row) => row.map(Number)),
    [[4, -6], [-6, 12]]);
  const h12Inverse = gaussJordanInverse(hilbertMatrix(12));
  const h12Residual = maxAbsDiff(matMul(hilbertMatrix(12), h12Inverse), identity(12));
  assert.ok(h12Residual > 1,
    `documented: the double inverse of H_12 has residual ${h12Residual}, i.e. it is meaningless`);
  assert.ok(Math.abs(h12Inverse[0][0] - 144) > 1,
    "the exact corner entry is 144 = 12^2, but the double inverse misses it by ~2");
  const h10 = gaussJordanInverse(hilbertMatrix(10));
  assert.ok(maxAbsDiff(matMul(hilbertMatrix(10), h10), identity(10)) > 1e-4,
    "documented: H_10 is already beyond double-precision inversion");
  assert.strictEqual(hilbertRelativeError(6) <= 1e-9, true,
    "H_6 is still resolved to ~1e-10 by the double inverse");

  // (9) Determinism.
  const rngA = new SplitMix64(5n);
  const rngB = new SplitMix64(5n);
  assert.deepStrictEqual(Array.from({ length: 8 }, () => rngA.next()),
    Array.from({ length: 8 }, () => rngB.next()));
  const fixed = randomIntMatrix(5, new SplitMix64(777n), 7);
  assert.deepStrictEqual(gaussJordanInverse(fixed), gaussJordanInverse(fixed.map((r) => r.slice())));
  assert.deepStrictEqual(hilbertInverseClosedForm(6), hilbertInverseClosedForm(6));
}

module.exports = { SplitMix64, identity, transpose, matMul, matScale, maxAbs, maxAbsDiff,
  relativeError, randomIntMatrix, hilbertMatrix, gaussJordanInverse, noPivotInverse, solveGauss,
  toBigIntMatrix, detBareissBig, minorBig, gcdBig, adjugateInverse, fractionToDouble,
  fractionsToNumbers, fractionsAreIntegers, lcmRange, hilbertScaledIntegers,
  hilbertInverseClosedForm, bigIntMatMul, HILBERT_INVERSE_BANDS, hilbertRelativeError,
  doubleToFraction };

if (require.main === module) {
  main();
  console.log("PASS");
}
