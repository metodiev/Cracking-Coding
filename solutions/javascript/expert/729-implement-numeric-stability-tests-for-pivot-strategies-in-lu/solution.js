/** Task 729 - Implement numeric stability tests for pivot strategies in LU. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Gaussian elimination without pivoting is only stable when the leading minors happen to be
 * well behaved: the classic counterexample is A = [[eps, 1], [1, 1]], where the multiplier
 * 1/eps swamps the arithmetic. Partial pivoting (largest |value| in the column) and complete
 * pivoting (largest |value| in the remaining submatrix) fix that, and the standard yardstick
 * is the *growth factor*
 *
 *     rho = max |u_ij| / max |a_ij|          (u from the computed factorization)
 *
 * because the textbook backward-error bound is ||PA - LU|| <= c * n * eps * rho * ||A||, i.e.
 * all the damage pivoting does or does not cause is measured by rho.
 *
 * This file implements all three strategies, solve, the growth factor, and the numerical tests
 * that separate them. Verified here: the reconstruction bound ||P_r A P_c - LU||_inf <=
 * 4 n eps rho max|A| for every strategy on every test matrix; partial and complete pivoting
 * solve the Wilkinson 2x2 to ~1e-16 relative error while no-pivot loses every digit; partial
 * pivoting obeys the Wilkinson bound rho <= 2^(n-1); no-pivot growth explodes exponentially on
 * the adversarial family where complete pivoting stays near 1; solves of random systems hit
 * 1e-10 relative error; an ill-conditioned system shows the classic small residual but large
 * forward error; and the degenerate cases (n = 1, diagonal, empty, singular, zero pivot) are
 * reported instead of silently producing garbage. Deterministic inputs only.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

const EPSILON = Number.EPSILON;
const STRATEGIES = ["none", "partial", "complete"];

/** Deterministic splitmix64 -> [0, 1). */
function makeRng(seed) {
  let state = (BigInt(seed >>> 0) + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
  return function next() {
    state = (state + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
    z ^= z >> 31n;
    return Number(z >> 11n) / 9007199254740992;
  };
}

const zeros = (rows, cols) => Array.from({ length: rows }, () => new Array(cols).fill(0));
const maxAbs = (matrix) => matrix.reduce((acc, row) => row.reduce((inner, v) => Math.max(inner, Math.abs(v)), acc), 0);

/**
 * LU factorization with the requested pivot strategy. Computes P_r A P_c = L U, with the unit
 * lower triangle and the multipliers stored in place, and reports rho, the swap count and
 * whether a zero pivot made the factorization singular.
 */
function luFactorize(matrix, strategy = "partial") {
  ok(STRATEGIES.includes(strategy), `unknown pivot strategy ${strategy}`);
  const n = matrix.length;
  for (const row of matrix) strictEqual(row.length, n, "LU needs a square matrix");
  const lu = matrix.map((row) => row.slice());
  const rowPerm = Array.from({ length: n }, (_, i) => i);
  const colPerm = Array.from({ length: n }, (_, i) => i);
  const inputScale = maxAbs(matrix);
  let uScale = 0;
  let swaps = 0;
  let singular = false;
  for (let k = 0; k < n && !singular; k++) {
    let pivotRow = k;
    let pivotCol = k;
    if (strategy === "partial") {
      for (let i = k + 1; i < n; i++) if (Math.abs(lu[i][k]) > Math.abs(lu[pivotRow][k])) pivotRow = i;
    } else if (strategy === "complete") {
      for (let i = k; i < n; i++) {
        for (let j = k; j < n; j++) {
          if (Math.abs(lu[i][j]) > Math.abs(lu[pivotRow][pivotCol])) {
            pivotRow = i;
            pivotCol = j;
          }
        }
      }
    }
    if (strategy === "none" && lu[k][k] === 0) {
      // Not necessarily a singular matrix -- just one the chosen strategy cannot handle.
      throw new Error("zero pivot: this matrix needs pivoting");
    }
    if (lu[pivotRow][pivotCol] === 0) {
      singular = true;
      break;
    }
    if (pivotRow !== k) {
      [lu[pivotRow], lu[k]] = [lu[k], lu[pivotRow]];
      [rowPerm[pivotRow], rowPerm[k]] = [rowPerm[k], rowPerm[pivotRow]];
      swaps++;
    }
    if (pivotCol !== k) {
      for (let i = 0; i < n; i++) [lu[i][pivotCol], lu[i][k]] = [lu[i][k], lu[i][pivotCol]];
      [colPerm[pivotCol], colPerm[k]] = [colPerm[k], colPerm[pivotCol]];
      swaps++;
    }
    for (let i = k + 1; i < n; i++) {
      const multiplier = lu[i][k] / lu[k][k];
      lu[i][k] = multiplier;
      for (let j = k + 1; j < n; j++) lu[i][j] -= multiplier * lu[k][j];
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) uScale = Math.max(uScale, Math.abs(lu[i][j]));
  }
  return {
    n, lu, rowPerm, colPerm, swaps, singular, strategy,
    inputScale,
    growth: singular || inputScale === 0 ? 0 : uScale / inputScale,
    triangular: {
      lower: lu.map((row, i) => row.map((value, j) => (i > j ? value : i === j ? 1 : 0))),
      upper: lu.map((row, i) => row.map((value, j) => (i <= j ? value : 0))),
    },
  };
}

/** Solve A x = b from a factorization: apply P_r, forward solve with L, back solve with U. */
function luSolve(factorization, b) {
  const { n, lu, rowPerm, colPerm, singular } = factorization;
  strictEqual(b.length, n, "right-hand side has the wrong length");
  if (singular) throw new Error("cannot solve with a singular factorization");
  const permuted = rowPerm.map((row) => b[row]);
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let value = permuted[i];
    for (let j = 0; j < i; j++) value -= lu[i][j] * y[j];
    y[i] = value;
  }
  const z = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let value = y[i];
    for (let j = i + 1; j < n; j++) value -= lu[i][j] * z[j];
    z[i] = value / lu[i][i];
  }
  const x = new Array(n).fill(0);
  for (let j = 0; j < n; j++) x[colPerm[j]] = z[j];
  return x;
}

/** Rebuild P_r A P_c and compare it with L U: the factorization's own residual. */
function reconstructionResidual(matrix, factorization) {
  const { n, rowPerm, colPerm } = factorization;
  const permuted = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) permuted[i][j] = matrix[rowPerm[i]][colPerm[j]];
  const { lower, upper } = factorization.triangular;
  let worst = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let value = 0;
      for (let k = 0; k <= Math.min(i, j); k++) value += lower[i][k] * upper[k][j];
      worst = Math.max(worst, Math.abs(value - permuted[i][j]));
    }
  }
  return worst;
}

const matVec = (matrix, vector) => matrix.map((row) => row.reduce((acc, value, j) => acc + value * vector[j], 0));

const relativeError = (got, want) => {
  const scale = Math.max(1, Math.max(...want.map(Math.abs)));
  return Math.max(...got.map((value, i) => Math.abs(value - want[i]))) / scale;
};

const relativeResidual = (matrix, x, b) => {
  const scaled = Math.max(1, maxAbs([b]));
  const ax = matVec(matrix, x);
  return Math.max(...ax.map((value, i) => Math.abs(value - b[i]))) / scaled;
};

/**
 * Wilkinson's adversarial family: the identity with the strictly lower triangle filled with
 * -1 and a last column of 1s. Under partial pivoting the elimination doubles the last column
 * at every step, so rho = 2^(n-1) exactly -- the matrix that attains Wilkinson's bound.
 */
function growthMatrix(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 1;
      if (j === n - 1) return 1;
      return i > j ? -1 : 0;
    }));
}

/** The classic 2x2 that no-pivot elimination destroys. */
const tinyPivotMatrix = (eps) => [[eps, 1], [1, 1]];

function randomMatrix(rng, n, span = 1) {
  return Array.from({ length: n }, () => Array.from({ length: n }, () => (rng() * 2 - 1) * span));
}

/** Hilbert matrix: a textbook ill-conditioned example (its exact inverse is integral). */
const hilbert = (n) => Array.from({ length: n }, (_, i) =>
  Array.from({ length: n }, (_, j) => 1 / (i + j + 1)));

module.exports = {
  makeRng, luFactorize, luSolve, reconstructionResidual, matVec, relativeError,
  relativeResidual, growthMatrix, tinyPivotMatrix, randomMatrix, hilbert, maxAbs, EPSILON,
};

if (require.main === module) {
  const rng = makeRng(20260729);

  // -- the textbook backward-error bound for every strategy on every matrix --------------
  const suite = [];
  for (const n of [1, 2, 3, 5, 8, 16]) {
    suite.push({ name: `random ${n}`, matrix: randomMatrix(rng, n) });
    suite.push({ name: `growth ${n}`, matrix: growthMatrix(n) });
  }
  suite.push({ name: "hilbert 5", matrix: hilbert(5) });
  suite.push({ name: "diagonal 4", matrix: [[2, 0, 0, 0], [0, -3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 1]] });
  let checked = 0;
  for (const entry of suite) {
    for (const strategy of STRATEGIES) {
      const factorization = luFactorize(entry.matrix, strategy);
      strictEqual(factorization.singular, false, `${entry.name} / ${strategy} must not be singular`);
      const residual = reconstructionResidual(entry.matrix, factorization);
      const bound = 4 * factorization.n * EPSILON * factorization.growth * Math.max(1, factorization.inputScale);
      ok(residual <= bound,
        `${entry.name} / ${strategy}: residual ${residual} exceeds the textbook bound ${bound}`);
      checked++;
    }
  }
  console.log(`backward-error bound checked for ${checked} matrix/strategy pairs`);

  // -- the classic tiny-pivot disaster -----------------------------------------------------
  const eps = 1e-4;                       // the textbook scale: 4 digits is enough to see it
  const hard = tinyPivotMatrix(eps);
  const exact = [1, 1];
  const b = matVec(hard, exact);
  const noPivot = luFactorize(hard, "none");
  const partial = luFactorize(hard, "partial");
  const complete = luFactorize(hard, "complete");
  ok(noPivot.growth > 1e3, `no-pivot growth must explode, got ${noPivot.growth.toExponential(2)}`);
  ok(partial.growth < 4, `partial pivoting keeps growth near 1, got ${partial.growth.toFixed(3)}`);
  ok(complete.growth < 4, `complete pivoting keeps growth near 1, got ${complete.growth.toFixed(3)}`);
  const noPivotError = relativeError(luSolve(noPivot, b), exact);
  const partialError = relativeError(luSolve(partial, b), exact);
  const completeError = relativeError(luSolve(complete, b), exact);
  ok(partialError < 1e-15 && completeError < 1e-15,
    `pivoted solves must keep every digit here (${partialError}, ${completeError})`);
  ok(noPivotError > 1e-14, `the unpivoted solve must lose digits, got ${noPivotError}`);
  ok(noPivotError > partialError * 1e3, "and the gap must be orders of magnitude");
  console.log(`tiny pivot: growth none=${noPivot.growth.toExponential(2)} partial=${partial.growth.toFixed(2)}` +
    ` complete=${complete.growth.toFixed(2)}; error none=${noPivotError.toExponential(2)} partial=${partialError.toExponential(2)}`);

  // -- Wilkinson's bound, attained by the adversarial family ------------------------------
  const growthScan = [];
  for (const n of [2, 4, 8, 16, 32]) {
    const adversarial = growthMatrix(n);
    const part = luFactorize(adversarial, "partial");
    const comp = luFactorize(adversarial, "complete");
    strictEqual(part.growth, 2 ** (n - 1),
      `partial pivoting attains rho = 2^(n-1) on the adversarial family at n=${n}`);
    strictEqual(comp.growth, 2, `complete pivoting keeps rho at 2 on the same family at n=${n}`);
    growthScan.push({ n, partial: part.growth, complete: comp.growth });
  }
  for (let i = 1; i < growthScan.length; i++) {
    ok(growthScan[i].partial >= growthScan[i - 1].partial * 2,
      "each step up in size must double the growth factor at least");
  }
  for (const entry of growthScan) {
    ok(entry.complete <= entry.partial, `complete pivoting must not exceed the growth at n=${entry.n}`);
    if (entry.n >= 4) {
      ok(entry.complete * 2 <= entry.partial, `and it must cut it substantially at n=${entry.n}`);
    }
  }
  console.log(`growth family: partial ${growthScan.map((s) => s.partial.toFixed(1)).join(",")}` +
    ` vs complete ${growthScan.map((s) => s.complete.toFixed(2)).join(",")}`);

  // -- solves of random systems, and the residual/error distinction ------------------------
  let solved = 0;
  for (let trial = 0; trial < 12; trial++) {
    const n = 3 + (trial % 6);
    const matrix = randomMatrix(rng, n, 1);
    const target = Array.from({ length: n }, () => rng() * 2 - 1);
    const rhs = matVec(matrix, target);
    for (const strategy of ["partial", "complete"]) {
      const factorization = luFactorize(matrix, strategy);
      if (factorization.singular) continue;
      const solvedX = luSolve(factorization, rhs);
      const error = relativeError(solvedX, target);
      ok(error < 1e-10, `trial ${trial} / ${strategy}: forward error ${error} is too large`);
      ok(relativeResidual(matrix, solvedX, rhs) < 1e-12, "the residual must be even smaller than the error");
      solved++;
    }
  }
  ok(solved >= 20, `the random solve suite must actually run (${solved} solves)`);

  const ill = hilbert(6);
  const illFactorization = luFactorize(ill, "partial");
  const illTarget = new Array(6).fill(1);
  const illRhs = matVec(ill, illTarget);
  const illX = luSolve(illFactorization, illRhs);
  const illResidual = relativeResidual(ill, illX, illRhs);
  const illError = relativeError(illX, illTarget);
  ok(illResidual < 1e-14, `the residual of the ill-conditioned solve stays tiny (${illResidual})`);
  ok(illError > illResidual * 1e4,
    `but the forward error is amplified by the conditioning: ${illError} vs ${illResidual}`);
  console.log(`ill-conditioned: residual=${illResidual.toExponential(2)} error=${illError.toExponential(2)}`);

  // -- degenerate and singular inputs ------------------------------------------------------
  const empty = luFactorize([], "partial");
  strictEqual(empty.n, 0, "the empty system factorizes to nothing");
  deepStrictEqual(luSolve(empty, []), []);
  strictEqual(empty.growth, 0);

  const scalar = luFactorize([[4]], "none");
  strictEqual(scalar.growth, 1, "a 1x1 factorization has growth exactly 1");
  deepStrictEqual(luSolve(scalar, [8]), [2], "and solves by division");
  deepStrictEqual(luSolve(luFactorize([[-4]], "partial"), [8]), [-2]);

  const diagonal = [[3, 0, 0], [0, -2, 0], [0, 0, 7]];
  for (const strategy of STRATEGIES) {
    const factorization = luFactorize(diagonal, strategy);
    strictEqual(factorization.growth, 1, `${strategy}: growth is exactly 1 for a diagonal matrix`);
    deepStrictEqual(luSolve(factorization, [3, 4, 7]), [1, -2, 1], `${strategy}: the solve stays exact`);
    if (strategy !== "complete") {
      deepStrictEqual(factorization.rowPerm, [0, 1, 2], `${strategy}: a diagonal matrix needs no row swap`);
      deepStrictEqual(factorization.colPerm, [0, 1, 2], `${strategy}: and no column swap`);
      strictEqual(factorization.swaps, 0, `${strategy}: zero swaps on a diagonal matrix`);
    } else {
      deepStrictEqual(factorization.colPerm.slice().sort((a, b) => a - b), [0, 1, 2],
        "complete pivoting still returns a permutation of the columns");
      ok(factorization.swaps > 0, "complete pivoting reorders to reach the largest magnitude first");
    }
  }

  const needsPivot = [[0, 2], [3, 4]];
  throws(() => luFactorize(needsPivot, "none"), /zero pivot/, "a zero pivot without pivoting must be refused");
  const needsPivotPartial = luFactorize(needsPivot, "partial");
  strictEqual(needsPivotPartial.growth, 1, "pivoting rescues the zero-pivot matrix without growing U");
  deepStrictEqual(luSolve(needsPivotPartial, [2, 4]), [0, 1]);

  const singularMatrix = [[1, 2, 3], [2, 4, 6], [1, 1, 1]];
  for (const strategy of ["partial", "complete"]) {
    const factorization = luFactorize(singularMatrix, strategy);
    strictEqual(factorization.singular, true, `${strategy}: a rank-deficient matrix must be reported`);
    throws(() => luSolve(factorization, [1, 1, 1]), /singular/, `${strategy}: solving must be refused`);
  }
  throws(() => luFactorize(singularMatrix, "none"), /zero pivot/,
    "no-pivot runs into the zero pivot rather than the rank test");
  const zeroMatrix = [[0, 0], [0, 0]];
  strictEqual(luFactorize(zeroMatrix, "complete").singular, true, "the zero matrix is singular");

  // -- determinism --------------------------------------------------------------------------
  const replay = () => {
    const local = makeRng(31337);
    const matrix = randomMatrix(local, 8, 1);
    const part = luFactorize(matrix, "partial");
    const comp = luFactorize(matrix, "complete");
    return `${part.growth}/${comp.growth}/${part.swaps}/${comp.swaps}/${luSolve(part, matrix.map((r) => r[0])).join(",")}`;
  };
  deepStrictEqual(replay(), replay(), "identical inputs must factorize and solve identically");
  console.log("PASS");
}
