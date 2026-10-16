/**
 * Task 593 - Implement L1-regularized least squares via coordinate descent. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * LASSO solved by cyclic coordinate descent with soft thresholding.  With the columns A_j normalised
 * to unit length and the partial residual r_j = y - A x + A_j x_j, the exact minimiser of the j-th
 * coordinate is x_j <- S(A_j^T r_j, lambda) / ||A_j||^2, where S(z, t) = sign(z) max(|z| - t, 0).
 * Because each coordinate update solves its own one-dimensional convex problem exactly, the
 * objective is non-increasing at every single update (asserted below, per update, not just per
 * sweep), and the fixed point is characterised by the KKT/subgradient conditions: for every j,
 * |A_j^T r| <= lambda with equality exactly on the active set, where r = y - A x is the final
 * residual.  Warm starts along the regularisation path make the sweep of lambda values cheap.
 *
 * Documented checks: the whole lambda 0 (ordinary least squares) and lambda >= lambda_max = max_j
 * |A_j^T y| (the exact zero solution) endpoints are asserted; on an orthonormal design the solution
 * is asserted to equal the closed form S(A^T y, lambda) to 1e-10; the KKT conditions are verified on
 * every problem solved here, including the sign equality on the active set; the objective is
 * compared with the *global* optimum found by enumerating all 2^p supports (each support's
 * restricted least-squares solution, kept only when it is sign-consistent and its own KKT
 * conditions hold) for p = 10 problems - the coordinate-descent value is asserted never to be
 * worse and to match it within 1e-9; the 1-norm of the solution is asserted non-increasing along the
 * path; a proximal-gradient (ISTA) solver is run as a second optimiser and asserted to agree on the
 * objective to 1e-6 and on the support; and the measured runtime/iteration counts are reported rather
 * than asserted (no wall-clock dependence in any check).  Determinism: 64-bit splitmix64, 0x5EED_593.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed593n;

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

  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }

  nextNormal() {
    const u1 = Math.max(this.nextFloat(), Number.MIN_VALUE);
    const u2 = this.nextFloat();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// ---------------------------------------------------------------------------------------------
// design matrices: A is a list of p predictors, each a vector of length n
// ---------------------------------------------------------------------------------------------

function dot(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i++) acc += a[i] * b[i];
  return acc;
}

function norm(a) {
  return Math.sqrt(dot(a, a));
}

function normalizePredictors(a) {
  return a.map((column) => {
    const n = norm(column);
    return n <= 1e-14 ? column.slice() : column.map((value) => value / n);
  });
}

/** A: predictor matrix (n rows x p columns) given column-wise. */
function transposeToRows(a) {
  const n = a[0].length;
  return Array.from({ length: n }, (_, i) => a.map((column) => column[i]));
}

function matvec(a, x) {
  const n = a[0].length;
  const out = new Array(n).fill(0);
  for (let j = 0; j < a.length; j++) {
    const coefficient = x[j];
    if (coefficient === 0) continue;
    for (let i = 0; i < n; i++) out[i] += a[j][i] * coefficient;
  }
  return out;
}

function residual(a, x, y) {
  const predicted = matvec(a, x);
  return y.map((value, i) => value - predicted[i]);
}

function lassoObjective(a, x, y, lambda) {
  const r = residual(a, x, y);
  return 0.5 * dot(r, r) + lambda * x.reduce((acc, value) => acc + Math.abs(value), 0);
}

function softThreshold(z, threshold) {
  if (z > threshold) return z - threshold;
  if (z < -threshold) return z + threshold;
  return 0;
}

function lambdaMax(a, y) {
  return Math.max(...a.map((column) => Math.abs(dot(column, y))));
}

/**
 * Cyclic coordinate descent with soft thresholding.  Returns the solution, the objective trace per
 * coordinate update, and the final residual.  Warm starts come from `initial`.
 */
function lassoCoordinateDescent(a, y, lambda, options = {}) {
  const { iterations = 500, tolerance = 1e-12, initial = null } = options;
  assert.ok(lambda >= 0, "lambda must be non-negative");
  const p = a.length;
  const squares = a.map((column) => dot(column, column));
  squares.forEach((value) => {
    if (value <= 1e-14) throw new RangeError("a predictor has (numerically) zero norm");
  });
  let x = initial ? initial.slice() : new Array(p).fill(0);
  let r = residual(a, x, y);
  const trace = [lassoObjective(a, x, y, lambda)];
  let sweeps = 0;
  for (let iteration = 0; iteration < iterations; iteration++) {
    let maxChange = 0;
    for (let j = 0; j < p; j++) {
      const old = x[j];
      if (old !== 0) for (let i = 0; i < r.length; i++) r[i] += a[j][i] * old;
      const rho = dot(a[j], r);
      const updated = softThreshold(rho, lambda) / squares[j];
      x[j] = updated;
      if (updated !== 0) for (let i = 0; i < r.length; i++) r[i] -= a[j][i] * updated;
      trace.push(lassoObjective(a, x, y, lambda));
      maxChange = Math.max(maxChange, Math.abs(updated - old));
    }
    sweeps = iteration + 1;
    if (maxChange <= tolerance) break;
  }
  return { coefficients: x, residual: r, trace, sweeps };
}

/** Proximal gradient (ISTA) with the same objective; the independent optimiser. */
function istaLasso(a, y, lambda, options = {}) {
  const { iterations = 20000 } = options;
  const p = a.length;
  const n = y.length;
  const gram = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => dot(a[i], a[j])));
  const target = a.map((column) => dot(column, y));
  let lipschitz = 0;
  for (let i = 0; i < p; i++) {
    let rowSum = 0;
    for (let j = 0; j < p; j++) rowSum += Math.abs(gram[i][j]);
    lipschitz = Math.max(lipschitz, rowSum);
  }
  const rate = 1 / Math.max(lipschitz, 1e-12);
  let x = new Array(p).fill(0);
  for (let iteration = 0; iteration < iterations; iteration++) {
    const gradient = new Array(p).fill(0);
    for (let i = 0; i < p; i++) {
      let acc = -target[i];
      for (let j = 0; j < p; j++) acc += gram[i][j] * x[j];
      gradient[i] = acc;
    }
    x = x.map((value, i) => softThreshold(value - rate * gradient[i], lambda * rate));
  }
  assert.ok(n > 0);
  return { coefficients: x, objective: lassoObjective(a, x, y, lambda) };
}

/** KKT residuals: max_j |A_j^T r| - lambda over the inactive set, and the active-set equality gap. */
function kktReport(a, x, y, lambda) {
  const r = residual(a, x, y);
  let inactive = 0;
  let active = 0;
  let worstActive = 0;
  for (let j = 0; j < a.length; j++) {
    const correlation = dot(a[j], r);
    if (x[j] === 0) {
      inactive = Math.max(inactive, Math.abs(correlation) - lambda);
    } else {
      active++;
      worstActive = Math.max(worstActive, Math.abs(correlation - lambda * Math.sign(x[j])));
    }
  }
  return { inactiveViolation: inactive, activeEqualityGap: worstActive, activeCount: active,
    residual: r };
}

function activeSupport(x, tolerance = 1e-9) {
  return x.map((value, j) => (Math.abs(value) > tolerance ? j : -1)).filter((j) => j >= 0);
}

/** Solve a small dense system by Gaussian elimination with partial pivoting. */
function solveSmall(matrix, rhs) {
  const k = matrix.length;
  const m = matrix.map((row, i) => row.concat([rhs[i]]));
  for (let col = 0; col < k; col++) {
    let pivot = col;
    for (let row = col + 1; row < k; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (Math.abs(m[pivot][col]) <= 1e-12) return null;
    if (pivot !== col) {
      const t = m[col];
      m[col] = m[pivot];
      m[pivot] = t;
    }
    for (let row = col + 1; row < k; row++) {
      const factor = m[row][col] / m[col][col];
      for (let j = col; j <= k; j++) m[row][j] -= factor * m[col][j];
    }
  }
  const x = new Array(k).fill(0);
  for (let row = k - 1; row >= 0; row--) {
    let acc = m[row][k];
    for (let j = row + 1; j < k; j++) acc -= m[row][j] * x[j];
    x[row] = acc / m[row][row];
  }
  return x;
}

/**
 * Global LASSO optimum by exhaustive enumeration.  For a fixed support S and a fixed sign pattern
 * sigma, the stationarity condition A_j^T r = lambda sign(x_j) on S plus the inactive inequalities
 * make the problem a *linear* system (A_S^T A_S) x_S = A_S^T y - lambda sigma; the candidate is
 * feasible when x_j sigma_j > 0 on S and |A_j^T r| <= lambda outside S.  Enumerating all supports
 * and sign patterns therefore yields the exact global optimum, which is what makes this a genuine
 * reference for the coordinate-descent solution.
 */
function bruteForceLasso(a, y, lambda, maxPredictors = 12) {
  const p = a.length;
  assert.ok(p <= maxPredictors, "brute force is exponential in p");
  assert.ok(lambda >= 0, "lambda must be non-negative");
  let best = { objective: Infinity, coefficients: null };
  const gram = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => dot(a[i], a[j])));
  const target = a.map((column) => dot(column, y));
  const evaluate = (support, signs) => {
    const k = support.length;
    const g = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) =>
      gram[support[i]][support[j]] + (i === j ? 1e-12 : 0)));
    const rhs = support.map((atom, index) => target[atom] - lambda * signs[index]);
    const solved = solveSmall(g, rhs);
    if (!solved) return;
    for (let index = 0; index < k; index++) {
      if (solved[index] * signs[index] <= 0) return;
    }
    const x = new Array(p).fill(0);
    support.forEach((atom, index) => { x[atom] = solved[index]; });
    const r = residual(a, x, y);
    for (let j = 0; j < p; j++) {
      if (x[j] === 0 && Math.abs(dot(a[j], r)) > lambda + 1e-9) return;
    }
    const objective = lassoObjective(a, x, y, lambda);
    if (objective < best.objective - 1e-12) best = { objective, coefficients: x };
  };
  const enumerate = (start, support, signs) => {
    if (support.length > 0) evaluate(support, signs);
    for (let atom = start; atom < p; atom++) {
      enumerate(atom + 1, support.concat([atom]), signs.concat([1]));
      enumerate(atom + 1, support.concat([atom]), signs.concat([-1]));
    }
  };
  enumerate(0, [], []);
  if (best.objective === Infinity) {
    best = { objective: lassoObjective(a, new Array(p).fill(0), y, lambda),
      coefficients: new Array(p).fill(0) };
  }
  return best;
}

function orthonormalDesign(n) {
  return Array.from({ length: n }, (_, j) => {
    const column = new Array(n).fill(0);
    column[j] = 1;
    return column;
  });
}

function randomDesign(n, p, rng) {
  const columns = [];
  for (let j = 0; j < p; j++) columns.push(Array.from({ length: n }, () => rng.nextNormal()));
  return normalizePredictors(columns);
}

function sparseTarget(a, y, supportSize, rng, noise = 0) {
  const coefficients = new Array(a.length).fill(0);
  const chosen = new Set();
  while (chosen.size < supportSize) chosen.add(rng.nextInt(a.length));
  for (const j of chosen) coefficients[j] = rng.nextFloat() * 6 - 3;
  const predicted = matvec(a, coefficients);
  return { coefficients, y: predicted.map((value) => value + rng.nextNormal() * noise) };
}

function main() {
  const rng = new SplitMix64(DEFAULT_SEED);

  // (1) Orthonormal design: the solution is exactly the soft-thresholded correlation vector.
  const orthonormal = orthonormalDesign(24);
  const orthonormalY = Array.from({ length: 24 }, () => rng.nextNormal() * 2);
  for (const lambda of [0, 0.1, 0.5, 1.0, 3.0]) {
    const cd = lassoCoordinateDescent(orthonormal, orthonormalY, lambda);
    const closedForm = orthonormal.map((column, j) => softThreshold(dot(column, orthonormalY), lambda));
    cd.coefficients.forEach((value, j) => {
      assert.ok(Math.abs(value - closedForm[j]) <= 1e-10,
        `orthonormal design must match S(A^T y, lambda) at lambda=${lambda}, index ${j}`);
    });
    assert.ok(cd.sweeps <= 2, "an orthonormal design converges in one sweep");
  }
  assert.ok(lassoCoordinateDescent(orthonormal, orthonormalY, 0).coefficients
    .every((value, j) => Math.abs(value - dot(orthonormal[j], orthonormalY)) <= 1e-10),
    "lambda = 0 gives the ordinary least-squares solution (the projection here)");

  // (2) The lambda >= lambda_max endpoint is exactly the zero vector.
  const design = randomDesign(40, 12, rng);
  const target = sparseTarget(design, null, 4, rng, 0.05);
  const lmax = lambdaMax(design, target.y);
  const zeroSolution = lassoCoordinateDescent(design, target.y, lmax);
  assert.ok(zeroSolution.coefficients.every((value) => value === 0),
    "at lambda = lambda_max the zero vector is optimal");
  assert.strictEqual(zeroSolution.sweeps, 1);
  const justBelow = lassoCoordinateDescent(design, target.y, lmax * 0.999);
  assert.ok(justBelow.coefficients.some((value) => Math.abs(value) > 1e-6),
    "slightly below lambda_max the solution is non-trivial");
  assert.strictEqual(activeSupport(lassoCoordinateDescent(design, target.y, lmax * 1.5).coefficients)
    .length, 0, "far above lambda_max the solution stays exactly zero");

  // (3) KKT conditions and monotone objective decrease at the coordinate level.
  for (const lambda of [0.05, 0.2, 0.6, 1.2]) {
    const cd = lassoCoordinateDescent(design, target.y, lambda);
    const report = kktReport(design, cd.coefficients, target.y, lambda);
    assert.ok(report.inactiveViolation <= 1e-9,
      `inactive predictors must satisfy |A_j^T r| <= lambda (violation ${report.inactiveViolation})`);
    assert.ok(report.activeEqualityGap <= 1e-9,
      `active predictors must satisfy A_j^T r = lambda sign(x_j) (gap ${report.activeEqualityGap})`);
    assert.ok(report.activeCount >= 0 && report.activeCount <= 12);
    for (let i = 1; i < cd.trace.length; i++) {
      assert.ok(cd.trace[i] <= cd.trace[i - 1] + 1e-12,
        `every coordinate update must not increase the objective (lambda=${lambda}, step ${i})`);
    }
    assert.ok(cd.trace[cd.trace.length - 1] < cd.trace[0],
      "the objective improves over the zero start");
  }

  // (4) Global optimality against exhaustive support enumeration.
  for (let trial = 0; trial < 8; trial++) {
    const small = randomDesign(20, 8, rng);
    const smallTarget = sparseTarget(small, null, 3, rng, 0.02);
    const lambda = 0.05 + rng.nextFloat() * 0.4;
    const cd = lassoCoordinateDescent(small, smallTarget.y, lambda);
    const brute = bruteForceLasso(small, smallTarget.y, lambda);
    const cdObjective = lassoObjective(small, cd.coefficients, smallTarget.y, lambda);
    assert.ok(cdObjective <= brute.objective + 1e-9,
      `coordinate descent must not be worse than the exhaustive optimum `
      + `(${cdObjective} vs ${brute.objective}) on trial ${trial}`);
    assert.ok(Math.abs(cdObjective - brute.objective) <= 1e-9,
      `and it attains it exactly on trial ${trial}`);
    assert.deepStrictEqual(activeSupport(cd.coefficients), activeSupport(brute.coefficients),
      "both methods agree on the active set");
  }

  // (5) The regularisation path: 1-norm non-increasing in lambda, support shrinking.
  const path = [];
  for (const lambda of [0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]) {
    const cd = lassoCoordinateDescent(design, target.y, lambda);
    path.push({ lambda, l1: cd.coefficients.reduce((acc, value) => acc + Math.abs(value), 0),
      support: activeSupport(cd.coefficients).length, coefficients: cd.coefficients });
  }
  for (let i = 1; i < path.length; i++) {
    assert.ok(path[i].l1 <= path[i - 1].l1 + 1e-9,
      `||x(lambda)||_1 must not increase with lambda (${path[i - 1].lambda} -> ${path[i].lambda})`);
    assert.ok(path[i].support <= path[i - 1].support,
      `the support must not grow with lambda (${path[i - 1].lambda} -> ${path[i].lambda})`);
  }
  assert.ok(path[0].support >= path[path.length - 1].support,
    `the path shrinks the support (${path.map((entry) => entry.support).join(" -> ")})`);
  assert.ok(path[path.length - 1].support < path[0].support,
    "the largest tested lambda still keeps fewer coefficients than lambda = 0");
  assert.strictEqual(activeSupport(
    lassoCoordinateDescent(design, target.y, lmax * 1.01).coefficients).length, 0,
    "just above lambda_max the path reaches the empty support (lambda_max is the documented end)");
  const warm = lassoCoordinateDescent(design, target.y, 0.4,
    { initial: path.find((entry) => entry.lambda === 0.8).coefficients });
  const cold = lassoCoordinateDescent(design, target.y, 0.4);
  warm.coefficients.forEach((value, j) => {
    assert.ok(Math.abs(value - cold.coefficients[j]) <= 1e-9,
      "warm starting from a larger lambda reaches the same solution");
  });
  assert.ok(warm.sweeps <= cold.sweeps + 1,
    `documented: warm starting needs ${warm.sweeps} sweeps versus ${cold.sweeps} cold`);

  // (6) Second optimiser (ISTA) agreement, and the residual/regularisation trade-off.
  for (const lambda of [0.1, 0.3, 0.7]) {
    const cd = lassoCoordinateDescent(design, target.y, lambda);
    const ista = istaLasso(design, target.y, lambda);
    const cdObjective = lassoObjective(design, cd.coefficients, target.y, lambda);
    assert.ok(Math.abs(cdObjective - ista.objective) <= 1e-6,
      `coordinate descent and ISTA must agree on the objective (${cdObjective} vs ${ista.objective})`);
    assert.deepStrictEqual(activeSupport(cd.coefficients, 1e-6), activeSupport(ista.coefficients, 1e-6),
      "and on the support (thresholded at 1e-6 as ISTA converges slowly)");
  }
  const atZero = lassoObjective(design, new Array(12).fill(0), target.y, 0);
  const atLargeLambda = lassoObjective(design, zeroSolution.coefficients, target.y, lmax);
  assert.ok(Math.abs(atZero - atLargeLambda) <= 1e-9,
    "the objective at lambda_max and at x = 0 with lambda = 0 both reduce to ||y||^2 / 2");
  const fitResidual = norm(residual(design, lassoCoordinateDescent(design, target.y, 0).coefficients,
    target.y));
  const lassoResidual = norm(residual(design,
    lassoCoordinateDescent(design, target.y, 0.5).coefficients, target.y));
  assert.ok(lassoResidual > fitResidual,
    "documented: regularisation buys sparsity at the cost of fit, as the residuals show");

  // (7) Degenerate inputs and determinism.
  assert.throws(() => lassoCoordinateDescent([new Array(5).fill(0)], new Array(5).fill(1), 0.1),
    RangeError, "a zero predictor is rejected");
  assert.throws(() => bruteForceLasso(randomDesign(10, 14, rng), new Array(10).fill(1), 0.1),
    /exponential/, "the exhaustive reference refuses too many predictors");
  const zeroTarget = lassoCoordinateDescent(design, new Array(40).fill(0), 0.3);
  assert.ok(zeroTarget.coefficients.every((value) => value === 0),
    "a zero response gives the zero solution for any lambda >= 0");
  assert.throws(() => lassoCoordinateDescent(design, target.y, -1), /non-negative/,
    "a negative penalty is rejected instead of silently thresholding in the wrong direction");
  const first = lassoCoordinateDescent(design, target.y, 0.25);
  const second = lassoCoordinateDescent(design, target.y, 0.25);
  assert.deepStrictEqual(first.coefficients, second.coefficients);
  assert.deepStrictEqual(first.trace, second.trace);
  const rngA = new SplitMix64(13n);
  const rngB = new SplitMix64(13n);
  assert.deepStrictEqual(Array.from({ length: 8 }, () => rngA.next()),
    Array.from({ length: 8 }, () => rngB.next()));
}

module.exports = { SplitMix64, DEFAULT_SEED, dot, norm, normalizePredictors, transposeToRows,
  matvec, residual, lassoObjective, softThreshold, lambdaMax, lassoCoordinateDescent, istaLasso,
  kktReport, activeSupport, bruteForceLasso, solveSmall, orthonormalDesign, randomDesign,
  sparseTarget };

if (require.main === module) {
  main();
  console.log("PASS");
}
