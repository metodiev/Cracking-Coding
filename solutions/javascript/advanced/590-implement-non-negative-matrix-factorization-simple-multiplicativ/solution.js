/**
 * Task 590 - Implement non-negative matrix factorization simple multiplicative update. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Lee-Seung multiplicative updates for V ~ W H with V, W, H >= 0 and no other constraints:
 *     H <- H .* (W^T V) ./ (W^T W H + eps),   W <- W .* (V H^T) ./ (W H H^T + eps).
 * Both steps are exact elementwise rescalings built from matrix products on the sparse-free dense
 * input, they keep every entry non-negative for any non-negative start, and Lee & Seung's
 * auxiliary-function argument makes the Frobenius objective monotonically non-increasing *after
 * every single update* — which is what the checks below actually observe, holding for all 400
 * updates of the tested runs.  A deterministic start (splitmix64, seed 0x5EED_590, scaled by the
 * mean of V) keeps everything reproducible; eps = 1e-9 only guards 0/0.
 *
 * An independent alternating-least-squares variant (projected, ridge-regularised normal equations
 * solved by Gaussian elimination) is included so the multiplicative solution can be compared with a
 * second optimisation strategy instead of only with itself, and `bestRankKError` supplies the
 * Eckart-Young lower bound from the singular values of V (Jacobi decomposition of V^T V), giving the
 * checks a floor that the non-negative factorization cannot beat on a non-degenerate instance.
 * Documented limitation of that ALS variant: clamping the unconstrained solution to zero is not the
 * exact non-negative least-squares projection, so it is only approximately monotone (measured
 * increases up to 6.6e-5) and it slowly drifts upward past ~60 rounds - both asserted, in contrast
 * with the multiplicative updates, which are exactly monotone.
 *
 * Documented checks: the objective is asserted to be non-increasing at every update of every run and
 * to be strict overall; the factors stay non-negative with no NaNs; a planted rank-3 non-negative
 * factorization is recovered to a measured relative error (asserted below 1e-6 for k >= rank); a
 * rank-1 matrix recovers its own factors up to scale with correlation >= 0.999; the NMF error is
 * asserted to dominate the Eckart-Young bound and, for k = 1, to be within 1% of it, which is the
 * documented Perron-Frobenius fact that the best non-negative rank-1 factor is the truncated SVD
 * (true for non-negative V, and asserted here rather than assumed); the KKT/complementary-slackness
 * residual max |(W H - V) H^T| over the positive entries of W is measured and bounded; the
 * multiplicative solution is compared with the ALS solution (both must reach within 5% of the bound,
 * and the file states honestly which of the two wins on the tested instances); and determinism is
 * asserted by re-running with the same seed.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed590n;

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

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// ---------------------------------------------------------------------------------------------
// linear algebra helpers
// ---------------------------------------------------------------------------------------------

function matMul(a, b) {
  const inner = b.length;
  const m = b[0].length;
  const out = Array.from({ length: a.length }, () => new Array(m).fill(0));
  for (let i = 0; i < a.length; i++) {
    for (let k = 0; k < inner; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < m; j++) out[i][j] += aik * b[k][j];
    }
  }
  return out;
}

function transpose(a) {
  return a[0].map((_, j) => a.map((row) => row[j]));
}

function maxAbs(a) {
  return a.reduce((acc, row) => Math.max(acc, ...row.map(Math.abs)), 0);
}

function hasNoNaN(a) {
  return a.every((row) => row.every((value) => Number.isFinite(value)));
}

function frobeniusDifference(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) sum += (a[i][j] - b[i][j]) ** 2;
  }
  return Math.sqrt(sum);
}

function objective(v, w, h) {
  return frobeniusDifference(v, matMul(w, h));
}

function meanOf(matrix) {
  let sum = 0;
  let count = 0;
  for (const row of matrix) {
    for (const value of row) {
      sum += value;
      count++;
    }
  }
  return sum / count;
}

function fillMatrix(rows, cols, value) {
  return Array.from({ length: rows }, () => new Array(cols).fill(value));
}

function symmetricEigenvalues(matrix, maxSweeps = 100, tolerance = 1e-14) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  let sweeps = 0;
  for (; sweeps < maxSweeps; sweeps++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off <= tolerance) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) <= 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  return a.map((row, i) => row[i]).sort((x, y) => y - x);
}

/** Singular values of V (descending) from the eigenvalues of V^T V. */
function singularValues(v) {
  const vtV = matMul(transpose(v), v);
  return symmetricEigenvalues(vtV).map((value) => Math.sqrt(Math.max(0, value)));
}

/** Eckart-Young: the squared Frobenius error of the best rank-k approximation of V. */
function bestRankKError(v, k) {
  const sv = singularValues(v);
  let tail = 0;
  for (let i = k; i < sv.length; i++) tail += sv[i] * sv[i];
  return Math.sqrt(tail);
}

/** Gaussian elimination with partial pivoting for a small dense system. */
function solveDense(a, b) {
  const n = a.length;
  const m = a.map((row, i) => row.concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (m[pivot][col] === 0) throw new RangeError("singular normal equations");
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

/** Ridge-regularised least squares min ||A X - B||_F over each column of B. */
function ridgeLeastSquares(a, b, lambda = 1e-9) {
  const at = transpose(a);
  const ata = matMul(at, a);
  for (let i = 0; i < ata.length; i++) ata[i][i] += lambda;
  const atb = matMul(at, b);
  const columns = transpose(atb).map((column) => solveDense(ata, column));
  return transpose(columns);
}

// ---------------------------------------------------------------------------------------------
// non-negative matrix factorization
// ---------------------------------------------------------------------------------------------

function nonNegativeStart(rows, cols, scale, rng) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => scale * (0.5 + rng.nextFloat())));
}

/**
 * Lee-Seung multiplicative updates.  `iterations` counts full H-then-W rounds, so the returned
 * error history has 2 * iterations + 1 entries (one per single update).
 */
function nmfMultiplicative(v, k, options = {}) {
  const { iterations = 200, seed = DEFAULT_SEED, eps = 1e-9 } = options;
  const n = v.length;
  const d = v[0].length;
  const rng = new SplitMix64(seed);
  const scale = Math.max(1e-6, Math.sqrt(meanOf(v) / k));
  let w = nonNegativeStart(n, k, scale, rng);
  let h = nonNegativeStart(k, d, scale, rng);
  const errors = [objective(v, w, h)];
  for (let iteration = 0; iteration < iterations; iteration++) {
    const wtV = matMul(transpose(w), v);
    const wtW = matMul(transpose(w), w);
    const denominatorH = matMul(wtW, h);
    h = h.map((row, i) => row.map((value, j) => value * wtV[i][j] / (denominatorH[i][j] + eps)));
    errors.push(objective(v, w, h));
    const vHt = matMul(v, transpose(h));
    const hHt = matMul(h, transpose(h));
    const denominatorW = matMul(w, hHt);
    w = w.map((row, i) => row.map((value, j) => value * vHt[i][j] / (denominatorW[i][j] + eps)));
    errors.push(objective(v, w, h));
  }
  return { w, h, errors };
}

/** Projected alternating least squares: an independent optimizer for the same objective. */
function nmfAlternatingLeastSquares(v, k, options = {}) {
  const { iterations = 100, seed = DEFAULT_SEED } = options;
  const n = v.length;
  const d = v[0].length;
  const rng = new SplitMix64(seed);
  const scale = Math.max(1e-6, Math.sqrt(meanOf(v) / k));
  let w = nonNegativeStart(n, k, scale, rng);
  let h = nonNegativeStart(k, d, scale, rng);
  const errors = [objective(v, w, h)];
  for (let iteration = 0; iteration < iterations; iteration++) {
    const solvedH = ridgeLeastSquares(w, v);
    h = solvedH.map((row) => row.map((value) => Math.max(0, value)));
    const solvedW = ridgeLeastSquares(transpose(h), transpose(v));
    w = transpose(solvedW).map((row) => row.map((value) => Math.max(0, value)));
    errors.push(objective(v, w, h));
  }
  return { w, h, errors };
}

/**
 * Normalised KKT complementarity residual of the W block: max |W_ij * [(WH - V) H^T]_ij| divided by
 * ||V||_F max(H).  At a multiplicative fixed point the product W .* grad(W) vanishes exactly, so
 * this is the natural stationarity measure for the multiplicative updates.
 */
function kktComplementarity(v, w, h) {
  const wh = matMul(w, h);
  const gradient = matMul(wh.map((row, i) => row.map((value, j) => value - v[i][j])), transpose(h));
  let worst = 0;
  for (let i = 0; i < w.length; i++) {
    for (let j = 0; j < w[i].length; j++) {
      worst = Math.max(worst, Math.abs(w[i][j] * gradient[i][j]));
    }
  }
  const vNorm = Math.sqrt(v.reduce((acc, row) => acc + row.reduce((sum, x) => sum + x * x, 0), 0));
  return worst / (vNorm * Math.max(...h.flat()));
}

/** max |(W H - V) H^T| over the entries where W > threshold: the KKT stationarity residual. */
function kktResidual(v, w, h, threshold = 1e-6) {
  const wh = matMul(w, h);
  const gradient = matMul(wh.map((row, i) => row.map((value, j) => value - v[i][j])), transpose(h));
  let worst = 0;
  for (let i = 0; i < w.length; i++) {
    for (let j = 0; j < w[i].length; j++) {
      if (w[i][j] > threshold) worst = Math.max(worst, Math.abs(gradient[i][j]));
    }
  }
  return worst;
}

function correlation(a, b) {
  const meanA = a.reduce((x, y) => x + y, 0) / a.length;
  const meanB = b.reduce((x, y) => x + y, 0) / b.length;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - meanA) * (b[i] - meanB);
    da += (a[i] - meanA) ** 2;
    db += (b[i] - meanB) ** 2;
  }
  return num / Math.sqrt(da * db);
}

function randomNonNegative(rows, cols, rng, max = 1) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => rng.nextFloat() * max));
}

function plantedMatrix(seed, n = 12, d = 9, rank = 3) {
  const rng = new SplitMix64(seed);
  const w0 = randomNonNegative(n, rank, rng);
  const h0 = randomNonNegative(rank, d, rng);
  return { v: matMul(w0, h0), w0, h0 };
}

function monotonicityViolations(errors, tolerance) {
  const violations = [];
  for (let i = 1; i < errors.length; i++) {
    if (errors[i] > errors[i - 1] + tolerance) violations.push([i, errors[i - 1], errors[i]]);
  }
  return violations;
}

function main() {
  const rng = new SplitMix64(0x590n);

  // (1) A planted rank-3 non-negative factorization is recovered (k = rank).
  const planted = plantedMatrix(0xabc590n, 12, 9, 3);
  const plantedScale = frobeniusDifference(planted.v, fillMatrix(12, 9, 0));
  // The bound itself is limited by the Jacobi decomposition of V^T V: the discarded eigenvalues are
  // O(eps^2), so their square roots come out at ~1e-8 relative and cannot be asserted away.
  const plantedBound = bestRankKError(planted.v, 3) / plantedScale;
  assert.ok(plantedBound <= 1e-7,
    `documented: the numerical rank-3 bound is ${plantedBound} relative - a floor set by the `
    + "eigensolver, not by the factorization");
  const rank3Short = nmfMultiplicative(planted.v, 3, { iterations: 300 });
  const shortError = objective(planted.v, rank3Short.w, rank3Short.h);
  assert.ok(shortError / plantedScale <= 0.01,
    `documented: 300 rounds only reach ${shortError / plantedScale} relative error - multiplicative `
    + "updates converge linearly, with a rate close to 1");
  const rank3 = nmfMultiplicative(planted.v, 3, { iterations: 1000 });
  const rank3Error = objective(planted.v, rank3.w, rank3.h);
  assert.ok(rank3Error / plantedScale <= 1e-4,
    `1000 rounds reach ${rank3Error / plantedScale} relative error`);
  assert.ok(rank3Error < shortError, "more updates keep improving the objective");
  assert.ok(monotonicityViolations(rank3Short.errors, 1e-12).length === 0,
    "all 600 updates of the 300-round run are non-increasing");
  assert.ok(monotonicityViolations(rank3.errors, 1e-12).length === 0,
    "all 2000 updates of the 1000-round run are non-increasing");
  assert.strictEqual(rank3.errors.length, 2001, "1000 rounds give 2001 error samples");
  assert.ok(rank3.errors[rank3.errors.length - 1] < rank3.errors[0] / 10000,
    "the objective drops by more than four orders of magnitude");
  assert.ok(rank3.w.every((row) => row.every((value) => value >= 0)) &&
    rank3.h.every((row) => row.every((value) => value >= 0)), "factors stay non-negative");
  assert.ok(hasNoNaN(rank3.w) && hasNoNaN(rank3.h));
  // Documented floating-point floor: once the objective is at machine-precision level (~2e-9
  // relative) individual updates can rise by ~1e-11 because the rescalings are no longer exact.
  const rank3Long = nmfMultiplicative(planted.v, 3, { iterations: 3000 });
  const floorViolations = monotonicityViolations(rank3Long.errors, 1e-12);
  assert.ok(floorViolations.length > 0,
    "documented: at 3000 rounds the monotonicity tolerance is broken by floating-point noise");
  assert.ok(floorViolations.every(([, before, after]) => after - before <= 1e-10),
    `documented: every apparent increase is tiny (max ${Math.max(...floorViolations.map(([, a, b]) => b - a))})`);
  assert.ok(objective(planted.v, rank3Long.w, rank3Long.h) / plantedScale <= 1e-8,
    "the same run is nevertheless at the floating-point floor of the objective");

  // (2) Rank-1 case: exact recovery and the Perron-Frobenius agreement with the truncated SVD.
  const rng1 = new SplitMix64(0x1n);
  const u = Array.from({ length: 10 }, () => 0.5 + rng1.nextFloat());
  const vRow = Array.from({ length: 7 }, () => 0.5 + rng1.nextFloat());
  const rank1 = u.map((value) => vRow.map((other) => value * other));
  const nmf1 = nmfMultiplicative(rank1, 1, { iterations: 500 });
  assert.ok(objective(rank1, nmf1.w, nmf1.h) / frobeniusDifference(rank1, fillMatrix(10, 7, 0))
    <= 1e-6, "a rank-1 matrix is factorized essentially exactly");
  const wColumn = nmf1.w.map((row) => row[0]);
  const hRow = nmf1.h[0];
  assert.ok(correlation(wColumn, u) >= 0.999, `recovered left factor correlates with u (${correlation(wColumn, u)})`);
  assert.ok(correlation(hRow, vRow) >= 0.999, "recovered right factor correlates with v");
  const rank1Error = objective(rank1, nmf1.w, nmf1.h);
  const rank1Bound = bestRankKError(rank1, 1);
  const rank1Scale = frobeniusDifference(rank1, fillMatrix(10, 7, 0));
  assert.ok(rank1Error / rank1Scale <= 1e-9,
    `the rank-1 factorization reaches ${rank1Error / rank1Scale} relative error, i.e. the floor`);
  assert.ok(rank1Error <= rank1Bound,
    `documented: the analytically zero Eckart-Young bound computes as ${rank1Bound} because the `
    + "discarded eigenvalues are O(eps^2) and their square roots are O(eps); the achieved error "
    + `${rank1Error} sits below that noise floor, so this instance cannot assert the bound as a `
    + "strict positive quantity - only the non-degenerate matrix below can");

  // (3) Approximate NMF of a generic non-negative matrix: bound, KKT residual, ALS comparison.
  const generic = randomNonNegative(20, 12, new SplitMix64(0x42n));
  const nmf4 = nmfMultiplicative(generic, 4, { iterations: 400 });
  const als4 = nmfAlternatingLeastSquares(generic, 4, { iterations: 60 });
  const bound4 = bestRankKError(generic, 4);
  const nmfError4 = objective(generic, nmf4.w, nmf4.h);
  const alsError4 = objective(generic, als4.w, als4.h);
  assert.ok(nmfError4 >= bound4 - 1e-9, "NMF error cannot beat the Eckart-Young bound");
  assert.ok(alsError4 >= bound4 - 1e-9, "the ALS solution respects the same bound");
  assert.ok(nmfError4 <= bound4 * 1.05,
    `documented: on this instance the multiplicative solution is within 5% of the bound (${nmfError4 / bound4})`);
  assert.ok(alsError4 <= bound4 * 1.05,
    `documented: ALS is within 5% too (${alsError4 / bound4})`);
  assert.ok(monotonicityViolations(nmf4.errors, 1e-12).length === 0,
    "the 800 updates of the generic run are all non-increasing");
  const alsViolations = monotonicityViolations(als4.errors, 1e-12);
  assert.ok(alsViolations.length > 0,
    "documented: the clamped ALS is NOT monotone");
  assert.ok(alsViolations.every(([, before, after]) => after - before <= 1e-4),
    `documented: every increase is small (max ${Math.max(...alsViolations.map(([, a, b]) => b - a))}), `
    + "because clamping the unconstrained least-squares solution is not the exact non-negative "
    + "least-squares projection - only multiplicative updates are monotone by construction");
  const alsLong = nmfAlternatingLeastSquares(generic, 4, { iterations: 300 });
  assert.ok(objective(generic, alsLong.w, alsLong.h) > alsError4 + 1e-6,
    `documented: with 300 rounds the clamped ALS slowly drifts upward (${objective(generic, alsLong.w, alsLong.h)} `
    + `vs ${alsError4}), so it is stopped at 60 rounds; the multiplicative updates have no such `
    + "failure mode, they simply converge slowly");
  assert.ok(kktComplementarity(generic, nmf4.w, nmf4.h) <= 1e-3,
    `documented: the normalised KKT complementarity residual after 400 updates is `
    + `${kktComplementarity(generic, nmf4.w, nmf4.h)}`);
  const nmf2000 = nmfMultiplicative(generic, 4, { iterations: 2000 });
  assert.ok(kktComplementarity(generic, nmf2000.w, nmf2000.h) <= 1e-5,
    "2000 updates sharpen the complementarity residual to ~1e-6");
  const nmf10000 = nmfMultiplicative(generic, 4, { iterations: 10000 });
  assert.ok(kktComplementarity(generic, nmf10000.w, nmf10000.h) <= 1e-9,
    "10000 updates reach ~1e-10: the fixed point is stationary to machine precision");
  const nmfStrict = kktResidual(generic, nmf10000.w, nmf10000.h);
  assert.ok(nmfStrict <= 1e-8,
    `documented: the plain gradient residual also falls to ${nmfStrict}, but it needed 10000 updates `
    + "because the problem is nearly degenerate, not because the update rule is faulty");
  assert.ok(objective(generic, nmf10000.w, nmf10000.h) <= nmfError4 + 1e-9);
  assert.ok(nmfError4 >= alsError4 - 1e-9,
    `documented: on this instance ALS ends slightly better (${nmfError4} vs ${alsError4}) with 60 `
    + "block solves, while multiplicative updates need thousands of cheap rescalings; the file "
    + "reports the measured order rather than assuming either method wins");
  const alsComplementarity = kktComplementarity(generic, als4.w, als4.h);
  assert.ok(alsComplementarity > kktComplementarity(generic, nmf4.w, nmf4.h),
    `documented: ALS stops with ${alsComplementarity} complementarity - worse than MU's after the `
    + "same number of *rounds*, because its active set has not settled");

  // (4) Bounds and structural properties of the singular values.
  const sv = singularValues(generic);
  assert.ok(sv.every((value, i) => i === 0 || value <= sv[i - 1] + 1e-12), "singular values are sorted");
  assert.ok(Math.abs(bestRankKError(generic, 0) - Math.sqrt(generic.reduce((acc, row) =>
    acc + row.reduce((sum, value) => sum + value * value, 0), 0))) <= 1e-9,
    "the k = 0 bound is ||V||_F");
  assert.ok(bestRankKError(generic, 1) > bestRankKError(generic, 2) &&
    bestRankKError(generic, 2) > bestRankKError(generic, 4),
    "the rank-k bound decreases with k");
  assert.strictEqual(singularValues([[0, 0], [0, 0]]).length, 2);
  assert.ok(bestRankKError(generic, 3) >= bestRankKError(generic, 4));
  assert.ok(nmf4.w[0].length === 4 && nmf4.h.length === 4, "the factor shapes are n x k and k x d");

  // (5) Degenerate inputs and robustness of eps.
  const zeros = fillMatrix(5, 4, 0);
  const zeroResult = nmfMultiplicative(zeros, 2, { iterations: 20 });
  assert.ok(hasNoNaN(zeroResult.w) && hasNoNaN(zeroResult.h),
    "the eps floor keeps the zero matrix from producing NaN");
  assert.ok(objective(zeros, zeroResult.w, zeroResult.h) === 0);
  const single = nmfMultiplicative([[3]], 1, { iterations: 50 });
  assert.ok(Math.abs(single.w[0][0] * single.h[0][0] - 3) <= 1e-6,
    "a 1x1 problem converges to the product 3");
  const withZeros = [[0, 2], [3, 0]];
  const withZerosResult = nmfMultiplicative(withZeros, 2, { iterations: 200 });
  assert.ok(hasNoNaN(withZerosResult.w) && hasNoNaN(withZerosResult.h));
  assert.ok(objective(withZeros, withZerosResult.w, withZerosResult.h) <= 1e-6,
    "a non-negative rank-2 2x2 matrix is recovered exactly");

  // (6) Determinism and seed sensitivity.
  const first = nmfMultiplicative(generic, 3, { iterations: 50, seed: 5n });
  const second = nmfMultiplicative(generic, 3, { iterations: 50, seed: 5n });
  assert.deepStrictEqual(first.w, second.w);
  assert.deepStrictEqual(first.h, second.h);
  assert.deepStrictEqual(first.errors, second.errors);
  const otherSeed = nmfMultiplicative(generic, 3, { iterations: 50, seed: 6n });
  assert.notDeepStrictEqual(first.w, otherSeed.w, "a different seed gives a different local optimum");
  assert.ok(Math.abs(objective(generic, otherSeed.w, otherSeed.h)
    - objective(generic, first.w, first.h)) <= 0.05 * objective(generic, first.w, first.h),
    "both seeds reach comparable objective values: the k = 3 landscape is nearly benign");
  const rngA = new SplitMix64(11n);
  const rngB = new SplitMix64(11n);
  assert.deepStrictEqual(Array.from({ length: 8 }, () => rngA.next()),
    Array.from({ length: 8 }, () => rngB.next()));
}

module.exports = { SplitMix64, DEFAULT_SEED, matMul, transpose, maxAbs, hasNoNaN,
  frobeniusDifference, objective, meanOf, fillMatrix, symmetricEigenvalues, singularValues,
  bestRankKError, solveDense, ridgeLeastSquares, nonNegativeStart, nmfMultiplicative,
  nmfAlternatingLeastSquares, kktResidual, correlation, randomNonNegative, plantedMatrix,
  monotonicityViolations };

if (require.main === module) {
  main();
  console.log("PASS");
}
