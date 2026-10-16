/** Task 730 - Implement randomized numerical linear algebra low-rank approx demo. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * The randomized range finder (Halko, Martinsson, Tropp) computes a near-optimal rank-k
 * approximation with two matrix products and no pivoting:
 *
 *   Omega = random n x (k + p)        Y = (A A^T)^q A Omega      Q = orth(Y)     B = Q^T A
 *
 * Q spans a subspace that captures almost all the action of A, so ||A - Q Q^T A||_F lands
 * within a small factor of the *optimal* rank-k error ||A - A_k||_F = sqrt(sum_{i>k} sigma_i^2)
 * -- and with a couple of extra "power iterations" (q >= 1) the factor collapses towards 1 even
 * for slowly decaying spectra.
 *
 * This demo implements the range finder, modified Gram-Schmidt with re-orthogonalization, a
 * one-sided Jacobi SVD (the exact oracle for the optimal error), and the verification that the
 * randomized error is close to optimal, that Q's columns are orthonormal, and that the
 * Pythagorean identity ||A||_F^2 = ||Q^T A||_F^2 + ||A - Q Q^T A||_F^2 holds exactly.
 *
 * Verified here: the Jacobi SVD reconstructs A and matches the Frobenius norm; the optimal
 * rank-k error equals the closed form in the singular values; a rank-r input with k >= r is
 * recovered to machine precision; for k < r the randomized error stays within a measured factor
 * of the optimum and extra power iterations shrink it; the orthonormality and Pythagorean
 * checks hold; and the degenerate inputs (k = 0, k >= min(m,n), zero matrix, single column) are
 * handled. All randomness comes from a seeded splitmix64 + Box-Muller generator.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

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

/** Box-Muller: deterministic standard normal samples from the uniform stream. */
function gaussian(rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const zeros = (rows, cols) => Array.from({ length: rows }, () => new Array(cols).fill(0));
const identity = (n) => Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => (r === c ? 1 : 0)));
const transpose = (matrix) => (matrix.length === 0 ? [] : matrix[0].map((_, c) => matrix.map((row) => row[c])));

function matMul(a, b) {
  const rows = a.length;
  const inner = rows === 0 ? 0 : a[0].length;
  const cols = b.length === 0 ? 0 : b[0].length;
  const out = zeros(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < inner; k++) {
      const value = a[i][k];
      if (value === 0) continue;
      for (let j = 0; j < cols; j++) out[i][j] += value * b[k][j];
    }
  }
  return out;
}

const subtract = (a, b) => a.map((row, r) => row.map((value, c) => value - b[r][c]));
const frobenius = (matrix) => Math.sqrt(matrix.reduce((acc, row) => acc + row.reduce((s, v) => s + v * v, 0), 0));
const maxAbs = (matrix) => matrix.reduce((acc, row) => row.reduce((s, v) => Math.max(s, Math.abs(v)), acc), 0);

/** Modified Gram-Schmidt with one re-orthogonalization pass; drops directions that vanish. */
function orthonormalize(matrix, requested) {
  const rows = matrix.length;
  if (rows === 0) return [];
  const columns = [];
  for (let c = 0; c < matrix[0].length && columns.length < requested; c++) {
    const candidate = matrix.map((row) => row[c]);
    for (let pass = 0; pass < 2; pass++) {
      for (const basis of columns) {
        let projection = 0;
        for (let i = 0; i < rows; i++) projection += basis[i] * candidate[i];
        for (let i = 0; i < rows; i++) candidate[i] -= projection * basis[i];
      }
    }
    const norm = Math.sqrt(candidate.reduce((acc, value) => acc + value * value, 0));
    if (norm > 1e-10) columns.push(candidate.map((value) => value / norm));
  }
  return columns[0] ? columns[0].map((_, i) => columns.map((column) => column[i])) : zeros(rows, 0);
}

/**
 * One-sided Jacobi SVD of a small matrix with m >= n: orthogonalize every column pair until
 * the off-diagonal mass vanishes. Returns singular values (descending) plus U and V.
 */
function jacobiSVD(matrix) {
  const m = matrix.length;
  const n = m === 0 ? 0 : matrix[0].length;
  if (n === 0) return { singularValues: [], U: zeros(m, 0), V: zeros(0, 0) };
  let work = matrix.map((row) => row.slice());
  let v = identity(n);
  for (let sweep = 0; sweep < 60; sweep++) {
    let offDiagonal = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0;
        let beta = 0;
        let gamma = 0;
        for (let i = 0; i < m; i++) {
          alpha += work[i][p] ** 2;
          beta += work[i][q] ** 2;
          gamma += work[i][p] * work[i][q];
        }
        const scale = Math.sqrt(alpha * beta);
        if (gamma === 0 || scale === 0) continue;
        offDiagonal += Math.abs(gamma) / scale;
        if (Math.abs(gamma) <= 1e-16 * scale) continue;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const cosine = 1 / Math.sqrt(1 + t * t);
        const sine = cosine * t;
        for (let i = 0; i < m; i++) {
          const left = work[i][p];
          const right = work[i][q];
          work[i][p] = cosine * left - sine * right;
          work[i][q] = sine * left + cosine * right;
        }
        for (let i = 0; i < n; i++) {
          const left = v[i][p];
          const right = v[i][q];
          v[i][p] = cosine * left - sine * right;
          v[i][q] = sine * left + cosine * right;
        }
      }
    }
    if (offDiagonal < 1e-15) break;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => {
    const na = Math.hypot(...work.map((row) => row[a]));
    const nb = Math.hypot(...work.map((row) => row[b]));
    return nb - na;
  });
  const singularValues = order.map((c) => Math.hypot(...work.map((row) => row[c])));
  const u = zeros(m, n);
  const vs = zeros(n, n);
  order.forEach((source, target) => {
    const sigma = singularValues[target];
    for (let i = 0; i < m; i++) u[i][target] = sigma > 0 ? work[i][source] / sigma : 0;
    for (let i = 0; i < n; i++) vs[i][target] = v[i][source];
  });
  return { singularValues, U: u, V: vs };
}

/** Closed form for the best rank-k error in the Frobenius norm. */
const optimalRankKError = (singularValues, k) =>
  Math.sqrt(singularValues.slice(k).reduce((acc, value) => acc + value * value, 0));

/**
 * Randomized range finder with oversampling and optional power iterations; returns the basis Q,
 * its coordinates B = Q^T A and the approximation error ||A - Q B||_F.
 */
function randomizedRange(matrix, k, { oversampling = 0, powerIterations = 0, rng }) {
  const m = matrix.length;
  const n = m === 0 ? 0 : matrix[0].length;
  const requested = Math.min(k + oversampling, m, n);
  if (k <= 0 || requested <= 0) return { Q: zeros(m, 0), B: zeros(0, n), error: frobenius(matrix), powerIterations };
  const omega = Array.from({ length: n }, () => Array.from({ length: requested }, () => gaussian(rng)));
  let sketch = matMul(matrix, omega);
  for (let iteration = 0; iteration < powerIterations; iteration++) {
    sketch = matMul(matrix, matMul(transpose(matrix), sketch));
  }
  const q = orthonormalize(sketch, requested);
  if (q.length === 0 || q[0].length === 0) {
    // The sketch collapsed (for instance on the zero matrix): no information to project with.
    return { Q: zeros(m, 0), B: zeros(0, n), error: frobenius(matrix), powerIterations, rankDeficient: true };
  }
  const b = matMul(transpose(q), matrix);
  const error = frobenius(subtract(matrix, matMul(q, b)));
  return { Q: q, B: b, error, powerIterations, rankDeficient: false };
}

/** Deterministic low-rank test matrix A = X Y^T with a prescribed spectrum shape. */
function lowRankMatrix(rng, m, n, rank, decay) {
  const x = Array.from({ length: m }, () => Array.from({ length: rank }, () => gaussian(rng)));
  const y = Array.from({ length: n }, () => Array.from({ length: rank }, () => gaussian(rng)));
  const scaled = x.map((row) => row.map((value, c) => value * decay(c)));
  return matMul(scaled, transpose(y));
}

module.exports = {
  makeRng, gaussian, orthonormalize, jacobiSVD, optimalRankKError, randomizedRange,
  lowRankMatrix, matMul, transpose, frobenius, subtract, maxAbs, zeros, identity,
};

if (require.main === module) {
  const rng = makeRng(20260730);

  // -- the exact SVD oracle checks out ------------------------------------------------------
  for (const [m, n] of [[6, 4], [8, 8], [12, 5], [5, 5]]) {
    const matrix = lowRankMatrix(rng, m, n, Math.min(m, n), () => 1);
    const { singularValues, U, V } = jacobiSVD(matrix);
    ok(singularValues.every((value, i) => value >= 0 && (i === 0 || singularValues[i - 1] >= value - 1e-12)),
      `${m}x${n}: singular values must be non-negative and descending`);
    const sigma = zeros(m, n);
    singularValues.forEach((value, i) => { sigma[i][i] = value; });
    const rebuilt = matMul(matMul(U, sigma), transpose(V));
    ok(maxAbs(subtract(matrix, rebuilt)) < 1e-10 * Math.max(1, maxAbs(matrix)),
      `${m}x${n}: U Sigma V^T must rebuild the matrix`);
    ok(Math.abs(frobenius(matrix) - Math.hypot(...singularValues)) < 1e-10 * Math.max(1, frobenius(matrix)),
      `${m}x${n}: the Frobenius norm is the 2-norm of the spectrum`);
    const gram = matMul(transpose(U), U);
    ok(maxAbs(subtract(gram, identity(Math.min(m, n)))) < 1e-10, `${m}x${n}: U must have orthonormal columns`);
  }
  console.log("jacobi SVD: reconstruction, norm identity and orthonormality all hold");

  // -- the closed form for the optimal rank-k error -----------------------------------------
  const rank = 12;
  const rows = 40;
  const columns = 30;
  const spectrum = (i) => 1 / (i + 1);
  const target = lowRankMatrix(rng, rows, columns, rank, spectrum);
  const exact = jacobiSVD(target);
  strictEqual(exact.singularValues.filter((value) => value > 1e-10).length, rank,
    "the constructed matrix must have exactly the prescribed rank");
  for (const k of [1, 3, 6, 12]) {
    const uk = exact.U.map((row) => row.slice(0, k));      // m x k
    const vk = exact.V.map((row) => row.slice(0, k));      // n x k
    const sigmaK = zeros(k, k);
    for (let i = 0; i < k; i++) sigmaK[i][i] = exact.singularValues[i];
    const truncated = matMul(matMul(uk, sigmaK), transpose(vk));
    const best = frobenius(subtract(target, truncated));
    const closed = optimalRankKError(exact.singularValues, k);
    ok(Math.abs(closed - best) <= 1e-9 * Math.max(1, closed),
      `k=${k}: the truncated SVD error ${best} must equal the closed form ${closed}`);
  }
  ok(optimalRankKError(exact.singularValues, rank) < 1e-10,
    "beyond the numerical rank the optimal error vanishes");
  console.log(`rank-${rank} test matrix: sigma_1=${exact.singularValues[0].toFixed(3)}` +
    ` sigma_${rank}=${exact.singularValues[rank - 1].toFixed(6)}`);

  // -- the range finder recovers a rank-k subspace to machine precision ---------------------
  const exactK = randomizedRange(target, rank, { oversampling: 4, rng });
  ok(exactK.error < 1e-9 * frobenius(target),
    `k = rank must be recovered almost exactly, got error ${exactK.error}`);
  const overK = randomizedRange(target, rank + 5, { oversampling: 2, rng });
  ok(overK.error < 1e-9 * frobenius(target),
    `asking for more directions only helps, got error ${overK.error}`);
  const zeroK = randomizedRange(target, 0, { rng });
  strictEqual(zeroK.error, frobenius(target), "k = 0 leaves the whole norm as error");
  strictEqual(zeroK.Q.length > 0 ? zeroK.Q[0].length : 0, 0, "k = 0 returns no basis at all");

  // -- for k < rank the randomized error stays close to optimal ----------------------------
  const measured = [];
  for (const k of [2, 4, 6, 8]) {
    const run = randomizedRange(target, k, { rng });      // no oversampling: Q has exactly k columns
    const optimal = optimalRankKError(exact.singularValues, k);
    const ratio = run.error / optimal;
    measured.push({ k, ratio, error: run.error, optimal });
    ok(ratio >= 0.999, `k=${k}: no sketch may beat the optimal rank-k error (${ratio})`);
    ok(ratio < 3, `k=${k}: randomized error ${run.error} must be within 3x of optimal ${optimal}`);
    strictEqual(run.Q[0].length, k, `k=${k}: without oversampling Q holds exactly k directions`);
    const gram = matMul(transpose(run.Q), run.Q);
    ok(maxAbs(subtract(gram, identity(k))) < 1e-10, `k=${k}: Q must be orthonormal`);
    const pythagoras = frobenius(target) ** 2 - frobenius(run.B) ** 2 - run.error ** 2;
    ok(Math.abs(pythagoras) < 1e-8 * frobenius(target) ** 2,
      `k=${k}: ||A||^2 = ||Q^T A||^2 + ||A - Q Q^T A||^2 must hold`);
  }
  const oversampled = randomizedRange(target, 4, { oversampling: 4, rng });
  const eightDirections = optimalRankKError(exact.singularValues, 8);
  strictEqual(oversampled.Q[0].length, 8, "oversampling adds directions to the basis");
  ok(oversampled.error >= eightDirections * 0.999,
    "so the comparison has to be against the optimal rank-(k+p) error");
  ok(oversampled.error < eightDirections * 3,
    `oversampling 4 landed at ${(oversampled.error / eightDirections).toFixed(3)}x the rank-8 optimum`);
  console.log(`rank finder vs optimal: ${measured.map((m) => `k${m.k}:${m.ratio.toFixed(3)}`).join(" ")}`);

  // -- power iterations buy accuracy on a slowly decaying spectrum --------------------------
  const slow = lowRankMatrix(rng, 50, 40, 20, (i) => 1 / (i + 1) ** 1.5);
  const slowOptimal = optimalRankKError(jacobiSVD(slow).singularValues, 5);
  const plainSketch = randomizedRange(slow, 5, { rng });
  const oneIteration = randomizedRange(slow, 5, { powerIterations: 1, rng });
  const twoIterations = randomizedRange(slow, 5, { powerIterations: 2, rng });
  ok(plainSketch.error >= slowOptimal * 0.999,
    "even a plain sketch cannot beat the optimal rank-5 error");
  ok(oneIteration.error < plainSketch.error,
    `a power iteration must sharpen a slowly decaying sketch (${oneIteration.error} < ${plainSketch.error})`);
  ok(Math.min(oneIteration.error, twoIterations.error) < plainSketch.error * 0.9,
    "and the improvement must be substantial");
  ok(oneIteration.error / slowOptimal < 1.3 && twoIterations.error / slowOptimal < 1.3,
    "the iterated sketches must land near the optimum");
  console.log(`power iterations: ratio q0=${(plainSketch.error / slowOptimal).toFixed(3)}` +
    ` q1=${(oneIteration.error / slowOptimal).toFixed(3)} q2=${(twoIterations.error / slowOptimal).toFixed(3)}`);

  // -- degenerate shapes ---------------------------------------------------------------------
  const single = randomizedRange([[3], [4], [0]], 1, { rng });
  ok(Math.abs(single.error) < 1e-12, "a single column is recovered exactly with k = 1");
  const zeroMatrix = zeros(4, 4);
  const zeroRun = randomizedRange(zeroMatrix, 2, { rng });
  strictEqual(zeroRun.error, 0, "the zero matrix has zero error at any k");
  ok(orthonormalize(zeros(3, 2), 2).length === 3 && orthonormalize(zeros(3, 2), 2)[0].length === 0,
    "orthonormalizing a zero sketch yields no basis rather than NaN");
  const bigK = randomizedRange(target, 40, { rng });
  ok(bigK.error < 1e-9 * frobenius(target), "k above min(m, n) is clamped and stays exact");
  const wide = randomizedRange([[1, 0, 2], [0, 3, 0]], 1, { oversampling: 2, rng });
  ok(wide.error >= 0 && Number.isFinite(wide.error), "wide matrices must not break the sketch");
  const flat = jacobiSVD([]);
  deepStrictEqual(flat.singularValues, [], "an empty matrix has an empty spectrum");
  deepStrictEqual(jacobiSVD([[]]).singularValues, [], "so does a matrix with no columns");

  // -- determinism -----------------------------------------------------------------------------
  const replay = () => {
    const local = makeRng(99);
    const run = randomizedRange(lowRankMatrix(local, 12, 9, 5, (i) => 1 / (i + 1)), 3, { oversampling: 2, rng: local });
    return `${run.error}/${JSON.stringify(run.B)}`;
  };
  deepStrictEqual(replay(), replay(), "identical seeds must give identical sketches and errors");
  console.log("PASS");
}
