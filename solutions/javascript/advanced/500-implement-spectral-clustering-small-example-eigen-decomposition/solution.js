/**
 * Task 500 - Implement spectral clustering small example (eigen decomposition). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Spectral 2-clustering of a handful of points: build the Gaussian similarity matrix
 * W_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)), form the combinatorial Laplacian L = D - W (D the
 * row sums), and take the Fiedler vector - the eigenvector of the second-smallest eigenvalue -
 * via cyclic Jacobi rotations (sweeps until the off-diagonal Frobenius norm drops below 1e-12,
 * documented cap 100 sweeps). The split threshold is the mean of the Fiedler vector, with the
 * sign fixed by making point 0 lie at or above the mean. Documented checks: the zero eigenvalue
 * and the residual ||L v - lambda2 v|| (<= 1e-9), lambda2 < lambda3 with the vector orthogonal
 * to the constant, and the resulting cut equals the minimum over all balanced 6-6 cuts of the
 * canonical two-blob example (the planted split). The canonical example is two six-point
 * clusters around (0,0) and (10,10) with sigma = 3.0 (cross-cluster similarities ~1e-4, so
 * lambda2 ~ 4.85e-4 - comfortably above the Jacobi tolerance); the tiny second example is two
 * LCG-jittered 4-point blobs around (0,0) and (3,0) with sigma = 0.5, where the Fiedler sign
 * pattern must again reproduce the planted partition.
 */
"use strict";

const CANONICAL = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 1.5], [1.5, 0.5],
                   [10.0, 10.0], [11.0, 10.0], [10.0, 11.0], [11.0, 11.0], [10.5, 10.5],
                   [9.5, 10.5]];
const CANONICAL_TRUTH = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
const SIGMA_CANONICAL = 3.0;
const SIGMA_TINY = 0.5;

function sqdist(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

/** Two 4-point blobs around (0,0) and (3,0), jitter +/-0.4 in 0.01 steps via the LCG. */
function tinyBlobs(seed) {
  const points = [];
  for (const cx of [0.0, 3.0]) {
    for (let i = 0; i < 4; i++) {
      seed = (seed * 75 + 74) % 65537;
      const x = cx + (seed % 81 - 40) / 100.0;
      seed = (seed * 75 + 74) % 65537;
      const y = (seed % 81 - 40) / 100.0;
      points.push([x, y]);
    }
  }
  return [points, [0, 0, 0, 0, 1, 1, 1, 1]];
}

function similarity(points, sigma) {
  const n = points.length;
  const w = [];
  for (let i = 0; i < n; i++) w.push(new Array(n).fill(0.0));
  const denom = 2.0 * sigma * sigma;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const value = Math.exp(-sqdist(points[i], points[j]) / denom);
      w[i][j] = value;
      w[j][i] = value;
    }
  }
  return w;
}

function laplacian(w) {
  const n = w.length;
  const out = [];
  for (let i = 0; i < n; i++) {
    let degree = 0.0;
    for (let j = 0; j < n; j++) degree += w[i][j];
    const row = new Array(n);
    for (let j = 0; j < n; j++) row[j] = (i === j ? degree : 0.0) - w[i][j];
    out.push(row);
  }
  return out;
}

/** Cyclic Jacobi eigen-decomposition of a symmetric matrix; eigenvectors are columns. */
function jacobiEigh(matrix, maxSweeps = 100, tol = 1e-12) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  const v = [];
  for (let i = 0; i < n; i++) {
    v.push(new Array(n).fill(0.0));
    v[i][i] = 1.0;
  }
  let sweeps = 0;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0.0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    }
    if (Math.sqrt(off) <= tol) break;
    sweeps += 1;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) <= 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = 1.0 / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
        if (theta < 0.0) t = -t;
        const c = 1.0 / Math.sqrt(t * t + 1.0);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
          a[p][k] = a[k][p];
          a[q][k] = a[k][q];
        }
        const app = a[p][p];
        const aqq = a[q][q];
        const apq = a[p][q];
        a[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return [a.map((row, i) => row[i]), v, sweeps];
}

function spectralSplit(points, sigma) {
  const n = points.length;
  const w = similarity(points, sigma);
  const lmat = laplacian(w);
  const [values, vectors, sweeps] = jacobiEigh(lmat);
  const order = values.map((value, i) => [value, i]).sort((x, y) => x[0] - y[0]);
  const fiedler = order[1][1];
  const lam2 = values[fiedler];
  let vec = [];
  for (let i = 0; i < n; i++) vec.push(vectors[i][fiedler]);
  const mean = vec.reduce((a, b) => a + b, 0.0) / n;
  if (vec[0] < mean) vec = vec.map((value) => -value);
  const labels = vec.map((value) => (value >= mean ? 0 : 1));
  let residual = 0.0;
  for (let i = 0; i < n; i++) {
    let acc = 0.0;
    for (let j = 0; j < n; j++) acc += lmat[i][j] * vec[j];
    residual += (acc - lam2 * vec[i]) ** 2;
  }
  const orthogonality = Math.abs(vec.reduce((a, b) => a + b, 0.0));
  return { labels, lam2, residual: Math.sqrt(residual), orthogonality, sweeps };
}

function cutWeight(w, side) {
  const n = w.length;
  let total = 0.0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (side[i] === 0 && side[j] === 1) total += w[i][j];
    }
  }
  return total;
}

/** Brute force over all balanced half-half splits (point 0 in side A, mirrors skipped). */
function minBalancedCut(w, half) {
  const n = w.length;
  const choose = half - 1;
  const idx = [];
  for (let i = 0; i < choose; i++) idx.push(i + 1);
  let best = null;
  for (;;) {
    const side = new Array(n).fill(1);
    side[0] = 0;
    for (const i of idx) side[i] = 0;
    if (side[0] !== side[n - 1]) {
      const weight = cutWeight(w, side);
      if (best === null || weight < best) best = weight;
    }
    let i = choose - 1;
    while (i >= 0 && idx[i] === n - choose + i) i--;
    if (i < 0) break;
    idx[i] += 1;
    for (let j = i + 1; j < choose; j++) idx[j] = idx[j - 1] + 1;
  }
  return best;
}

function main() {
  const assert = require("node:assert");

  const [values2] = jacobiEigh([[2.0, 1.0], [1.0, 2.0]]);
  const sorted2 = values2.slice().sort((a, b) => a - b);
  assert.ok(Math.abs(sorted2[0] - 1.0) <= 1e-12 && Math.abs(sorted2[1] - 3.0) <= 1e-12);

  const canonical = spectralSplit(CANONICAL, SIGMA_CANONICAL);
  assert.ok(canonical.residual <= 1e-9, "eigen-pair residual");
  assert.ok(canonical.orthogonality <= 1e-8, "Fiedler vector orthogonal to the constant");
  assert.ok(canonical.lam2 >= 1e-6 && canonical.lam2 <= 1e-2,
    "two separated blobs: tiny positive Fiedler value");
  assert.ok(canonical.sweeps <= 100);

  const w = similarity(CANONICAL, SIGMA_CANONICAL);
  const [spectrum] = jacobiEigh(laplacian(w));
  const asc = spectrum.slice().sort((a, b) => a - b);
  assert.ok(Math.abs(asc[0]) <= 1e-9, "connected graph: one zero eigenvalue");
  assert.ok(asc[1] < asc[2], "Fiedler eigenvalue is strictly second smallest");
  assert.ok(asc[1] <= 1e-2 * asc[2], "documented: the Fiedler value separates the blobs");

  assert.deepStrictEqual(canonical.labels, CANONICAL_TRUTH,
    "canonical split must match the planted structure");
  const spectralCut = cutWeight(w, canonical.labels);
  const bestBalanced = minBalancedCut(w, 6);
  assert.ok(Math.abs(spectralCut - bestBalanced) <= 1e-9,
    "spectral cut equals the min 6-6 cut");
  const totalMass = w.reduce((acc, row) => acc + row.reduce((a, b) => a + b, 0.0), 0.0);
  assert.ok(spectralCut <= 1e-3 * totalMass);

  const [tiny, truth] = tinyBlobs(2024);
  const tinySplit = spectralSplit(tiny, SIGMA_TINY);
  assert.deepStrictEqual(tinySplit.labels, truth, "tiny LCG blobs split correctly");
  assert.ok(tinySplit.residual <= 1e-9 && tinySplit.orthogonality <= 1e-8);
  assert.ok(tinySplit.lam2 >= 1e-8 && tinySplit.lam2 <= 1e-2, "tiny positive Fiedler value");
  assert.deepStrictEqual(spectralSplit(tiny, SIGMA_TINY).labels, tinySplit.labels,
    "deterministic re-run");

  const spaced = tiny.map(([x, y]) => [x * 10.0, y * 10.0]);
  assert.deepStrictEqual(spectralSplit(spaced, SIGMA_TINY * 10.0).labels, truth);
}

module.exports = { CANONICAL, CANONICAL_TRUTH, SIGMA_CANONICAL, SIGMA_TINY, sqdist, tinyBlobs,
  similarity, laplacian, jacobiEigh, spectralSplit, cutWeight, minBalancedCut };

if (require.main === module) {
  main();
  console.log("PASS");
}
