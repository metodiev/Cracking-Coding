/**
 * Task 502 - Implement randomized SVD basic using power method. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * The randomized (Halko-Martinsson-Tropp) SVD skeleton: draw a Gaussian test matrix Omega of
 * size n x l (l = k + oversampling, capped at min(m, n)) with the file's own LCG, form the sketch
 * Y = A Omega, optionally sharpen it with q power iterations of the "Y <- A (A^T Y)" form (each
 * round re-orthonormalising both intermediate sketches), extract an orthonormal basis Q of
 * range(Y) by modified Gram-Schmidt with one re-orthogonalisation pass (a rank-deficient sketch
 * is completed from the canonical basis so Q stays orthonormal), project B = Q^T A (l x n) and
 * factor that small matrix exactly by cyclic Jacobi on B B^T: sigma_i = sqrt(lambda_i) and
 * V_i = B^T u_i / sigma_i. Everything is deterministic given the seed - no Math.random anywhere.
 *
 * Documented checks, each against an independent method: the singular values from the randomized
 * factorisation match the exact spectrum obtained by Jacobi on A^T A (with the exact U column
 * lifted as A v / sigma), and the truncated reconstruction error matches the Eckart-Young optimum
 * sqrt(sum_{i > k} sigma_i^2), on: a 6 x 4 matrix with spectrum (3, 1, 0.5, 0.1) truncated to
 * k = 2; a rank-3 8 x 6 matrix with spectrum (5, 3, 1, 0, 0, 0), where the randomised
 * factorisation must return sigma_1..sigma_3 to 1e-8, a relative reconstruction error <= 1e-10
 * (Q spans range(A) exactly once l >= rank) and rank 3 with the 1e-8 threshold; and a 5 x 3
 * matrix with l = k = n and no oversampling at all. Q, U and V are orthonormal to 1e-9 and shapes
 * are (m x l), (m x k), (k x n). Theory is pinned down explicitly: on the spectrum
 * (1, 0.5, 0.25, 0.125, 0.0625) with k = 1, the sin(angle) between the computed and the exact
 * leading left singular vector must fall for every extra power iteration and reach <= 0.05 at
 * q = 3 (a third of its q = 0 value) - the documented (sigma_2 / sigma_1)^(2q + 1) rate. Finally
 * four different seeds reproduce sigma_1 to 1e-6 with oversampling plus power iterations, and two
 * seeds agree on the top-2 spectrum to 1e-6: the sketch is stable, not lucky.
 */
"use strict";

function nextNormal(seedIn) {
  let seed = seedIn;
  let total = 0.0;
  for (let i = 0; i < 12; i++) {
    seed = (seed * 75 + 74) % 65537;
    total += (seed % 65537) / 65537.0;
  }
  return [total - 6.0, seed];
}

function dot(a, b) {
  let total = 0.0;
  for (let i = 0; i < a.length; i++) total += a[i] * b[i];
  return total;
}

function frobenius(matrix) {
  let total = 0.0;
  for (const row of matrix) for (const value of row) total += value * value;
  return Math.sqrt(total);
}

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0.0));
}

function transpose(matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const out = zeros(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) out[j][i] = matrix[i][j];
  }
  return out;
}

function matmul(a, b) {
  const n = a.length;
  const inner = b.length;
  const p = b[0].length;
  const bt = transpose(b);
  const out = zeros(n, p);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) out[i][j] = dot(a[i], bt[j]);
  }
  return out;
}

function lcgMatrix(rows, cols, seed) {
  const out = zeros(rows, cols);
  let s = seed;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const pair = nextNormal(s);
      s = pair[1];
      out[i][j] = pair[0];
    }
  }
  return out;
}

/** Modified Gram-Schmidt with one re-orthogonalisation pass; always returns `cols` columns. */
function qrGramSchmidt(y, cols) {
  const rows = y.length;
  const q = zeros(rows, cols);
  for (let j = 0; j < cols; j++) {
    let v = y.map((row) => row[j]);
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < j; i++) {
        const qi = q.map((row) => row[i]);
        const scale = dot(qi, v);
        v = v.map((value, k) => value - scale * qi[k]);
      }
    }
    let length = Math.sqrt(dot(v, v));
    if (length <= 1e-12) {
      // rank-deficient sketch column: fall back to a canonical vector, re-orthogonalised
      let candidate = null;
      for (let attempt = 0; attempt < rows; attempt++) {
        const e = new Array(rows).fill(0.0);
        e[attempt] = 1.0;
        let w = e;
        for (let pass = 0; pass < 2; pass++) {
          for (let i = 0; i < j; i++) {
            const qi = q.map((row) => row[i]);
            const scale = dot(qi, w);
            w = w.map((value, k) => value - scale * qi[k]);
          }
        }
        if (Math.sqrt(dot(w, w)) >= 1e-8) {
          candidate = w;
          break;
        }
      }
      if (candidate === null) candidate = new Array(rows).fill(0.0);
      v = candidate;
      length = Math.sqrt(dot(v, v));
      if (length <= 1e-300) continue;                 // give up: leave a zero column
    }
    for (let i = 0; i < rows; i++) q[i][j] = v[i] / length;
  }
  return q;
}

/** Cyclic Jacobi eigen-decomposition of a symmetric matrix; ascending eigen-pairs. */
function jacobiEigh(matrix, maxSweeps = 100, tol = 1e-15) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  const v = [];
  for (let i = 0; i < n; i++) {
    v.push(new Array(n).fill(0.0));
    v[i][i] = 1.0;
  }
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0.0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    }
    if (Math.sqrt(off) <= tol) break;
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
  const order = a.map((row, i) => [row[i], i]).sort((x, y) => x[0] - y[0]);
  return {
    values: order.map(([value]) => value),
    vectors: order.map(([, i]) => v.map((row) => row[i])),
  };
}

/**
 * Randomized SVD. Returns { u (m x k), s (k), v (k x n), q, l, sweeps } with rows of v being
 * right singular vectors (so A ~ u diag(s) v^T).
 */
function randomizedSvd(a, k, oversampling = 5, powerIters = 1, seed = 12345) {
  const m = a.length;
  const n = a[0].length;
  const l = Math.min(k + oversampling, Math.min(m, n));
  const omega = lcgMatrix(n, l, seed);
  let y = matmul(a, omega);
  const at = transpose(a);
  for (let i = 0; i < powerIters; i++) {
    const qy = qrGramSchmidt(y, l);
    const z = matmul(at, qy);
    const qz = qrGramSchmidt(z, l);
    y = matmul(a, qz);
  }
  const q = qrGramSchmidt(y, l);
  const b = matmul(transpose(q), a);                    // l x n
  const bbT = matmul(b, transpose(b));                  // l x l
  const { values, vectors } = jacobiEigh(bbT);
  const order = values.map((value, i) => [value, i]).sort((x, y2) => y2[0] - x[0]);
  const sigma = [];
  const ub = [];
  const vb = [];
  for (let idx = 0; idx < Math.min(k, order.length); idx++) {
    const [lambda, col] = order[idx];
    const value = Math.sqrt(Math.max(lambda, 0.0));
    sigma.push(value);
    ub.push(vectors[col]);
    if (value > 1e-12) {
      const uCol = vectors[col];
      const vt = new Array(n).fill(0.0);
      for (let idx2 = 0; idx2 < uCol.length; idx2++) {
        if (uCol[idx2] === 0.0) continue;
        const scale = uCol[idx2];
        for (let j = 0; j < n; j++) vt[j] += scale * b[idx2][j];
      }
      vb.push(vt.map((x) => x / value));
    } else {
      vb.push(new Array(n).fill(0.0));
    }
  }
  const u = matmul(q, transpose(ub));                   // m x k
  return { u, s: sigma, v: vb, q, l, powerIters };
}

/** Exact singular triple for a small matrix: eigen-decompose A^T A and lift U = A V / sigma. */
function exactSvdSmall(a) {
  const gram = matmul(transpose(a), a);
  const { values, vectors } = jacobiEigh(gram);
  const order = values.map((value, i) => [value, i]).sort((x, y) => y[0] - x[0]);
  const s = [];
  const v = [];
  const u = [];
  for (const [lambda, col] of order) {
    const value = Math.sqrt(Math.max(lambda, 0.0));
    s.push(value);
    const vec = vectors[col];
    v.push(vec);
    if (value > 1e-12) {
      const column = matmul(a, vec.map((x) => [x])).map((row) => row[0] / value);
      u.push(column);
    } else {
      u.push(new Array(a.length).fill(0.0));
    }
  }
  return { s, v, u };
}

function exactSingularValues(a) {
  return exactSvdSmall(a).s;
}

function reconstruct(u, s, v, k) {
  const m = u.length;
  const n = v[0].length;
  const out = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let total = 0.0;
      for (let t = 0; t < k; t++) total += u[i][t] * s[t] * v[t][j];
      out[i][j] = total;
    }
  }
  return out;
}

function maxAbsDiff(a, b) {
  let best = 0.0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) best = Math.max(best, Math.abs(a[i][j] - b[i][j]));
  }
  return best;
}

/** U diag(s) V^T with orthonormal U (m x r) and V (r x n) built from LCG vectors. */
function matrixWithSpectrum(sigmas, rows, cols, seed) {
  const r = sigmas.length;
  const u = qrGramSchmidt(lcgMatrix(rows, r, seed), r);
  const vT = qrGramSchmidt(lcgMatrix(cols, r, seed + 101), r);
  const v = transpose(vT);                              // r x cols
  const out = zeros(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let total = 0.0;
      for (let t = 0; t < r; t++) total += u[i][t] * sigmas[t] * v[t][j];
      out[i][j] = total;
    }
  }
  return out;
}

function orthonormalityError(matrix) {
  const t = transpose(matrix);
  const product = matmul(t, matrix);
  let best = 0.0;
  for (let i = 0; i < product.length; i++) {
    for (let j = 0; j < product[0].length; j++) {
      best = Math.max(best, Math.abs(product[i][j] - (i === j ? 1.0 : 0.0)));
    }
  }
  return best;
}

function main() {
  const assert = require("node:assert");
  const rel = (numer, denom) => numer / Math.max(denom, 1e-300);

  // (1) well-conditioned 6 x 4 matrix with a known spectrum.
  const sigmas1 = [3.0, 1.0, 0.5, 0.1];
  const a1 = matrixWithSpectrum(sigmas1, 6, 4, 4242);
  const exact1 = exactSingularValues(a1);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(exact1[i] - sigmas1[i]) <= 1e-9 * sigmas1[0],
      `planted sigma_${i} recovered by the exact reference`);
  }
  const k = 2;
  const svd1 = randomizedSvd(a1, k, 5, 1, 20260116);
  assert.strictEqual(svd1.u.length, 6);
  assert.strictEqual(svd1.u[0].length, k);
  assert.strictEqual(svd1.v.length, k);
  assert.strictEqual(svd1.v[0].length, 4);
  assert.strictEqual(svd1.s.length, k);
  assert.strictEqual(svd1.l, 4, "l = k + oversampling capped at min(m, n) = 4");
  assert.ok(orthonormalityError(svd1.u) <= 1e-9, "U orthonormal");
  assert.ok(orthonormalityError(transpose(svd1.v)) <= 1e-9, "V orthonormal");
  assert.ok(orthonormalityError(svd1.q) <= 1e-9, "Q orthonormal");
  for (let i = 0; i < k; i++) {
    assert.ok(Math.abs(svd1.s[i] - exact1[i]) <= 1e-8 * exact1[0],
      `randomised sigma_${i} matches the exact spectrum`);
  }
  assert.ok(svd1.s[0] >= svd1.s[1], "singular values sorted descending");
  const ey1 = Math.sqrt(exact1.slice(k).reduce((acc, value) => acc + value * value, 0.0));
  const err1 = frobenius(sub(a1, reconstruct(svd1.u, svd1.s, svd1.v, k)));
  assert.ok(Math.abs(err1 - ey1) <= 1e-7 * frobenius(a1),
    "rank-2 error equals the Eckart-Young optimum sqrt(sum_{i > 2} sigma_i^2)");

  // (2) rank-3 8 x 6 matrix: Q must span range(A) exactly once l >= rank.
  const sigmas2 = [5.0, 3.0, 1.0, 0.0, 0.0, 0.0];
  const a2 = matrixWithSpectrum(sigmas2, 8, 6, 777);
  const exact2 = exactSingularValues(a2);
  const svd2 = randomizedSvd(a2, 3, 5, 2, 999);
  assert.strictEqual(svd2.l, 6, "l capped at min(m, n) = 6 >= rank");
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(svd2.s[i] - exact2[i]) <= 1e-8 * exact2[0],
      `rank-deficient: sigma_${i} exact`);
  }
  const residual2 = sub(a2, reconstruct(svd2.u, svd2.s, svd2.v, 3));
  assert.ok(frobenius(residual2) / frobenius(a2) <= 1e-10,
    "a rank-3 matrix is reconstructed to 1e-10 once l >= rank");
  assert.strictEqual(rankOf(svd2, 1e-8), 3, "the sketch reveals rank 3");
  assert.deepStrictEqual(
    randomizedSvd(a2, 3, 5, 2, 999).s.map((value) => value.toFixed(12)),
    svd2.s.map((value) => value.toFixed(12)), "same seed, identical output");

  // (3) an extra exact check on a rectangular 5 x 3 matrix with all singular values moderate.
  const sigmas3 = [2.0, 1.2, 0.7];
  const a3 = matrixWithSpectrum(sigmas3, 5, 3, 31337);
  const exact3 = exactSingularValues(a3);
  const svd3 = randomizedSvd(a3, 3, 0, 0, 55);
  assert.strictEqual(svd3.l, 3, "no oversampling: l = k = 3 = n");
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(svd3.s[i] - exact3[i]) <= 1e-9 * exact3[0], `sigma_${i} of the 5x3`);
  }
  assert.ok(frobenius(sub(a3, reconstruct(svd3.u, svd3.s, svd3.v, 3))) <= 1e-10 * frobenius(a3),
    "full rank captured without oversampling when l = n");

  // (4) power iterations sharpen the leading subspace (rate (sigma_2 / sigma_1)^(2q + 1)).
  const sigmas4 = [1.0, 0.5, 0.25, 0.125, 0.0625];
  const a4 = matrixWithSpectrum(sigmas4, 20, 5, 8080);
  const exact4 = exactSvdSmall(a4);
  const angle = (q) => {
    const out = randomizedSvd(a4, 1, 1, q, 424243);
    const cos = Math.abs(dot(out.u.map((row) => row[0]), exact4.u[0]));
    return Math.sqrt(Math.max(1.0 - cos * cos, 0.0));
  };
  const sinQ0 = angle(0);
  const sinQ1 = angle(1);
  const sinQ3 = angle(3);
  assert.ok(sinQ0 >= 0.05, `q = 0 leaves a visible subspace error (sin = ${sinQ0})`);
  assert.ok(sinQ1 < sinQ0 && sinQ3 < sinQ1, "each power iteration shrinks the angle");
  assert.ok(sinQ3 <= 0.05 && sinQ3 <= sinQ0 / 3,
    `q = 3 brings sin(angle) to <= 0.05 (${sinQ3})`);
  const sharped = randomizedSvd(a4, 1, 1, 3, 424243);
  assert.ok(Math.abs(sharped.s[0] - exact4.s[0]) <= 1e-4, "sigma_1 sharpened by the sketch");

  // (5) stability across seeds.
  for (const seed of [1, 2, 3, 987654]) {
    const run = randomizedSvd(a4, 1, 2, 2, seed);
    assert.ok(Math.abs(run.s[0] - 1.0) <= 1e-6,
      `seed ${seed} recovers sigma_1 = 1 to 1e-6 when oversampling plus power iterations are used`);
  }
  const seedA = randomizedSvd(a1, 2, 5, 1, 11);
  const seedB = randomizedSvd(a1, 2, 5, 1, 12);
  for (let i = 0; i < 2; i++) {
    assert.ok(Math.abs(seedA.s[i] - seedB.s[i]) <= 1e-6, "two seeds agree on the top spectrum");
  }
}

/** a - b for equal-shaped matrices. */
function sub(a, b) {
  return a.map((row, i) => row.map((value, j) => value - b[i][j]));
}

function rankOf(svdResult, tol) {
  const threshold = tol * (svdResult.s[0] || 1.0);
  return svdResult.s.filter((value) => value > threshold).length;
}

module.exports = { nextNormal, dot, frobenius, zeros, transpose, matmul, lcgMatrix,
  qrGramSchmidt, jacobiEigh, randomizedSvd, exactSvdSmall, exactSingularValues, reconstruct,
  maxAbsDiff, matrixWithSpectrum, orthonormalityError, sub, rankOf };

if (require.main === module) {
  main();
  console.log("PASS");
}
