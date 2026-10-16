/**
 * Task 501 - Implement PCA via power iteration for largest eigenvector. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Principal component analysis built from scratch on top of a single primitive: the power
 * iteration v <- C v / ||C v||, whose Rayleigh quotient v^T C v converges to the dominant
 * eigenvalue with the classical (lambda_2 / lambda_1)^t error rate (cap 500 iterations, stop when
 * the Rayleigh quotient moves by <= 1e-15). The data matrix is mean-centred and the sample
 * covariance C = X^T X / (n - 1) is formed explicitly (small d). Further components come from two
 * interchangeable deflations: Hotelling deflation (C <- C - lambda_j v_j v_j^T, which works on the
 * matrix) and Gram-Schmidt orthogonalisation of every iterate against the previously accepted
 * vectors (which works on the vectors). Both must agree component-wise, and the resulting
 * spectrum is cross-checked against a Jacobi eigen-decomposition of C - an independent method.
 * The starting vector is seeded from the file's own LCG (never random) and eigenvectors are
 * sign-canonicalised by making the largest-magnitude entry positive, so runs are reproducible.
 *
 * Documented checks: (a) the exact rank-1 cloud (t, 2t) has a closed form - PC1 = [1, 2]/sqrt(5)
 * with lambda_1 = 5 * sum(t^2) / (n - 1) and lambda_2 = 0, asserted to 1e-12; (b) the Rayleigh
 * sequence of a full power run is non-decreasing (to 1e-12) and the residual ||C v - lambda v||
 * is <= 1e-9; (c) Jacobi and power-iteration spectra agree to 1e-9 on well-separated clouds and
 * the explained-variance ratio matches the Jacobi spectrum; (d) on an LCG cloud with planted
 * orthonormal directions of amplitudes 3.0 / 1.0 / 0.4 (n = 400, d = 3) each recovered component
 * has |cos| >= 0.98 with its planted direction and lambda_j is within 15% of amplitude^2;
 * (e) degenerate inputs are handled - constant data gives an exactly zero covariance, the regular
 * tetrahedron has the exact isotropic covariance (4/3) I so all three eigenvalues coincide to
 * 1e-12 while the components stay orthonormal; (f) truncating to k components leaves a squared
 * residual equal to (n - 1) * (sum of dropped eigenvalues) - Eckart-Young for an orthonormal
 * basis on the centred rows - and a short run leaves strictly more error than a long one.
 * Hotelling-deflated components are only asserted orthogonal to 1e-6 (deflation loses digits as
 * lambda_3 / lambda_1 shrinks) while the Gram-Schmidt variant holds 1e-9 by construction.
 */
"use strict";

const MAX_ITER = 500;
const RAYLEIGH_TOL = 1e-15;

function lcgUnit(seed) {
  return ((seed * 75 + 74) % 65537) / 65537.0;
}

/** Standard-normal-ish deviate: sum of 12 uniforms in [0,1) minus 6 (deterministic). */
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

function norm(a) {
  return Math.sqrt(dot(a, a));
}

function normalize(a) {
  const length = norm(a);
  return a.map((value) => value / length);
}

function matVec(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}

function mean(rows) {
  const d = rows[0].length;
  const out = new Array(d).fill(0.0);
  for (const row of rows) {
    for (let j = 0; j < d; j++) out[j] += row[j];
  }
  return out.map((value) => value / rows.length);
}

/** Rows minus the column means. */
function center(rows) {
  const mu = mean(rows);
  return rows.map((row) => row.map((value, j) => value - mu[j]));
}

/** Sample covariance matrix (divide by n - 1) of a centred data matrix. */
function covariance(rows) {
  const n = rows.length;
  const d = rows[0].length;
  const out = [];
  for (let i = 0; i < d; i++) out.push(new Array(d).fill(0.0));
  for (const row of rows) {
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) out[i][j] += row[i] * row[j];
    }
  }
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      const value = out[i][j] / (n - 1);
      out[i][j] = value;
      out[j][i] = value;
    }
  }
  return out;
}

function rayleigh(matrix, vector) {
  return dot(vector, matVec(matrix, vector));
}

/** Largest-magnitude entry made positive, so the sign of a component is reproducible. */
function canonicalize(vector) {
  let pivot = 0;
  for (let i = 1; i < vector.length; i++) {
    if (Math.abs(vector[i]) > Math.abs(vector[pivot])) pivot = i;
  }
  return vector[pivot] < 0.0 ? vector.map((value) => -value) : vector.slice();
}

function unitFromSeed(d, seedIn) {
  let seed = seedIn;
  return normalize(Array.from({ length: d }, () => {
    seed = (seed * 75 + 74) % 65537;
    return 1.0 + lcgUnit(seed);
  }));
}

/**
 * Power iteration on a symmetric matrix from a deterministic start vector.
 * Returns the eigen-pair, the iteration count actually used and the Rayleigh history.
 */
function powerIteration(matrix, iters = MAX_ITER, tol = RAYLEIGH_TOL, start = null) {
  const d = matrix.length;
  let v = start === null ? unitFromSeed(d, 20261016) : normalize(start);
  let lambda = rayleigh(matrix, v);
  const history = [lambda];
  let used = 0;
  for (let t = 0; t < iters; t++) {
    const w = matVec(matrix, v);
    const length = norm(w);
    if (length <= 1e-300) break;                     // exactly zero matrix
    v = w.map((value) => value / length);
    lambda = rayleigh(matrix, v);
    history.push(lambda);
    used = t + 1;
    if (history.length >= 3
        && Math.abs(history[history.length - 1] - history[history.length - 2]) <= tol) break;
  }
  return { vector: canonicalize(v), eigenvalue: rayleigh(matrix, v), iterations: used, history };
}

function projectOut(vector, basis) {
  let out = vector.slice();
  for (const b of basis) {
    const scale = dot(out, b);
    out = out.map((value, i) => value - scale * b[i]);
  }
  const length = norm(out);
  return length <= 1e-300 ? out : out.map((value) => value / length);
}

/** Hotelling deflation: strip each accepted component out of the matrix itself. */
function pcaDeflation(cov, k) {
  const d = cov.length;
  let work = cov.map((row) => row.slice());
  const components = [];
  const values = [];
  for (let j = 0; j < Math.min(k, d); j++) {
    const { vector, eigenvalue } = powerIteration(work);
    components.push(vector);
    values.push(eigenvalue);
    work = work.map((row, i) => row.map((value, col) =>
      value - eigenvalue * vector[i] * vector[col]));
  }
  return { components, values };
}

/** Orthogonalise every iterate against the accepted components; same answers as deflation. */
function pcaOrthogonalized(cov, k) {
  const d = cov.length;
  const components = [];
  const values = [];
  for (let j = 0; j < Math.min(k, d); j++) {
    let v = projectOut(unitFromSeed(d, 777 + j * 31), components);
    for (let t = 0; t < MAX_ITER; t++) {
      const w = projectOut(matVec(cov, v), components);
      const length = norm(w);
      if (length <= 1e-300) break;
      v = w.map((value) => value / length);
    }
    components.push(canonicalize(v));
    values.push(rayleigh(cov, v));
  }
  return { components, values };
}

/** Cyclic Jacobi eigen-decomposition (symmetric matrices); returns ascending eigen-pairs. */
function jacobiEigh(matrix, maxSweeps = 100, tol = 1e-14) {
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

/** Deterministic orthonormal basis built by Gram-Schmidt from LCG vectors. */
function lcgBasis(d, seed) {
  const basis = [];
  for (let j = 0; j < d; j++) {
    let s = seed + j * 97;
    const raw = Array.from({ length: d }, () => {
      s = (s * 75 + 74) % 65537;
      return lcgUnit(s) * 2.0 - 1.0;
    });
    basis.push(normalize(projectOut(raw, basis)));
  }
  return basis;
}

/** Cloud = sum_j amplitude_j * z_ij * basis_j, z ~ N(0,1)-ish, so Cov ~ diag(amplitude^2). */
function plantedCloud(n, amplitudes, basis, seed) {
  const d = basis.length;
  const rows = [];
  let s = seed;
  for (let i = 0; i < n; i++) {
    const row = new Array(d).fill(0.0);
    for (let j = 0; j < d; j++) {
      const pair = nextNormal(s);
      s = pair[1];
      for (let t = 0; t < d; t++) row[t] += amplitudes[j] * pair[0] * basis[j][t];
    }
    rows.push(row);
  }
  return rows;
}

function explainedRatio(values, k) {
  const total = values.reduce((a, b) => a + Math.max(b, 0.0), 0.0);
  const kept = values.slice(0, k).reduce((a, b) => a + Math.max(b, 0.0), 0.0);
  return kept / total;
}

/** Squared-error norm left after projecting the centred rows onto k orthonormal components. */
function reconstructionError(centred, components) {
  let total = 0.0;
  for (const row of centred) {
    let left = row.slice();
    for (const c of components) {
      const scale = dot(left, c);
      left = left.map((value, i) => value - scale * c[i]);
    }
    total += dot(left, left);
  }
  return Math.sqrt(total);
}

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b}| > ${tol}`);

  // (a) exact rank-1 cloud: closed form.
  const ts = [];
  for (let i = -9; i <= 10; i++) ts.push(i / 3.0);
  const line = ts.map((t) => [t, 2.0 * t]);
  const lineCentred = center(line);
  const lineCov = covariance(lineCentred);
  const linePca = pcaDeflation(lineCov, 2);
  const meanT = ts.reduce((a, t) => a + t, 0.0) / ts.length;
  const sumSq = ts.reduce((a, t) => a + (t - meanT) * (t - meanT), 0.0);
  const expected = 5.0 * sumSq / (ts.length - 1);
  close(linePca.values[0], expected, 1e-12, "rank-1 cloud: lambda_1");
  close(linePca.values[1], 0.0, 1e-12, "rank-1 cloud: lambda_2");
  const truth1 = [1.0 / Math.sqrt(5.0), 2.0 / Math.sqrt(5.0)];
  close(Math.abs(dot(linePca.components[0], truth1)), 1.0, 1e-12,
    "rank-1 cloud: PC1 = [1, 2]/sqrt(5)");
  assert.ok(linePca.components[0][1] > 0.0, "sign canonicalised with a positive entry");
  assert.ok(reconstructionError(lineCentred, [linePca.components[0]]) <= 1e-12,
    "one component reproduces a rank-1 cloud");

  // (b) Rayleigh sequence monotone, residual tiny.
  const lineRun = powerIteration(lineCov);
  for (let i = 1; i < lineRun.history.length; i++) {
    assert.ok(lineRun.history[i] >= lineRun.history[i - 1] - 1e-12,
      "Rayleigh quotient is non-decreasing");
  }
  const residualOf = (matrix, vector, value) => {
    const cv = matVec(matrix, vector);
    let acc = 0.0;
    for (let i = 0; i < cv.length; i++) acc += (cv[i] - value * vector[i]) ** 2;
    return Math.sqrt(acc);
  };
  assert.ok(residualOf(lineCov, lineRun.vector, lineRun.eigenvalue) <= 1e-9, "residual <= 1e-9");
  assert.ok(lineRun.iterations <= MAX_ITER, "iteration cap respected");

  // (c) Jacobi (independent method) agrees; explained variance matches.
  const basis3 = lcgBasis(3, 12345);
  const cloud = plantedCloud(400, [3.0, 1.0, 0.4], basis3, 20260101);
  const centred = center(cloud);
  const cov = covariance(centred);
  const jacobi = jacobiEigh(cov);
  const desc = jacobi.values.slice().reverse();
  const deflated = pcaDeflation(cov, 3);
  const orthogonal = pcaOrthogonalized(cov, 3);
  for (let j = 0; j < 3; j++) {
    close(deflated.values[j], desc[j], 1e-9, `deflation vs Jacobi, component ${j}`);
    close(orthogonal.values[j], desc[j], 1e-9, `orthogonalised vs Jacobi, component ${j}`);
    assert.ok(Math.abs(dot(deflated.components[j], orthogonal.components[j])) >= 1 - 1e-9,
      `the two deflations agree on component ${j}`);
    assert.ok(Math.abs(dot(deflated.components[j], jacobi.vectors[2 - j])) >= 1 - 1e-9,
      `component ${j} matches the Jacobi eigenvector`);
  }
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const overlap = Math.abs(dot(deflated.components[i], deflated.components[j]));
      if (i === j) close(overlap, 1.0, 1e-9, "unit norm");
      else assert.ok(overlap <= 1e-6,
        "Hotelling-deflated components are orthogonal to 1e-6 (deflation loses digits as "
        + "lambda_3 / lambda_1 shrinks)");
    }
  }
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      assert.ok(Math.abs(dot(orthogonal.components[i], orthogonal.components[j])) <= 1e-9,
        "Gram-Schmidt components are orthogonal to 1e-9");
    }
  }
  const trace = desc.reduce((a, b) => a + b, 0.0);
  close(explainedRatio(deflated.values, 2), (desc[0] + desc[1]) / trace, 1e-9,
    "two-component explained variance");
  assert.ok(explainedRatio(deflated.values, 1) <= explainedRatio(deflated.values, 2));
  assert.ok(explainedRatio(deflated.values, 2) < 1.0, "third component carries signal");
  assert.ok(deflated.values[0] > deflated.values[1] && deflated.values[1] > deflated.values[2],
    "spectrum sorted descending");

  // (d) planted directions and amplitudes recovered.
  const amplitudes = [3.0, 1.0, 0.4];
  for (let j = 0; j < 3; j++) {
    assert.ok(Math.abs(dot(deflated.components[j], basis3[j])) >= 0.98,
      `component ${j} recovers its planted direction`);
    assert.ok(Math.abs(deflated.values[j] - amplitudes[j] ** 2) <= 0.15 * amplitudes[j] ** 2,
      `lambda_${j} within 15% of amplitude^2`);
  }
  const varianceSum = desc.reduce((a, b) => a + b, 0.0);
  assert.ok(Math.abs(varianceSum - (9.0 + 1.0 + 0.16)) <= 0.15 * 10.16,
    "total variance within 15% of the planted amplitude^2 sum");
  assert.ok(deflated.values[0] >= trace / 3 - 1e-9 && deflated.values[0] <= trace + 1e-9,
    "lambda_1 between trace/d and trace");

  // (e) degenerate inputs.
  const flat = Array.from({ length: 12 }, () => [2.0, -3.0, 0.5]);
  const flatCov = covariance(center(flat));
  assert.ok(flatCov.every((row) => row.every((value) => Math.abs(value) <= 1e-24)),
    "zero-variance data gives a zero covariance matrix");
  const flatPca = pcaDeflation(flatCov, 2);
  assert.ok(Math.abs(flatPca.values[0]) <= 1e-24, "zero-variance data has zero eigenvalues");

  const tetra = [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]];
  const tetraCov = covariance(center(tetra));
  const tetraDesc = jacobiEigh(tetraCov).values.slice().reverse();
  for (const value of tetraDesc) close(value, 4.0 / 3.0, 1e-12, "tetrahedron covariance = 4/3 * I");
  const isoPower = pcaDeflation(tetraCov, 3);
  for (let j = 0; j < 3; j++) close(isoPower.values[j], 4.0 / 3.0, 1e-9, "isotropic spectrum");
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      assert.ok(Math.abs(dot(isoPower.components[i], isoPower.components[j])) <= 1e-9,
        "isotropic components stay orthogonal despite the degenerate spectrum");
    }
  }
  assert.ok(isoPower.components.every((c) => Math.abs(norm(c) - 1.0) <= 1e-9),
    "components keep unit norm");

  // (f) truncated reconstruction error equals the dropped eigenvalues (Eckart-Young).
  for (let k = 1; k <= 2; k++) {
    const dropped = (centred.length - 1) * desc.slice(k).reduce((a, b) => a + b, 0.0);
    const err = reconstructionError(centred, deflated.components.slice(0, k));
    close(err * err, dropped, 1e-4 * Math.max(1.0, dropped),
      `k = ${k}: squared truncation error equals (n - 1) * dropped eigenvalues`);
  }
  close(reconstructionError(centred, deflated.components), 0.0, 1e-6, "full basis reconstructs");
  assert.ok(reconstructionError(centred, deflated.components.slice(0, 1))
    > reconstructionError(centred, deflated.components.slice(0, 2)),
    "more components mean less error");

  // a short run must leave more error than a long one on a gap-dominated problem.
  const shortRun = powerIteration(cov, 3, 0.0);
  const longRun = powerIteration(cov, 200, 0.0);
  assert.ok(residualOf(cov, shortRun.vector, shortRun.eigenvalue)
    > residualOf(cov, longRun.vector, longRun.eigenvalue), "power iteration refines with t");
  assert.ok(longRun.iterations <= MAX_ITER);
  assert.ok(Math.abs(longRun.eigenvalue - desc[0]) <= 1e-9, "long run reaches lambda_1");
}

module.exports = { MAX_ITER, RAYLEIGH_TOL, lcgUnit, nextNormal, dot, norm, normalize, matVec, mean,
  center, covariance, rayleigh, canonicalize, unitFromSeed, powerIteration, projectOut,
  pcaDeflation, pcaOrthogonalized, jacobiEigh, lcgBasis, plantedCloud, explainedRatio,
  reconstructionError };

if (require.main === module) {
  main();
  console.log("PASS");
}
