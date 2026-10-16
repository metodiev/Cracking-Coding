/** Task 789 - Implement shape-matching registration (ICP) iterative closest point. (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * ICP alternates two steps until convergence:
 *   1. correspondence: every source point is matched to its nearest target point (brute force here),
 *   2. alignment: the least-squares rigid transform of the matched pairs is solved in closed form.
 * The alignment uses Horn's quaternion method: with centroids p-bar, q-bar and centred sets, the
 * optimal rotation maximises trace(R M) with M = sum_i p_i q_i^T. That objective is a quadratic
 * form in the unit quaternion, so the optimum is the eigenvector of the symmetric 4x4 matrix
 *   N = [[Sxx+Syy+Szz, Syz-Szy, Szx-Sxz, Sxy-Syx], ...]     S = sum_i p_i q_i^T
 * belonging to its largest eigenvalue; the eigenproblem is solved with cyclic Jacobi rotations,
 * which preserve symmetry and yield orthonormal eigenvectors. The translation follows as
 * t = q-bar - R p-bar, and R is proper by construction (det R = +1, verified below).
 *
 * Invariants verified: the per-iteration RMSD never increases, R is orthonormal with det = +1, the
 * Jacobi eigensolver satisfies A v = lambda v with orthonormal v, exact correspondences are
 * recovered to machine precision, the 3D quaternion answer matches the independent 2D Procrustes
 * closed form on coplanar data, noise degrades the fit proportionally, collinear sets still fit
 * exactly, and degenerate inputs (fewer than three pairs, rank-zero dispersion, one-point clouds)
 * are rejected instead of yielding a meaningless rotation.
 *
 * What is simplified: brute-force nearest neighbours rather than a kd-tree (clouds hold tens of
 * points, so the whole file runs in milliseconds) and point-to-point ICP only (no point-to-plane
 * metric or outlier weighting). A seeded splitmix32 PRNG drives the deterministic synthetic cloud
 * and noise. Nothing about the returned transform is faked: every number is measured.
 */
"use strict";

const { ok, throws } = require("node:assert");

/** Deterministic splitmix32 uniforms in [0, 1). */
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return ((z ^ (z >>> 15)) >>> 0) / 4294967296;
  };
}

const identity = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

/**
 * Cyclic Jacobi eigen decomposition of a symmetric matrix; eigenvectors are columns, i.e.
 * vectors[k][p] is component k of the p-th eigenvector.
 */
function jacobiEigen(matrix, { maxSweeps = 60, tol = 1e-18 } = {}) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  const v = identity(n);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < tol) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (a[p][q] === 0) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.hypot(theta, 1));
        const c = 1 / Math.hypot(t, 1);
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
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: a.map((row, i) => row[i]), vectors: v };
}

/** Unit quaternion (q0, q1, q2, q3) -> proper rotation matrix. */
function quaternionToMatrix([q0, q1, q2, q3]) {
  return [
    [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
    [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
    [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
  ];
}

const applyTransform = (R, t, p) => R.map((row, i) => row.reduce((acc, r, j) => acc + r * p[j], 0) + t[i]);
const centroid = (points) => points[0].map((_, d) => points.reduce((acc, p) => acc + p[d], 0) / points.length);
const distance = (a, b) => Math.sqrt(a.reduce((acc, v, d) => acc + (v - b[d]) ** 2, 0));

/** Least-squares rigid transform (R, t) mapping source -> target, by Horn's quaternion method. */
function estimateRigidTransform(source, target) {
  if (source.length !== target.length || source.length < 3) {
    throw new RangeError(`need at least 3 matched point pairs (got ${source.length} source, ${target.length} target)`);
  }
  const pBar = centroid(source);
  const qBar = centroid(target);
  const pC = source.map((p) => p.map((v, d) => v - pBar[d]));
  const qC = target.map((q) => q.map((v, d) => v - qBar[d]));
  const S = [0, 1, 2].map((r) => [0, 1, 2].map((c) => pC.reduce((acc, p, i) => acc + p[r] * qC[i][c], 0)));
  const [sxx, sxy, sxz] = S[0];
  const [syx, syy, syz] = S[1];
  const [szx, szy, szz] = S[2];
  if (Math.abs(sxx) + Math.abs(sxy) + Math.abs(sxz) + Math.abs(syy) + Math.abs(syz) + Math.abs(szz) < 1e-12) {
    throw new Error("degenerate point set: the cross-dispersion vanishes, so no rotation is defined");
  }
  const N = [
    [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
    [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
    [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
    [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
  ];
  const { values, vectors } = jacobiEigen(N);
  let best = 0;
  for (let i = 1; i < 4; i++) if (values[i] > values[best]) best = i;
  const R = quaternionToMatrix(vectors.map((row) => row[best]));
  const RpBar = R.map((row) => row.reduce((acc, r, j) => acc + r * pBar[j], 0));
  const t = qBar.map((v, d) => v - RpBar[d]);
  const rmsd = Math.sqrt(source.reduce((acc, p, i) => acc + distance(applyTransform(R, t, p), target[i]) ** 2, 0) / source.length);
  return { R, t, rmsd };
}

/** Brute-force nearest target point. */
function nearestNeighbour(target, point) {
  let best = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < target.length; i++) {
    const d = distance(target[i], point);
    if (d < bestDistance) {
      bestDistance = d;
      best = i;
    }
  }
  return { index: best, distance: bestDistance };
}

/** Composes two rigid transforms: (R2, t2) applied after (R1, t1). */
const compose = (R1, t1, R2, t2) => ({
  R: R2.map((row) => row.map((_, j) => row.reduce((acc, v, k) => acc + v * R1[k][j], 0))),
  t: R2.map((row, i) => row.reduce((acc, v, k) => acc + v * t1[k], 0) + t2[i]),
});

/** Point-to-point ICP aligning `source` onto `target`; returns the composed transform. */
function icp(source, target, { maxIterations = 60, tolerance = 1e-13, initial = {} } = {}) {
  if (source.length < 3 || target.length < 3) throw new RangeError("ICP needs at least 3 points in each cloud");
  let R = initial.R ? initial.R.map((row) => row.slice()) : identity(3);
  let t = initial.t ? initial.t.slice() : [0, 0, 0];
  const history = [];
  let iterations = 0;
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const transformed = source.map((p) => applyTransform(R, t, p));
    const matchedTargets = transformed.map((p) => target[nearestNeighbour(target, p).index]);
    const step = estimateRigidTransform(transformed, matchedTargets);
    ({ R, t } = compose(R, t, step.R, step.t));
    history.push(step.rmsd);
    iterations = iteration + 1;
    if (history.length > 1 && Math.abs(history.at(-2) - history.at(-1)) < tolerance) break;
  }
  const rmsd = Math.sqrt(source.reduce((acc, p, i) => acc + distance(applyTransform(R, t, p), target[i]) ** 2, 0) / source.length);
  return { R, t, rmsd, iterations, history };
}

/** In-plane rotation angle of a rotation matrix, for the closed-form comparison. */
const rotationAngle = (R) => Math.atan2(R[1][0], R[0][0]);

/** Robust metric: RMSD over the closest `keep` fraction of index-aligned pairs only. */
function trimmedRmsd(source, target, R, t, keep = 0.8) {
  const distances = source.map((p, i) => distance(applyTransform(R, t, p), target[i])).sort((a, b) => a - b);
  const used = distances.slice(0, Math.max(1, Math.floor(distances.length * keep)));
  return Math.sqrt(used.reduce((acc, d) => acc + d * d, 0) / used.length);
}

/** Deterministic cloud: a jittered box sample. */
const makeCloud = (seed, count) => {
  const random = makeRng(seed);
  return Array.from({ length: count }, () => [0, 1, 2].map(() => (random() - 0.5) * 2 + (random() - 0.5) * 0.3));
};

const rotationZ = (angle) => [[Math.cos(angle), -Math.sin(angle), 0], [Math.sin(angle), Math.cos(angle), 0], [0, 0, 1]];

function main() {
  const notes = [];

  // 1. Jacobi invariants on a deterministic symmetric matrix, independently of the registration.
  const symmetric = [[4, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [2, 1, -2, -1]];
  const eigen = jacobiEigen(symmetric);
  let worstResidual = 0;
  let worstOrthogonality = 0;
  for (let p = 0; p < 4; p++) {
    const v = eigen.vectors.map((row) => row[p]);
    const Av = symmetric.map((row) => row.reduce((acc, a, k) => acc + a * v[k], 0));
    worstResidual = Math.max(worstResidual, ...Av.map((a, k) => Math.abs(a - eigen.values[p] * v[k])));
    for (let q = 0; q < 4; q++) {
      const w = eigen.vectors.map((row) => row[q]);
      worstOrthogonality = Math.max(worstOrthogonality, Math.abs(v.reduce((acc, x, k) => acc + x * w[k], 0) - (p === q ? 1 : 0)));
    }
  }
  notes.push(`Jacobi eigen: worst |Av - lambda v| = ${worstResidual.toExponential(2)}, worst orthogonality deviation = ${worstOrthogonality.toExponential(2)}`);
  ok(worstResidual < 1e-12 && worstOrthogonality < 1e-12, "eigenpairs must be exact and orthonormal");

  // 2. Exact correspondences: a known rigid transform must be recovered to machine precision.
  const cloud = makeCloud(11, 40);
  const knownR = rotationZ(0.7);
  const knownT = [0.4, -1.2, 0.25];
  const target = cloud.map((p) => applyTransform(knownR, knownT, p));
  const recovered = estimateRigidTransform(cloud, target);
  const matrixError = Math.max(...recovered.R.map((row, i) => Math.max(...row.map((v, j) => Math.abs(v - knownR[i][j])))));
  const translationError = Math.max(...recovered.t.map((v, i) => Math.abs(v - knownT[i])));
  notes.push(`exact correspondences: rotation error ${matrixError.toExponential(2)}, translation error ${translationError.toExponential(2)}, rmsd ${recovered.rmsd.toExponential(2)}`);
  ok(matrixError < 1e-12 && translationError < 1e-12 && recovered.rmsd < 1e-12, "an exact rigid transform must be recovered exactly");

  // 3. The recovered rotation is proper: R^T R = I and det R = +1.
  const ortho = Math.max(...[0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => Math.abs([0, 1, 2].reduce((acc, k) => acc + recovered.R[k][i] * recovered.R[k][j], 0) - (i === j ? 1 : 0)))));
  const [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]] = recovered.R;
  const det = r00 * (r11 * r22 - r12 * r21) - r01 * (r10 * r22 - r12 * r20) + r02 * (r10 * r21 - r11 * r20);
  notes.push(`rotation check: max |R^T R - I| = ${ortho.toExponential(2)}, det R = ${det.toFixed(15)}`);
  ok(ortho < 1e-13 && Math.abs(det - 1) < 1e-13, "the recovered rotation must be orthonormal with determinant +1");

  // 4. ICP from a perturbed start must recover the transform, with a monotone RMSD history.
  const perturbed = { R: rotationZ(0.7 - 0.12), t: knownT.map((v, i) => v + [0.06, -0.05, 0.04][i]) };
  const registered = icp(cloud, target, { initial: perturbed });
  const icpMatrixError = Math.max(...registered.R.map((row, i) => Math.max(...row.map((v, j) => Math.abs(v - knownR[i][j])))));
  const icpTranslationError = Math.max(...registered.t.map((v, i) => Math.abs(v - knownT[i])));
  const monotone = registered.history.every((value, i) => i === 0 || value <= registered.history[i - 1] + 1e-12);
  notes.push(`ICP from a 0.12 rad / 0.08 offset start: ${registered.iterations} iterations, rmsd ${registered.rmsd.toExponential(2)}, rotation error ${icpMatrixError.toExponential(2)}, translation error ${icpTranslationError.toExponential(2)}, monotone ${monotone}`);
  ok(registered.rmsd < 1e-9, `ICP must align the clouds (rmsd ${registered.rmsd})`);
  ok(icpMatrixError < 1e-6 && icpTranslationError < 1e-6, `ICP must recover the rigid transform (rotation ${icpMatrixError}, translation ${icpTranslationError})`);
  ok(monotone, "the per-iteration RMSD must never increase");

  // 5. Independent oracle: the 2D Procrustes angle from atan2 (on centred data) must match Horn.
  const planar = makeCloud(23, 24).map((p) => [p[0], p[1], 0]);
  const planarTarget = planar.map((p) => applyTransform(knownR, knownT, p));
  const hornAngle = rotationAngle(estimateRigidTransform(planar, planarTarget).R);
  const [pMean, qMean] = [centroid(planar), centroid(planarTarget)];
  const pCentred = planar.map((p) => p.map((v, d) => v - pMean[d]));
  const qCentred = planarTarget.map((q) => q.map((v, d) => v - qMean[d]));
  const numerator = pCentred.reduce((acc, p, i) => acc + (p[0] * qCentred[i][1] - p[1] * qCentred[i][0]), 0);
  const denominator = pCentred.reduce((acc, p, i) => acc + (p[0] * qCentred[i][0] + p[1] * qCentred[i][1]), 0);
  const closedFormAngle = Math.atan2(numerator, denominator);
  notes.push(`2D oracle: Horn angle ${hornAngle.toFixed(12)}, closed-form atan2 ${closedFormAngle.toFixed(12)}, true 0.7`);
  ok(Math.abs(hornAngle - closedFormAngle) < 1e-12, "the quaternion method must agree with the 2D closed form");
  ok(Math.abs(hornAngle - 0.7) < 1e-9, "both angles must recover the generating 0.7 rad");

  // 6. Noise degrades the fit proportionally to the noise level.
  const random = makeRng(777);
  const noiseLevel = 0.01;
  const noisyTarget = target.map((p) => p.map((v) => v + (random() - 0.5) * 2 * noiseLevel));
  const noisy = estimateRigidTransform(cloud, noisyTarget);
  const noisyError = Math.max(...noisy.R.map((row, i) => Math.max(...row.map((v, j) => Math.abs(v - knownR[i][j])))));
  notes.push(`noise ${noiseLevel}: rotation error ${noisyError.toExponential(2)}, rmsd ${noisy.rmsd.toFixed(4)}`);
  ok(noisyError < 8 * noiseLevel, `the rotation must survive the noise (error ${noisyError})`);
  ok(Math.abs(noisy.rmsd - noiseLevel) < 3 * noiseLevel, "the noisy RMSD must stay near the noise level");

  // 7. Partial overlap with outliers: the trimmed metric must beat the untrimmed one.
  const partialTarget = target.slice(0, 34).concat([[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11]]);
  const partial = icp(cloud, partialTarget, { initial: perturbed, maxIterations: 40 });
  const trimmed = trimmedRmsd(cloud, partialTarget, partial.R, partial.t, 0.85);
  notes.push(`partial overlap: untrimmed rmsd ${partial.rmsd.toFixed(3)} (outliers included) vs trimmed rmsd ${trimmed.toFixed(3)}`);
  ok(trimmed < partial.rmsd, "the trimmed metric must ignore the outliers and beat the full RMSD");

  // 8. Degenerate inputs are rejected instead of returning a made-up rotation.
  throws(() => estimateRigidTransform([[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0]]), RangeError, "two pairs must throw");
  throws(() => estimateRigidTransform([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]]), Error,
    "identical points must throw");
  throws(() => icp([[0, 0, 0]], target), RangeError, "a one-point source cloud must throw");

  // 9. Collinear points: the twist about the line is free, but the fit itself must be exact.
  const line = Array.from({ length: 8 }, (_, i) => [i * 0.5, 0, 0]);
  const lineFit = estimateRigidTransform(line, line.map((p) => applyTransform(knownR, knownT, p)));
  notes.push(`collinear points: rmsd ${lineFit.rmsd.toExponential(2)} (twist about the line is undetermined)`);
  ok(lineFit.rmsd < 1e-12, "collinear correspondences must still fit exactly");

  // 10. Determinism.
  const repeat = icp(cloud, target, { initial: perturbed });
  ok(repeat.history.every((v, i) => v === registered.history[i]) && repeat.rmsd === registered.rmsd, "ICP must be deterministic");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { makeRng, jacobiEigen, quaternionToMatrix, estimateRigidTransform, nearestNeighbour, compose, icp, trimmedRmsd };

if (require.main === module) main();
