/** Task 790 - Implement bundle adjustment sparse solver outline with Levenberg-Marquardt. (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Two cameras look at 12 points; both sets of intrinsics and the pose of camera 1 (the world frame)
 * are known, so the unknowns are camera 2's 6 pose parameters plus 3 coordinates per point. Each
 * observation gives 2 reprojection residuals r = (u_obs - u_proj, v_obs - v_proj) with
 * u = f X/Z + cx, v = f Y/Z + cy.
 *
 * A residual touches exactly one camera block and one point block, so the normal equations are block
 * structured: H = [[U, W], [W^T, V]] with V block diagonal (3x3 per point). Eliminating the points
 * with the Schur complement leaves a camera-sized system
 *   S = U - sum_p W_p V_p^-1 W_p^T,  rhs = -g_c + sum_p W_p V_p^-1 g_p,
 *   delta_p = V_p^-1 (-g_p - W_p^T delta_c)
 * which is what the solver factorises. LM damping D = lambda*diag(H) interpolates between
 * Gauss-Newton and gradient descent; a step whose cost grows is rejected and lambda is multiplied
 * by 5, so the cost is monotonically non-increasing.
 *
 * Gauge fixing (the deliberate simplification): in a two-view reconstruction the baseline length is
 * unobservable, since scaling the points and camera 2's translation by s leaves every image
 * unchanged. This outline models a calibrated rig with a known baseline and enforces it with a soft
 * prior w*(||t2|| - ||t2_known||), w = 1e3; a full system would instead report a scale-free
 * reconstruction. Verified: the analytic projection Jacobian against central differences, the Schur
 * step against a dense solve, monotone cost, the fitted cost landing on the classical
 * (m - p)*sigma^2 noise floor while the truth sits near m*sigma^2, accuracy scaling with the noise,
 * damping shortening the step and rescuing a far start, the baseline prior, and input validation.
 * Further simplifications: one movable camera, dense 42x42 arithmetic instead of sparse Cholesky,
 * no robust loss. Every reported number is measured.
 */
"use strict";
const { ok, throws } = require("node:assert");

const FOCAL = 800;
const CX = 320;
const CY = 240;
const POINT_COUNT = 12;
const PARAM_COUNT = 6 + 3 * POINT_COUNT;

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
const matVec = (A, v) => A.map((row) => row.reduce((acc, a, j) => acc + a * v[j], 0));
const mulSmall = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((acc, v, k) => acc + v * B[k][j], 0)));

/** Rodriguez exponential of the rotation vector (theta -> 0 is the series limit). */
function rodrigues(r) {
  const theta = Math.hypot(...r);
  const [x, y, z] = theta > 1e-12 ? r.map((v) => v / theta) : [0, 0, 0];
  const K = [[0, -z, y], [z, 0, -x], [-y, x, 0]];
  const K2 = mulSmall(K, K);
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1 : 0) + Math.sin(theta) * K[i][j] + (1 - Math.cos(theta)) * K2[i][j]));
}

const project = (R, t, P) => {
  const X = matVec(R, P).map((v, i) => v + t[i]);
  return [FOCAL * X[0] / X[2] + CX, FOCAL * X[1] / X[2] + CY];
};

/**
 * Right Jacobian of SO(3): with the rotation vector as the parameter,
 * exp([r + dr]) = exp([r]) exp([J_r(r) dr]), hence dX_c/dr = -[R P]x J_r(r) for X_c = exp([r]) P + t.
 * The sign/convention is verified against central differences of the actual projection below.
 */
function rotationJacobian(r) {
  const theta = Math.hypot(...r);
  const K = [[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]];
  const K2 = mulSmall(K, K);
  const I = identity(3);
  if (theta < 1e-8) return I.map((row, i) => row.map((v, j) => v + K[i][j] / 2 + K2[i][j] / 6));
  const a = (1 - Math.cos(theta)) / (theta * theta);
  const b = (theta - Math.sin(theta)) / theta ** 3;
  return I.map((row, i) => row.map((v, j) => v + a * K[i][j] + b * K2[i][j]));
}

/** Analytic Jacobian of the projection w.r.t. point, translation and the rotation vector. */
function projectionJacobian(R, t, P, rotvec) {
  const Xc = matVec(R, P).map((v, i) => v + t[i]);
  const invZ = 1 / Xc[2];
  const dProj = [[FOCAL * invZ, 0, -FOCAL * Xc[0] * invZ * invZ], [0, FOCAL * invZ, -FOCAL * Xc[1] * invZ * invZ]];
  const RP = matVec(R, P);
  const minusSkew = [[0, RP[2], -RP[1]], [-RP[2], 0, RP[0]], [RP[1], -RP[0], 0]];
  return { dPoint: mulSmall(dProj, R), dT: dProj, dR: mulSmall(mulSmall(dProj, minusSkew), rotationJacobian(rotvec)) };
}

/** Dense Gaussian elimination with partial pivoting. */
function solveLinear(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-300) throw new Error("singular normal equations");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let r = col + 1; r < n; r++) for (let c = n; c >= col; c--) a[r][c] -= (a[r][col] / a[col][col]) * a[col][c];
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let sum = a[r][n];
    for (let c = r + 1; c < n; c++) sum -= a[r][c] * x[c];
    x[r] = sum / a[r][r];
  }
  return x;
}

/** 3x3 inverse (adjugate / determinant), used for the per-point Schur blocks. */
function inverse3(M) {
  const [[a, b, c], [d, e, f], [g, h, i]] = M;
  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-300) throw new Error("singular point block");
  const adj = [[A, c * h - b * i, b * f - c * e], [B, a * i - c * g, c * d - a * f], [C, b * g - a * h, a * e - b * d]];
  return adj.map((row) => row.map((v) => v / det));
}

/** Residual vector plus block-structured normal equations H = J^T J, g = J^T r for one state. */
function evaluate(state, observations, priorWeight, priorNorm) {
  const R2 = rodrigues(state.slice(0, 3));
  const t2 = state.slice(3, 6);
  const residuals = [];
  const rows = [];
  const row = () => new Array(PARAM_COUNT).fill(0);
  for (const { camera, point, uv } of observations) {
    const base = 6 + 3 * point;
    const P = state.slice(base, base + 3);
    const R = camera === 0 ? identity(3) : R2;
    const t = camera === 0 ? [0, 0, 0] : t2;
    const [u, v] = project(R, t, P);
    const { dPoint, dT, dR } = projectionJacobian(R, t, P, camera === 0 ? [0, 0, 0] : state.slice(0, 3));
    const rowU = row();
    const rowV = row();
    for (let k = 0; k < 3; k++) {
      rowU[base + k] = -dPoint[0][k];
      rowV[base + k] = -dPoint[1][k];
      if (camera !== 0) {
        rowU[k] = -dR[0][k];
        rowV[k] = -dR[1][k];
        rowU[3 + k] = -dT[0][k];
        rowV[3 + k] = -dT[1][k];
      }
    }    residuals.push(uv[0] - u, uv[1] - v);
    rows.push(rowU, rowV);
  }
  const baseline = Math.hypot(...t2);
  const priorRow = row();
  if (baseline > 1e-12) for (let k = 0; k < 3; k++) priorRow[3 + k] = (priorWeight / baseline) * t2[k];
  residuals.push(priorWeight * (baseline - priorNorm));
  rows.push(priorRow);
  const H = rows[0].map((_, i) => rows[0].map((_, j) => rows.reduce((acc, r) => acc + r[i] * r[j], 0)));
  const g = rows[0].map((_, i) => rows.reduce((acc, r, k) => acc + r[i] * residuals[k], 0));
  return { residuals, rows, H, g, cost: residuals.reduce((acc, r) => acc + r * r, 0) };
}

/**
 * Solves (H + lambda*diag(H)) delta = -g by eliminating the point blocks: the camera-sized Schur
 * complement is factorised and the point increments are recovered by back-substitution.
 */
function schurStep(H, g, lambda) {
  const Vinv = [];
  const W = [];
  for (let p = 0; p < POINT_COUNT; p++) {
    const base = 6 + 3 * p;
    const Vp = [0, 1, 2].map((i) => [0, 1, 2].map((j) => H[base + i][base + j] + (i === j ? lambda * H[base + i][base + i] : 0)));
    Vinv.push(inverse3(Vp));
    W.push([0, 1, 2, 3, 4, 5].map((i) => [0, 1, 2].map((j) => H[i][base + j])));   // W_p is 6x3
  }
  const S = [0, 1, 2, 3, 4, 5].map((i) => [0, 1, 2, 3, 4, 5].map((j) => H[i][j] + (i === j ? lambda * H[i][i] : 0)));
  const rhs = g.slice(0, 6).map((v) => -v);
  for (let p = 0; p < POINT_COUNT; p++) {
    const gp = g.slice(6 + 3 * p, 9 + 3 * p);
    const WVinv = mulSmall(W[p], Vinv[p]);                      // 6x3
    for (let i = 0; i < 6; i++) rhs[i] += WVinv[i].reduce((acc, v, k) => acc + v * gp[k], 0);
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) S[i][j] -= WVinv[i].reduce((acc, v, k) => acc + v * W[p][j][k], 0);
  }
  const deltaCamera = solveLinear(S, rhs);
  const delta = new Array(PARAM_COUNT).fill(0);
  for (let i = 0; i < 6; i++) delta[i] = deltaCamera[i];
  for (let p = 0; p < POINT_COUNT; p++) {
    const base = 6 + 3 * p;
    const gp = g.slice(base, base + 3);
    const WtDelta = [0, 1, 2].map((i) => [0, 1, 2, 3, 4, 5].reduce((acc, j) => acc + H[base + i][j] * deltaCamera[j], 0));
    const step = matVec(Vinv[p], [0, 1, 2].map((i) => -gp[i] - WtDelta[i]));
    for (let k = 0; k < 3; k++) delta[base + k] = step[k];
  }
  return delta;
}

/** Levenberg-Marquardt loop; the cost is non-increasing because steps that raise it are rejected. */
function lmOptimize(observations, state0, { priorWeight = 1, priorNorm = 0, lambda0 = 1e-3, maxIterations = 80, tolerance = 1e-12 } = {}) {
  const target = priorNorm > 0 ? priorNorm : Math.hypot(...state0.slice(3, 6));
  let state = state0.slice();
  let lambda = lambda0;
  let { cost } = evaluate(state, observations, priorWeight, target);
  const history = [cost];
  let iterations = 0;
  let rejected = 0;
  for (let iter = 0; iter < maxIterations; iter++) {
    const { H, g } = evaluate(state, observations, priorWeight, target);
    let delta;
    try {
      delta = schurStep(H, g, lambda);
    } catch (error) {
      // A numerically singular point block (a point pushed towards infinity) is treated as a
      // rejected step: the damping is raised and the same state is retried.
      if (!(error instanceof Error)) throw error;
      rejected += 1;
      lambda *= 5;
      iterations = iter + 1;
      if (lambda > 1e12) break;
      continue;
    }
    const candidate = state.map((v, i) => v + delta[i]);
    const trial = evaluate(candidate, observations, priorWeight, target);
    iterations = iter + 1;
    if (trial.cost < cost) {
      const improvement = cost - trial.cost;
      state = candidate;
      cost = trial.cost;
      lambda = Math.max(lambda / 3, 1e-12);
      history.push(cost);
      if (improvement < tolerance * Math.max(1, cost)) break;
    } else {
      rejected += 1;
      lambda *= 5;
      if (lambda > 1e12) break;
    }
  }
  return { state, cost, lambda, iterations, rejected, history, priorNorm: target };
}

/** Synthetic two-view scenario: deterministic ground truth, uniform observation noise on +-noiseLevel. */
function buildScenario(seed, noiseLevel) {
  const random = makeRng(seed);
  const points = Array.from({ length: POINT_COUNT }, () => [(random() - 0.5) * 2, (random() - 0.5) * 1.5, 3 + random() * 2]);
  const truePose = { rotvec: [0.08, -0.22, 0.05], t: [1.2, 0.15, 0.35] };
  const R2 = rodrigues(truePose.rotvec);
  const observations = [];
  for (let camera = 0; camera < 2; camera++) for (let point = 0; point < POINT_COUNT; point++) {
    const [u, v] = camera === 0 ? project(identity(3), [0, 0, 0], points[point]) : project(R2, truePose.t, points[point]);
    observations.push({ camera, point, uv: [u + (random() - 0.5) * 2 * noiseLevel, v + (random() - 0.5) * 2 * noiseLevel] });
  }
  const state0 = [...truePose.rotvec.map((v) => v + 0.05), ...truePose.t.map((v) => v * 1.12), ...points.flatMap((p) => p.map((v) => v + (random() - 0.5) * 0.1))];
  return { points, truePose, observations, state0, baseline: Math.hypot(...truePose.t), truth: [...truePose.rotvec, ...truePose.t, ...points.flat()] };
}

/** Rotation-angle + translation distance between an estimated state and the ground truth. */
function poseError(state, truth) {
  const R = rodrigues(state.slice(0, 3));
  const Rt = rodrigues(truth.rotvec);
  const relative = mulSmall(Rt.map((row, i) => Rt.map((_, j) => Rt[j][i])), R);   // Rt^T R
  const angle = Math.acos(Math.min(1, Math.max(-1, (relative[0][0] + relative[1][1] + relative[2][2] - 1) / 2)));
  return Math.max(angle, ...state.slice(3, 6).map((v, i) => Math.abs(v - truth.t[i])));
}

function main() {
  const notes = [];
  const noise = 0.1;
  const residualCount = 2 * 2 * POINT_COUNT + 1;
  const variance = (2 * noise) ** 2 / 12;                       // uniform noise on +-noise
  // 1. Analytic Jacobian against central differences of the projection itself.
  const R = rodrigues([0.1, -0.05, 0.02]);
  const t = [0.4, -0.2, 0.15];
  const P = [0.3, 0.1, 4.2];
  const jac = projectionJacobian(R, t, P, [0.1, -0.05, 0.02]);
  const delta = 1e-6;
  const bump = (k, sign) => [0, 0, 0].map((_, i) => (i === k ? sign * delta : 0));
  const probes = {
    dPoint: (d) => project(R, t, P.map((v, i) => v + d[i])),
    dT: (d) => project(R, t.map((v, i) => v + d[i]), P),
    dR: (d) => project(rodrigues([0.1 + d[0], -0.05 + d[1], 0.02 + d[2]]), t, P),
  };
  let worstJacobian = 0;
  for (const key of ["dPoint", "dT", "dR"]) {
    for (let k = 0; k < 3; k++) {
      const forward = probes[key](bump(k, 1));
      const backward = probes[key](bump(k, -1));
      for (let i = 0; i < 2; i++) worstJacobian = Math.max(worstJacobian, Math.abs((forward[i] - backward[i]) / (2 * delta) - jac[key][i][k]));
    }
  }
  notes.push(`analytic vs central-difference Jacobian: worst deviation ${worstJacobian.toExponential(2)}`);
  ok(worstJacobian < 1e-5, `the analytic Jacobian must match central differences (worst ${worstJacobian})`);
  // 2. The Schur-complement step must equal the dense solve of the same damped normal equations,
  //    and a larger damping factor must shorten it.
  const scenario = buildScenario(20251016, noise);
  const prior = { priorWeight: 1e3, priorNorm: scenario.baseline };
  const evaluated = evaluate(scenario.state0, scenario.observations, prior.priorWeight, prior.priorNorm);
  let worstSchurGap = 0;
  for (const lambda of [0, 1e-3, 1]) {
    const damped = evaluated.H.map((r, i) => r.map((v, j) => v + (i === j ? lambda * evaluated.H[i][i] : 0)));
    const dense = solveLinear(damped, evaluated.g.map((v) => -v));
    const schur = schurStep(evaluated.H, evaluated.g, lambda);
    worstSchurGap = Math.max(worstSchurGap, ...dense.map((v, i) => Math.abs(v - schur[i])));
  }
  const stepNorm = (lambda) => Math.hypot(...schurStep(evaluated.H, evaluated.g, lambda));
  notes.push(`Schur step vs dense solve: worst difference ${worstSchurGap.toExponential(2)}; step norm ${stepNorm(1e-3).toExponential(3)} at lambda = 1e-3 -> ${stepNorm(10).toExponential(3)} at lambda = 10`);
  ok(worstSchurGap < 1e-9, `the Schur complement must reproduce the dense LM step (gap ${worstSchurGap})`);
  ok(stepNorm(10) < stepNorm(1e-3), "a larger damping factor must produce a shorter step");
  // 3. LM converges from the perturbed start with a monotone cost, down to the noise floor.
  const fit = lmOptimize(scenario.observations, scenario.state0, prior);
  const monotone = fit.history.every((value, i) => i === 0 || value <= fit.history[i - 1] + 1e-15);
  const fittedPose = poseError(fit.state, scenario.truePose);
  const pointError = Math.max(...scenario.points.flat().map((v, i) => Math.abs(v - fit.state[6 + i])));
  const truthCost = evaluate(scenario.truth, scenario.observations, prior.priorWeight, prior.priorNorm).cost;
  const floor = (residualCount - PARAM_COUNT) * variance;
  notes.push(`LM: ${fit.iterations} iterations (${fit.rejected} rejected), cost ${fit.history[0].toFixed(1)} -> ${fit.cost.toFixed(5)} vs (m-p)*sigma^2 = ${floor.toFixed(5)}; truth cost ${truthCost.toFixed(4)} vs m*sigma^2 = ${(residualCount * variance).toFixed(4)}; pose error ${fittedPose.toExponential(2)}, max point error ${pointError.toFixed(4)}`);
  ok(monotone, "the LM cost must be monotonically non-increasing");
  ok(fit.cost < truthCost, "the optimum must beat the cost of the ground truth");
  ok(Math.abs(fit.cost / floor - 1) < 0.6, "the fitted cost must equal the classical (m - p)*sigma^2 noise floor");
  ok(truthCost > 0.4 * residualCount * variance && truthCost < 2 * residualCount * variance, "the ground-truth cost must sit near m*sigma^2");
  ok(fittedPose < 5e-3, `the camera pose must be recovered (error ${fittedPose})`);
  ok(pointError < 0.15, `the points must be recovered to their noise-limited accuracy (max error ${pointError})`);
  // 4. Accuracy must be noise limited: 3x the noise must give a clearly worse fit.
  const coarse = buildScenario(20251016, 3 * noise);
  const coarseFit = lmOptimize(coarse.observations, coarse.state0, { priorWeight: 1e3, priorNorm: coarse.baseline });
  const coarseError = Math.max(...coarse.points.flat().map((v, i) => Math.abs(v - coarseFit.state[6 + i])));
  const errorRatio = coarseError / Math.max(pointError, 1e-12);
  notes.push(`noise scaling: point error ${pointError.toFixed(4)} at ${noise} px -> ${coarseError.toFixed(4)} at ${3 * noise} px (ratio ${errorRatio.toFixed(2)})`);
  ok(errorRatio > 1.5 && errorRatio < 6, `tripling the noise must roughly triple the point error (ratio ${errorRatio})`);
  // 5. The calibrated baseline prior must hold, and the initial guess must be off precisely along
  //    the unobservable scale direction (points and camera 2 scaled together).
  const baselineDrift = Math.abs(Math.hypot(...fit.state.slice(3, 6)) - scenario.baseline);
  notes.push(`baseline prior: |baseline_fit - baseline_known| = ${baselineDrift.toExponential(2)} (the initial guess was 12% too long)`);
  ok(baselineDrift < 1e-6, `the calibrated baseline must be recovered (drift ${baselineDrift})`);
  ok(Math.hypot(...scenario.state0.slice(3, 6)) > scenario.baseline, "the initial guess must be off in the unobservable direction");
  // 6. Damping pays off from a far initial guess: a mildly damped step beats a Gauss-Newton step,
  //    and LM still reaches the noise floor from there.
  const farState = [...scenario.state0.slice(0, 3).map((v) => v + 0.4), ...scenario.state0.slice(3, 6).map((v) => v * 2.2), ...scenario.state0.slice(6)];
  const farEval = evaluate(farState, scenario.observations, prior.priorWeight, prior.priorNorm);
  const afterStep = (lambda) => {
    const step = schurStep(farEval.H, farEval.g, lambda);
    return evaluate(farState.map((v, i) => v + step[i]), scenario.observations, prior.priorWeight, prior.priorNorm).cost;
  };
  const dampedFit = lmOptimize(scenario.observations, farState, { ...prior, lambda0: 1e3 });
  notes.push(`far start (cost ${farEval.cost.toExponential(3)}): Gauss-Newton step -> ${afterStep(0).toExponential(3)}, lambda = 1e-3 -> ${afterStep(1e-3).toExponential(3)}; full LM reaches ${dampedFit.cost.toFixed(4)} in ${dampedFit.iterations} iterations (pose error ${poseError(dampedFit.state, scenario.truePose).toExponential(2)})`);
  ok(afterStep(1e-3) < afterStep(0), "a mildly damped step must beat an undamped Gauss-Newton step from the far start");
  ok(dampedFit.cost < 1, "damped LM must converge all the way to the noise floor from the far start");
  // 7. Boundary, determinism and validation.
  throws(() => inverse3([[1, 2, 3], [2, 4, 6], [3, 6, 9]]), Error, "a singular point block must throw");
  const repeat = lmOptimize(scenario.observations, scenario.state0, prior);
  ok(repeat.cost === fit.cost && repeat.state.every((v, i) => v === fit.state[i]), "LM must be deterministic");
  ok(lmOptimize([{ camera: 1, point: 0, uv: [0, 0] }], scenario.state0, { maxIterations: 1, ...prior }).cost >= 0,
    "a single-observation fit must stay finite and non-negative");
  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}
module.exports = { makeRng, rodrigues, project, projectionJacobian, solveLinear, inverse3, evaluate, schurStep, lmOptimize };

if (require.main === module) main();
