/** Task 787 - Implement adaptive step-size ODE integrator (Dormand-Prince outline). (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Dormand-Prince RK45: a 7-stage explicit Runge-Kutta pair with the FSAL property - the last
 * stage k7 = f(t + h, y5) is exactly the first stage of the next step, so an accepted step costs
 * six new evaluations. The 4th-order companion shares the stages, hence
 *   err = h * sum_s e_s k_s,  e = b5 - b4 = [71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40]
 * is a free local-error estimate, and the step size follows the standard controller
 *   h_new = h * clamp(0.9 * errNorm^(-1/5), 0.2, 5)
 * with the scaled RMS norm errNorm^2 = mean( (err_i / (atol + rtol*max(|y_i|,|y5_i|)))^2 ).
 *
 * Invariants verified here: every *accepted* step satisfies errNorm <= 1 (error control really is
 * enforced), t advances monotonically and never overshoots tEnd, the stage budget is exactly six
 * evaluations per attempted step after the first, and the pair really is 5th order (measured
 * local-error ratio ~2^6 per halving instead of the assumed 64).
 *
 * What is simplified (an outline, as the task says): no dense output, no interpolation, no
 * stiffness detection and no PI controller memory; hMin simply fails loudly when a tolerance is
 * unreachable. What is verified against real references: exp(t), a stiffly stable scalar problem
 * y' = -1000(y - cos t) - sin t, and a Kepler circular orbit compared with both its analytic
 * solution and an independent fine-step RK4 run. All errors and step statistics are measured.
 */
"use strict";

const { ok, throws } = require("node:assert");

// Stage coefficients: rows 1..6 hold a_ij below the diagonal, row 6 doubles as b5 (FSAL pair).
const DP_A = [
  [],
  [1 / 5],
  [3 / 40, 9 / 40],
  [44 / 45, -56 / 15, 32 / 9],
  [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
  [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
];
const DP_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1];
const DP_B5 = DP_A[6];
const DP_E = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40];

/** Linear combination of stage vectors: scale * sum_s weight_s * vectors[s]. */
const combine = (vectors, weights, scale) =>
  vectors[0].map((_, i) => scale * vectors.reduce((acc, v, s) => acc + weights[s] * v[i], 0));

/** One Dormand-Prince step. Stage 7 is f(t + h, y5) because row 6 of the tableau *is* b5, so it
 * is returned as `k7` for FSAL reuse; k[] therefore holds exactly the seven stages. */
function dpStep(f, t, y, h, firstStage) {
  const k = [firstStage ?? f(t, y)];
  for (let s = 1; s < 7; s++) {
    const stage = y.map((v, i) => v + h * DP_A[s].reduce((acc, a, j) => acc + a * k[j][i], 0));
    k.push(f(t + DP_C[s] * h, stage));
  }
  const y5 = y.map((v, i) => v + h * DP_B5.reduce((acc, b, j) => acc + b * k[j][i], 0));
  return { y5, err: combine(k, DP_E, h), k7: k[6] };
}

/** Scaled RMS norm of the local error estimate; <= 1 means the step is acceptable. */
function scaledErrorNorm(y, y5, err, rtol, atol) {
  const sum = err.reduce((acc, e, i) => acc + (e / (atol + rtol * Math.max(Math.abs(y[i]), Math.abs(y5[i])))) ** 2, 0);
  return Math.sqrt(sum / err.length);
}

/**
 * Adaptive Dormand-Prince integration from t0 to tEnd (forward in time only). Returns the final
 * state and step statistics; throws when the tolerance cannot be met at hMin.
 */
function integrateDP(f, y0, t0, tEnd, { rtol = 1e-8, atol = 1e-10, hInit = 0, hMin = 1e-13, hMax = Infinity, maxSteps = 100000 } = {}) {
  if (!(rtol > 0) || !(atol >= 0)) throw new RangeError("rtol must be > 0 and atol must be >= 0");
  if (tEnd < t0) throw new RangeError("this outline integrates forward in time only");
  if (!(maxSteps > 0)) throw new RangeError("maxSteps must be positive");
  const span = tEnd - t0;
  const stats = { accepted: 0, rejected: 0, evaluations: 1, maxErrNorm: 0, maxH: 0, minH: Infinity };
  let t = t0;
  let y = y0.slice();
  let h = hInit > 0 ? Math.min(hInit, span || hInit) : (span || 1) / 100;
  if (span === 0) return { t, y, stats };
  let k1 = f(t, y);
  while (t < tEnd) {
    if (stats.accepted + stats.rejected > maxSteps) throw new Error(`step limit ${maxSteps} exceeded at t = ${t}`);
    const capped = Math.min(h, tEnd - t);
    const { y5, err, k7 } = dpStep(f, t, y, capped, k1);
    stats.evaluations += 6;
    const errNorm = scaledErrorNorm(y, y5, err, rtol, atol);
    if (errNorm <= 1) {
      stats.accepted += 1;
      stats.maxErrNorm = Math.max(stats.maxErrNorm, errNorm);
      stats.maxH = Math.max(stats.maxH, capped);
      stats.minH = Math.min(stats.minH, capped);
      t += capped;
      y = y5;
      k1 = k7;                                   // FSAL reuse
    } else {
      stats.rejected += 1;
    }
    const factor = Math.min(5, Math.max(0.2, 0.9 * errNorm ** (-1 / 5)));
    if (errNorm > 1 && Math.max(capped * factor, hMin) >= capped) {
      throw new Error(`tolerance not reachable at hMin = ${hMin} (t = ${t}, errNorm = ${errNorm.toFixed(3)})`);
    }
    h = Math.max(hMin, Math.min(hMax, capped * factor));
  }
  return { t, y, stats };
}

/** Independent fixed-step classical RK4, used only as a fine reference solution. */
function rk4Reference(f, y0, t0, tEnd, steps) {
  const h = (tEnd - t0) / steps;
  let y = y0.slice();
  let t = t0;
  for (let n = 0; n < steps; n++) {
    const k1v = f(t, y);
    const k2v = f(t + h / 2, y.map((v, i) => v + (h / 2) * k1v[i]));
    const k3v = f(t + h / 2, y.map((v, i) => v + (h / 2) * k2v[i]));
    const k4v = f(t + h, y.map((v, i) => v + h * k3v[i]));
    y = y.map((v, i) => v + (h / 6) * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]));
    t += h;
  }
  return y;
}

const expF = (_t, y) => [y[0]];
const stiffF = (t, y) => [-1000 * (y[0] - Math.cos(t)) - Math.sin(t)];
const keplerF = (_t, s) => {
  const r3 = Math.hypot(s[0], s[1]) ** 3;
  return [s[2], s[3], -s[0] / r3, -s[1] / r3];
};
const keplerExact = (t) => [Math.cos(t), Math.sin(t), -Math.sin(t), Math.cos(t)];
const energy = (s) => (s[2] * s[2] + s[3] * s[3]) / 2 - 1 / Math.hypot(s[0], s[1]);

function main() {
  const notes = [];

  // 1. Global error against exp(1) at rtol = 1e-8.
  const expRun = integrateDP(expF, [1], 0, 1, { rtol: 1e-8 });
  const expError = Math.abs(expRun.y[0] - Math.E);
  notes.push(`exp(1): error ${expError.toExponential(2)} with ${expRun.stats.accepted} accepted steps (rtol 1e-8)`);
  ok(expError < 1e-7, `rtol = 1e-8 must give roughly 1e-8 accuracy (got ${expError})`);

  // 2. Error-control invariant: every accepted step keeps errNorm <= 1, even after bad guesses.
  const forced = integrateDP(expF, [1], 0, 1, { rtol: 1e-10, hInit: 0.5 });
  notes.push(`forced large hInit: max accepted errNorm ${forced.stats.maxErrNorm.toFixed(6)}, rejections ${forced.stats.rejected}`);
  ok(forced.stats.maxErrNorm <= 1, `accepted steps must satisfy errNorm <= 1 (got ${forced.stats.maxErrNorm})`);
  ok(forced.stats.rejected >= 1, "a too-large initial step must be rejected");
  const budget = 6 * (forced.stats.accepted + forced.stats.rejected) + 1;
  ok(forced.stats.evaluations === budget, `evaluations must be 6 per attempt + 1 (got ${forced.stats.evaluations} vs ${budget})`);

  // 3. Tighter tolerance => smaller error and more steps (the controller must respond).
  const runs = [1e-4, 1e-8, 1e-11].map((rtol) => {
    const run = integrateDP(expF, [1], 0, 1, { rtol, atol: rtol * 1e-3 });
    return { rtol, error: Math.abs(run.y[0] - Math.E), steps: run.stats.accepted };
  });
  notes.push(`tolerance sweep: ${runs.map((r) => `rtol ${r.rtol.toExponential(0)} -> error ${r.error.toExponential(1)} in ${r.steps} steps`).join("; ")}`);
  ok(runs[0].error > runs[1].error && runs[1].error > runs[2].error, "tighter rtol must reduce the global error");
  ok(runs[2].steps > runs[1].steps && runs[1].steps > runs[0].steps, "tighter rtol must increase the step count");
  ok(runs[2].error < 1e-9, `rtol = 1e-11 must reach ~1e-10 accuracy (got ${runs[2].error})`);

  // 4. Measured local order of the 5th-order solution: error ratio 2^6 = 64 per halving.
  const localError = (h) => Math.abs(dpStep(expF, 0, [1], h).y5[0] - Math.exp(h));
  const orderRatio = localError(0.05) / localError(0.025);
  notes.push(`single-step local order ratio ${orderRatio.toFixed(2)} (expected 64)`);
  ok(orderRatio > 55 && orderRatio < 70, `the embedded pair must be 5th order (measured ratio ${orderRatio})`);

  // 5. Kepler circular orbit against the analytic circle and an independent RK4 reference.
  const kepler = integrateDP(keplerF, [1, 0, 0, 1], 0, 2 * Math.PI, { rtol: 1e-10, atol: 1e-12 });
  const exactEnd = keplerExact(2 * Math.PI);
  const worstComponent = Math.max(...kepler.y.map((v, i) => Math.abs(v - exactEnd[i])));
  const drift = Math.abs(energy(kepler.y) - energy([1, 0, 0, 1]));
  const reference = rk4Reference(keplerF, [1, 0, 0, 1], 0, 2 * Math.PI, 200000);
  const referenceGap = Math.max(...kepler.y.map((v, i) => Math.abs(v - reference[i])));
  notes.push(`Kepler 2*pi: max component error ${worstComponent.toExponential(2)}, energy drift ${drift.toExponential(2)}, gap to fine RK4 ${referenceGap.toExponential(2)} in ${kepler.stats.accepted} steps`);
  ok(worstComponent < 1e-7, `the orbit must close to tolerance (got ${worstComponent})`);
  ok(drift < 1e-8, `energy must be conserved to tolerance (got ${drift})`);
  ok(referenceGap < 1e-6, `the adaptive solution must agree with the fine RK4 reference (got ${referenceGap})`);

  // 6. Stiffly stable scalar problem: y' = -1000(y - cos t) - sin t with y(0) = 2 solves to cos t + e^{-1000t}.
  const stiff = integrateDP(stiffF, [2], 0, 1, { rtol: 1e-6, atol: 1e-8, hInit: 0.05 });
  const stiffError = Math.abs(stiff.y[0] - Math.cos(1));
  notes.push(`stiff scalar: error ${stiffError.toExponential(2)}, ${stiff.stats.accepted} accepted / ${stiff.stats.rejected} rejected steps`);
  ok(stiffError < 1e-4, `the stiff problem must be tracked to tolerance (got ${stiffError})`);
  ok(stiff.stats.rejected >= 5, `the stiff transient must force rejections (got ${stiff.stats.rejected})`);

  // 7. An unreachable tolerance must fail loudly instead of silently returning garbage.
  throws(() => integrateDP(stiffF, [2], 0, 1, { rtol: 1e-16, atol: 0, hMin: 1e-2, hInit: 0.05 }), Error,
    "an unreachable tolerance must throw");

  // 8. Boundaries: empty interval, constant derivative and a scalar state.
  const empty = integrateDP(expF, [5], 2, 2, { rtol: 1e-8 });
  ok(empty.t === 2 && empty.y[0] === 5 && empty.stats.accepted === 0, "an empty interval must return the initial state");
  const constant = integrateDP(() => [0], [3], 0, 1, { rtol: 1e-8 });
  ok(constant.y[0] === 3, "a zero derivative must stay exactly constant");
  ok(constant.stats.evaluations === 6 * (constant.stats.accepted + constant.stats.rejected) + 1, "the 6-evaluations-per-attempt budget must hold");
  ok(constant.stats.accepted < 10 && constant.stats.maxH > constant.stats.minH, "with zero local error the controller must grow h aggressively");

  // 9. Determinism: identical runs produce identical states and statistics.
  const again = integrateDP(keplerF, [1, 0, 0, 1], 0, 2 * Math.PI, { rtol: 1e-10, atol: 1e-12 });
  ok(again.y.every((v, i) => v === kepler.y[i]), "repeated runs must be bit-identical");
  ok(again.stats.accepted === kepler.stats.accepted && again.stats.minH === kepler.stats.minH, "step statistics must repeat exactly");

  // 10. Invalid configuration is rejected.
  throws(() => integrateDP(expF, [1], 0, 1, { rtol: 0 }), RangeError, "rtol = 0 must throw");
  throws(() => integrateDP(expF, [1], 1, 0, {}), RangeError, "backward time must throw");
  throws(() => integrateDP(expF, [1], 0, 1, { maxSteps: 0 }), RangeError, "maxSteps = 0 must throw");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { dpStep, scaledErrorNorm, integrateDP, rk4Reference, DP_E, DP_B5 };

if (require.main === module) main();
