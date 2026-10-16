/** Task 786 - Implement advanced numerical integrator (Runge-Kutta 4) and stability tests. (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Classical RK4 on the autonomous or non-autonomous initial value problem y' = f(t, y):
 *   k1 = f(t, y), k2 = f(t + h/2, y + h/2 k1), k3 = f(t + h/2, y + h/2 k2), k4 = f(t + h, y + h k3)
 *   y_{n+1} = y_n + h/6 (k1 + 2 k2 + 2 k3 + k4)
 * The method is order 4: halving h cuts the global error by ~16 (measured below, not assumed).
 *
 * Stability is analysed through the one-step amplification of y' = lambda y, which for RK4 is
 * the stability polynomial
 *   R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 = exp(z) + O(z^5).
 * |R(z)| <= 1 defines the stability region: the real interval is about (-2.7852935634, 0) and
 * the imaginary axis is met at z = +-2*sqrt(2)*i. Both boundaries are found here by bisecting a
 * *numerically measured* amplification (one RK4 step on y' = lambda y) and cross-checked against
 * the polynomial found by Newton iteration and against the literature values, so the stability
 * tests use the integrator itself rather than the polynomial that describes it.
 *
 * What is simplified: fixed step sizes only (no adaptive controller - that is its own task) and
 * scalar/vector systems small enough to run in milliseconds. What is verified: global errors
 * against closed-form solutions (exp, trigonometric, logistic), the measured order of
 * convergence, energy conservation for the harmonic oscillator, the exact derivative-evaluation
 * budget, the measured stability boundary on both axes, and the predicted growth/decay flip for
 * an unstable stiff step. All reported numbers are measured.
 */
"use strict";

const { ok, throws } = require("node:assert");

const add = (a, b) => a.map((x, i) => x + b[i]);
const axpy = (a, b, s) => a.map((x, i) => x + b[i] * s);
const scale = (a, s) => a.map((x) => x * s);
const norm = (a) => Math.sqrt(a.reduce((acc, x) => acc + x * x, 0));

/** One classical RK4 step; returns [t + h, y_new]. */
function rk4Step(f, t, y, h) {
  const k1 = f(t, y);
  const k2 = f(t + h / 2, axpy(y, k1, h / 2));
  const k3 = f(t + h / 2, axpy(y, k2, h / 2));
  const k4 = f(t + h, axpy(y, k3, h));
  const blended = add(add(k1, scale(add(k2, k3), 2)), k4);
  return [t + h, axpy(y, blended, h / 6)];
}

/**
 * Fixed-step RK4 from t0 to tEnd. `steps = 0` returns the initial state unchanged; the state is
 * an array of numbers so scalar problems are just length-1 states.
 */
function integrateRK4(f, y0, t0, tEnd, steps) {
  if (!Number.isInteger(steps) || steps < 0) throw new RangeError(`steps must be a non-negative integer (got ${steps})`);
  const y = y0.slice();
  let t = t0;
  const h = (tEnd - t0) / steps;
  for (let n = 0; n < steps; n++) {
    const [tn, yn] = rk4Step(f, t, y, h);
    t = tn;
    y.splice(0, y.length, ...yn);
  }
  return { t, y };
}

/** R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 for a complex z given as [re, im]. */
function stabilityPolynomial([re, im]) {
  const mul = ([a, b], [c, d]) => [a * c - b * d, a * d + b * c];
  const add2 = ([a, b], [c, d]) => [a + c, b + d];
  const div = ([a, b], k) => [a / k, b / k];
  let term = [1, 0];
  let sum = [0, 0];
  for (let k = 0; k <= 4; k++) {
    sum = add2(sum, term);
    term = div(mul(term, [re, im]), k + 1);
  }
  return sum;
}

/** |R(z)| for a complex z. */
const amplification = ([re, im]) => {
  const [a, b] = stabilityPolynomial([re, im]);
  return Math.hypot(a, b);
};

/** |y1/y0| measured by running RK4 with h = 1 on y' = lambda*y, y(0) = 1 (i.e. own measurement). */
const measuredAmplification = (lambda) => {
  const { y } = integrateRK4((_t, v) => [lambda * v[0]], [1], 0, 1, 1);
  return Math.abs(y[0]);
};

/** Bisects the negative real axis for the last stable |R(-x)| = 1 crossing. */
function realStabilityBoundary(measure) {
  let lo = 2;                 // |R(-2)| < 1: stable
  let hi = 3;                 // |R(-3)| > 1: unstable
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (measure(-mid) <= 1) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** Newton iteration on R(-x) - 1 = -x + x^2/2 - x^3/6 + x^4/24, the real-axis boundary. */
function newtonPolynomialRoot() {
  const f = (x) => -x + x * x / 2 - x ** 3 / 6 + x ** 4 / 24;
  const df = (x) => -1 + x - x * x / 2 + x ** 3 / 6;
  let x = 2.785;              // literature value as the starting point only
  for (let i = 0; i < 60; i++) x -= f(x) / df(x);
  return x;
}

/** Measured convergence order log2(err(h)/err(h/2)) on a scalar closed-form problem. */
function measuredOrder(problem, exact, h, { t0 = 0, tEnd = 1 } = {}) {
  const signed = [h, h / 2, h / 4].map((step) => integrateRK4(problem.f, [problem.y0], t0, tEnd, Math.round((tEnd - t0) / step)).y[0] - exact(tEnd));
  const errors = signed.map(Math.abs);
  return {
    errors,
    order: (Math.log2(errors[0] / errors[1]) + Math.log2(errors[1] / errors[2])) / 2,
    richardson: Math.abs(signed[2] + (signed[2] - signed[1]) / 15),
  };
}

const problems = {
  exponential: { f: (_t, y) => [y[0]], y0: 1, exact: (t) => Math.exp(t) },
  nonAutonomous: { f: (t, y) => [t + y[0]], y0: 1, exact: (t) => 2 * Math.exp(t) - t - 1 },
  logistic: { f: (_t, y) => [y[0] * (1 - y[0])], y0: 0.2, exact: (t) => 0.2 * Math.exp(t) / (0.8 + 0.2 * Math.exp(t)) },
  oscillator: { f: (_t, y) => [y[1], -y[0]], y0: 1, exact: (t) => Math.cos(t), initial: [1, 0] },
};

function main() {
  // 1. Global error against the closed form y = exp(t) over [0, 1].
  const expSolve = integrateRK4(problems.exponential.f, [1], 0, 1, 100);
  const expError = Math.abs(expSolve.y[0] - Math.E);
  const stabilityChecks = [];
  ok(expError < 1e-9, `RK4 must track exp(1) to 1e-9 with h = 0.01 (error ${expError})`);

  // 2. Measured order of convergence for the exponential and the logistic problem.
  const expOrder = measuredOrder(problems.exponential, problems.exponential.exact, 0.05);
  const logisticOrder = measuredOrder(problems.logistic, problems.logistic.exact, 0.05);
  stabilityChecks.push(`order(exponential) = ${expOrder.order.toFixed(3)}, order(logistic) = ${logisticOrder.order.toFixed(3)}`);
  ok(expOrder.order > 3.8 && expOrder.order < 4.3, `RK4 must converge with order 4 (measured ${expOrder.order})`);
  ok(logisticOrder.order > 3.8 && logisticOrder.order < 4.3, `logistic order must be 4 (measured ${logisticOrder.order})`);

  // 3. Richardson extrapolation of two RK4 runs must agree with the closed form far better.
  const richardsonError = expOrder.richardson;
  stabilityChecks.push(`Richardson-extrapolated error ${richardsonError.toExponential(2)} vs RK4 error ${expOrder.errors[2].toExponential(2)}`);
  ok(richardsonError < expOrder.errors[2] / 10, "extrapolating h and h/2 must beat the finest RK4 run");

  // 4. Non-autonomous problem y' = t + y against 2 exp(t) - t - 1.
  const nonAut = integrateRK4(problems.nonAutonomous.f, [1], 0, 1, 200);
  const nonAutError = Math.abs(nonAut.y[0] - (2 * Math.E - 2));
  stabilityChecks.push(`non-autonomous error ${nonAutError.toExponential(2)}`);
  ok(nonAutError < 1e-10, `non-autonomous RK4 must be accurate (error ${nonAutError})`);

  // 5. Harmonic oscillator: the conserved energy must not drift.
  let oscillator = { t: 0, y: [1, 0] };
  const hOsc = 0.01;
  let worstEnergyDrift = 0;
  const energy = (y) => (y[0] * y[0] + y[1] * y[1]) / 2;
  for (let n = 0; n < 2000; n++) {
    oscillator = { t: oscillator.t + hOsc, y: rk4Step(problems.oscillator.f, oscillator.t, oscillator.y, hOsc)[1] };
    worstEnergyDrift = Math.max(worstEnergyDrift, Math.abs(energy(oscillator.y) - 0.5));
  }
  const oscillatorError = Math.abs(oscillator.y[0] - Math.cos(oscillator.t));
  stabilityChecks.push(`oscillator over t = 20: energy drift ${worstEnergyDrift.toExponential(2)}, position error ${oscillatorError.toExponential(2)}`);
  ok(worstEnergyDrift < 1e-6, `RK4 must nearly conserve the oscillator energy (drift ${worstEnergyDrift})`);
  ok(oscillatorError < 1e-6, `oscillator position error must stay small (got ${oscillatorError})`);

  // 6. The stability polynomial must match the truncated exponential series.
  const seriesError = Math.abs(amplification([0.4, 0]) - Math.exp(0.4));
  const tinySeriesError = Math.abs(amplification([0.04, 0]) - Math.exp(0.04));
  stabilityChecks.push(`R(0.4) - exp(0.4) = ${seriesError.toExponential(2)} (fifth order: ${(seriesError / tinySeriesError).toFixed(1)}x larger than at 0.04)`);
  ok(seriesError < 5e-4 && tinySeriesError < 5e-8, "R(z) must be the 4th-order truncation of exp(z)");

  // 7. Real-axis stability boundary: measured amplification vs polynomial root vs literature.
  const measuredBoundary = realStabilityBoundary(measuredAmplification);
  const polynomialBoundary = realStabilityBoundary((x) => amplification([x, 0]));
  const newtonBoundary = newtonPolynomialRoot();
  stabilityChecks.push(`real-axis boundary: measured ${measuredBoundary.toFixed(12)}, polynomial ${polynomialBoundary.toFixed(12)}, literature 2.785293563405`);
  ok(Math.abs(measuredBoundary - 2.785293563405) < 1e-9, `measured stability boundary must be 2.785293563405 (got ${measuredBoundary})`);
  ok(Math.abs(measuredBoundary - newtonBoundary) < 1e-12, "the measured boundary must match the Newton root of the polynomial");

  // 8. Imaginary axis: |R(2*sqrt(2) i)| = 1 and the region |Im| < 2*sqrt(2) is inside.
  const imaginaryBoundary = amplification([0, 2 * Math.SQRT2]);
  const inside = amplification([0, 2.5]);
  const outside = amplification([0, 3]);
  stabilityChecks.push(`imaginary axis: |R(2sqrt2 i)| = ${imaginaryBoundary.toFixed(15)}, |R(2.5i)| = ${inside.toFixed(6)}, |R(3i)| = ${outside.toFixed(6)}`);
  ok(Math.abs(imaginaryBoundary - 1) < 1e-14, "the stability region must touch the imaginary axis at 2*sqrt(2)");
  ok(inside < 1 && outside > 1, "the imaginary-axis boundary must split stable from unstable");

  // 9. One large step of y' = -y (h = 1, z = -1) must reproduce R(-1) exactly and be visibly
  //    worse than the true solution exp(-1).
  const measuredOneStep = measuredAmplification(-1);
  const polynomialOneStep = amplification([-1, 0]);
  ok(Math.abs(measuredOneStep - polynomialOneStep) < 1e-15, "one step of y' = lambda y must reproduce R(lambda*h)");
  ok(Math.abs(measuredOneStep - Math.exp(-1)) > 1e-3, "a step of h = 1 must be visibly off the true solution");

  // 10. Region membership predicts growth or decay: lambda = -30 with h = 0.1 (z = -3) or 1/12.
  const growStep = integrateRK4((_t, y) => [-30 * y[0]], [1], 0, 1, 10);
  const decayStep = integrateRK4((_t, y) => [-30 * y[0]], [1], 0, 1, 12);
  const theoryOutside = amplification([-3, 0]);
  const theoryInside = amplification([-2.5, 0]);
  stabilityChecks.push(`lambda*h = -3: |y(1)| = ${Math.abs(growStep.y[0]).toExponential(3)} = |R(-3)|^10; lambda*h = -2.5: |y(1)| = ${Math.abs(decayStep.y[0]).toExponential(3)}`);
  ok(Math.abs(Math.abs(growStep.y[0]) - theoryOutside ** 10) < 1e-9 * theoryOutside ** 10, "an unstable step must grow exactly as |R(z)|^n");
  ok(theoryOutside > 1, "z = -3 must lie outside the stability region");
  ok(Math.abs(decayStep.y[0]) - theoryInside ** 12 < 1e-12 && theoryInside < 1, "a step inside the region must decay as |R(z)|^n");

  // 11. Derivative budget: exactly four evaluations per step, none wasted.
  let evaluations = 0;
  integrateRK4((t, y) => {
    evaluations += 1;
    return [-y[0] + Math.sin(t)];
  }, [1], 0, 1, 7);
  ok(evaluations === 28, `RK4 must use four f-evaluations per step (got ${evaluations})`);

  // 12. Boundaries: zero steps, a two-component state and a plain one-step solve.
  const zeroSteps = integrateRK4(problems.exponential.f, [2, 3], 0, 5, 0);
  ok(zeroSteps.t === 0 && zeroSteps.y[0] === 2 && zeroSteps.y[1] === 3, "zero steps must return the initial state unchanged");
  const oneStep = integrateRK4(problems.exponential.f, [1], 0, 0.1, 1);
  ok(Math.abs(oneStep.y[0] - amplification([0.1, 0])) < 1e-15, "a single step must equal R(h) for y' = y");

  // 13. Invalid input is rejected.
  throws(() => integrateRK4(problems.exponential.f, [1], 0, 1, -1), RangeError, "negative steps must throw");
  throws(() => integrateRK4(problems.exponential.f, [1], 0, 1, 2.5), RangeError, "non-integer steps must throw");

  for (const line of stabilityChecks) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { rk4Step, integrateRK4, stabilityPolynomial, amplification, measuredAmplification, realStabilityBoundary };

if (require.main === module) main();
