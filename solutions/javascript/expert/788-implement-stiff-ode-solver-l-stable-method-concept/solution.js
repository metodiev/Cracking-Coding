/** Task 788 - Implement stiff ODE solver L-stable method concept. (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * A stiff problem has a fast mode whose *stability*, not accuracy, forces an explicit method to
 * take tiny steps. The cure is an implicit method with an L-stable stability function:
 *   implicit Euler   Y = y + h f(t+h, Y)                 R(z) = 1/(1-z)                       order 1
 *   trapezoidal      Y = y + h/2 [f(t,y) + f(t+h,Y)]     R(z) = (1+z/2)/(1-z/2)              order 2
 *   SDIRK2 (Alexander, gamma = 1 - 1/sqrt(2)), two stages coupled through the same diagonal
 *                    R(z) = (1 + z(1-2*gamma)) / (1 - gamma*z)^2                             order 2
 * Each stage is solved by Newton's method with a numerical Jacobian, so R(z) can be measured by
 * running the method on y' = lambda*y with h = 1.
 *
 * The point of the concept: A-stability (|R(z)| <= 1 for Re z <= 0) is not enough. Trapezoidal is
 * A-stable but R(z) -> -1 as z -> -infinity, so a fast mode is *preserved with alternating sign*
 * and rings; implicit Euler and SDIRK2 are L-stable, R(z) -> 0, so the fast mode dies in one step.
 * Both properties are verified numerically here (measured amplification versus the closed-form
 * R(z), the ringing of trapezoidal versus the one-step decay of the L-stable pair, and the
 * fast-mode kill ratio for h*lambda = -100).
 *
 * What is simplified: the Jacobian is obtained by finite differences rather than analytically and
 * only small stiff systems with a handful of states are used (no sparse assembly, no order/step
 * adaptation). Verified: measured orders 1/2/2 against closed-form solutions, quadratic Newton
 * convergence on log residuals, the exact stability functions, L-stability limits, the stiff-mode
 * kill ratio, and unit-step boundary behaviour as h -> infinity.
 */
"use strict";

const { ok, throws } = require("node:assert");

const GAMMA = 1 - Math.SQRT1_2;      // SDIRK2: gamma = 1 - 1/sqrt(2) is the L-stable choice

const add = (a, b) => a.map((x, i) => x + b[i]);
const axpy = (a, b, s) => a.map((x, i) => x + b[i] * s);
const sub = (a, b) => a.map((x, i) => x - b[i]);
const normInf = (a) => Math.max(...a.map(Math.abs));

/** Numerical Jacobian J[j][i] = d f_i / d y_j (central-free, one-sided for simplicity). */
function jacobian(f, t, y, eps = 1e-7) {
  const base = f(t, y);
  return y.map((value, j) => {
    const step = eps * Math.max(1, Math.abs(value));
    const shifted = y.slice();
    shifted[j] += step;
    return f(t, shifted).map((v, i) => (v - base[i]) / step);
  });
}

/**
 * Solves the stage system Y - rhs - theta*h*f(tStage, Y) = 0 by full Newton iterations with a
 * per-iteration Jacobian (so the convergence is quadratic). Throws if it cannot converge.
 */
function newtonStage(f, tStage, rhs, theta, h, y0, { tol = 1e-13, maxIter = 40 } = {}) {
  let Y = y0.slice();
  const residuals = [];
  for (let iter = 0; iter < maxIter; iter++) {
    const fy = f(tStage, Y);
    const g = Y.map((v, i) => v - rhs[i] - theta * h * fy[i]);
    const residual = normInf(g);
    residuals.push(residual);
    if (residual < tol) return { Y, iterations: iter, residuals };
    const J = jacobian(f, tStage, Y);
    // Solve (I - theta*h*J) delta = g for delta; J[j][i] means row j = state, column i = variable.
    const n = Y.length;
    const m = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0) - theta * h * J[j][i]));
    const delta = solveLinear(m, g);
    Y = sub(Y, delta);   // Newton: solve (I - theta*h*J) delta = g, then Y <- Y - delta
  }
  throw new Error(`Newton did not converge in ${maxIter} iterations (last residual ${residuals.at(-1)})`);
}

/** Dense Gaussian elimination with partial pivoting. */
function solveLinear(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (a[pivot][col] === 0) throw new Error("singular stage matrix");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let r = col + 1; r < n; r++) {
      const factor = a[r][col] / a[col][col];
      for (let c = col; c <= n; c++) a[r][c] -= factor * a[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let sum = a[r][n];
    for (let c = r + 1; c < n; c++) sum -= a[r][c] * x[c];
    x[r] = sum / a[r][r];
  }
  return x;
}

/** One implicit-Euler step: Y = y + h f(t+h, Y). */
const implicitEulerStep = (f, t, y, h) => newtonStage(f, t + h, y, 1, h, y).Y;

/** One trapezoidal step: Y = y + h/2 [f(t, y) + f(t+h, Y)]. */
const trapezoidalStep = (f, t, y, h) => newtonStage(f, t + h, axpy(y, f(t, y), h / 2), 0.5, h, y).Y;

/** One SDIRK2 (Alexander) step: two stages sharing the diagonal coefficient gamma. */
function sdirktStep(f, t, y, h) {
  const Y1 = newtonStage(f, t + GAMMA * h, y, GAMMA, h, y).Y;
  const f1 = f(t + GAMMA * h, Y1);
  const rhs = axpy(y, f1, (1 - GAMMA) * h);
  const Y2 = newtonStage(f, t + h, rhs, GAMMA, h, Y1).Y;
  const f2 = f(t + h, Y2);
  return y.map((v, i) => v + h * ((1 - GAMMA) * f1[i] + GAMMA * f2[i]));
}

/** Fixed-step integration with a given one-step method. */
function integrate(step, f, y0, t0, tEnd, steps) {
  const h = (tEnd - t0) / steps;
  let y = y0.slice();
  let t = t0;
  for (let n = 0; n < steps; n++) {
    y = step(f, t, y, h);
    t += h;
  }
  return y;
}

/** Closed-form stability functions. */
const stability = {
  implicitEuler: (z) => Math.hypot(1, 0) / Math.hypot(1 - z[0], -z[1]),
  trapezoidal: ([re, im]) => Math.hypot(1 + re / 2, im / 2) / Math.hypot(1 - re / 2, -im / 2),
  sdirkt2: ([re, im]) => {
    const num = Math.hypot(1 + re * (1 - 2 * GAMMA), im * (1 - 2 * GAMMA));
    const den = Math.hypot(1 - GAMMA * re, -GAMMA * im) ** 2;
    return num / den;
  },
};

/** Measured amplification |y(1)/y(0)| of one method applied to y' = lambda y with h = 1. */
const measuredAmplification = (step, lambda) => Math.abs(integrate(step, (_t, y) => [lambda * y[0]], [1], 0, 1, 1)[0]);

/** Measured order of a method on y' = -y over [0, 1] (exact y = exp(-1)). */
function measuredOrder(step) {
  const errors = [0.1, 0.05, 0.025].map((h) => Math.abs(integrate(step, (_t, y) => [-y[0]], [1], 0, 1, Math.round(1 / h))[0] - Math.exp(-1)));
  return (Math.log2(errors[0] / errors[1]) + Math.log2(errors[1] / errors[2])) / 2;
}

const stiffF = (t, y) => [-1000 * (y[0] - Math.cos(t)) - Math.sin(t)];

function main() {
  const notes = [];

  // 1. Measured amplification must equal the closed-form stability function for all three methods.
  const lambdas = [-0.5, -3, -1e3, -1e6];
  const worstDrift = ["implicitEuler", "trapezoidal", "sdirkt2"].reduce((acc, name) => {
    const step = { implicitEuler: implicitEulerStep, trapezoidal: trapezoidalStep, sdirkt2: sdirktStep }[name];
    return Math.max(...lambdas.map((lam) => Math.abs(measuredAmplification(step, lam) - stability[name]([lam, 0]))));
  }, 0);
  notes.push(`measured vs closed-form R(z): worst deviation ${worstDrift.toExponential(2)}`);
  ok(worstDrift < 1e-6, `the measured amplification must match R(z) (worst deviation ${worstDrift})`);

  // 2. L-stability: R(z) -> 0 for both L-stable methods, while trapezoidal stalls at -1.
  const eulerFar = stability.implicitEuler([-1e9, 0]);
  const sdirktFar = stability.sdirkt2([-1e9, 0]);
  const trapFar = stability.trapezoidal([-1e9, 0]);
  notes.push(`R(-1e9): implicit Euler ${eulerFar.toExponential(2)}, SDIRK2 ${sdirktFar.toExponential(2)}, trapezoidal ${trapFar.toFixed(6)}`);
  ok(eulerFar < 1e-8 && sdirktFar < 1e-8, "L-stable methods must vanish at -infinity");
  ok(Math.abs(trapFar - 1) < 1e-8, "trapezoidal must keep |R| = 1 in the far left half-plane (not L-stable)");

  // 3. A-stability on the left half-plane: |R(z)| <= 1 for all three, checked on a grid.
  const worstOutside = [];
  for (const name of ["implicitEuler", "trapezoidal", "sdirkt2"]) {
    let worst = 0;
    for (let re = -20; re <= 0; re += 0.5) for (let im = -20; im <= 20; im += 0.5) worst = Math.max(worst, stability[name]([re, im]));
    worstOutside.push({ name, worst });
  }
  notes.push(`A-stability grid: ${worstOutside.map((w) => `${w.name} max |R| = ${w.worst.toFixed(6)}`).join(", ")}`);
  ok(worstOutside.every((w) => w.worst <= 1 + 1e-12), "all three methods must be A-stable on the grid");

  // 4. Fast-mode kill ratio for h*lambda = -100: one step kills or preserves the fast mode.
  const kill = { implicitEuler: measuredAmplification(implicitEulerStep, -100), trapezoidal: measuredAmplification(trapezoidalStep, -100), sdirkt2: measuredAmplification(sdirktStep, -100) };
  notes.push(`fast mode h*lambda = -100: |implicit Euler| = ${kill.implicitEuler.toExponential(2)}, |SDIRK2| = ${kill.sdirkt2.toExponential(2)}, |trapezoidal| = ${kill.trapezoidal.toFixed(6)}`);
  ok(kill.implicitEuler < 0.02 && kill.sdirkt2 < 0.1, "L-stable methods must kill the fast mode in one step");
  ok(kill.trapezoidal > 0.9, "trapezoidal must preserve the fast mode with alternating sign");

  // 5. Measured orders: implicit Euler 1, trapezoidal 2, SDIRK2 2.
  const orders = { implicitEuler: measuredOrder(implicitEulerStep), trapezoidal: measuredOrder(trapezoidalStep), sdirkt2: measuredOrder(sdirktStep) };
  notes.push(`measured orders: implicit Euler ${orders.implicitEuler.toFixed(3)}, trapezoidal ${orders.trapezoidal.toFixed(3)}, SDIRK2 ${orders.sdirkt2.toFixed(3)}`);
  ok(Math.abs(orders.implicitEuler - 1) < 0.1, `implicit Euler must be order 1 (measured ${orders.implicitEuler})`);
  ok(Math.abs(orders.trapezoidal - 2) < 0.15 && Math.abs(orders.sdirkt2 - 2) < 0.15, "trapezoidal and SDIRK2 must be order 2");

  // 6. Stiff problem y' = -1000(y - cos t) - sin t (exact cos t + e^{-1000t}): L-stable methods cope,
  //    trapezoidal rings and explicit RK4 with the same step diverges.
  const h = 0.1;
  const implicitDeviation = [];
  const trapezoidDeviation = [];
  let yIE = [2];
  let yTR = [2];
  for (let n = 0; n < 10; n++) {
    yIE = implicitEulerStep(stiffF, n * h, yIE, h);
    yTR = trapezoidalStep(stiffF, n * h, yTR, h);
    implicitDeviation.push(yIE[0] - Math.cos((n + 1) * h));
    trapezoidDeviation.push(yTR[0] - Math.cos((n + 1) * h));
  }
  const rk4StepExplicit = (f, t, y, step) => {
    const k1 = f(t, y);
    const k2 = f(t + step / 2, axpy(y, k1, step / 2));
    const k3 = f(t + step / 2, axpy(y, k2, step / 2));
    const k4 = f(t + step, axpy(y, k3, step));
    return y.map((v, i) => v + (step / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  };
  const explicitBlowUp = integrate(rk4StepExplicit, stiffF, [2], 0, 1, 10);
  const ieError = normInf(sub(yIE, [Math.cos(1)]));
  const trError = normInf(sub(yTR, [Math.cos(1)]));
  const flips = trapezoidDeviation.filter((d, n) => n > 0 && d * trapezoidDeviation[n - 1] < 0).length;
  notes.push(`stiff at h = 0.1: implicit Euler error ${ieError.toExponential(2)}, trapezoidal error ${trError.toExponential(2)} (${flips} sign flips = ringing), explicit RK4 |y(1)| = ${Math.abs(explicitBlowUp[0]).toExponential(2)}`);
  ok(Math.abs(explicitBlowUp[0]) > 1, "explicit RK4 with h = 0.1 must be unstable on this stiff problem");
  ok(ieError < 1e-2, `implicit Euler must stay accurate on the stiff problem (error ${ieError})`);
  ok(Math.max(...trapezoidDeviation.map(Math.abs)) > Math.max(...implicitDeviation.map(Math.abs)), "trapezoidal must ring more than the L-stable method");
  ok(flips >= 2, `the trapezoidal deviation must oscillate (got ${flips} sign flips)`);

  // 7. Quadratic Newton convergence on a nonlinear stage: Y = 0.2 + 0.5 Y^2.
  const nonlinearStage = newtonStage((_t, y) => [y[0] * y[0]], 0.5, [0.2], 1, 0.5, [0.2]);
  const ratios = nonlinearStage.residuals
    .map((residual, i, all) => (i === 0 ? NaN : Math.log(residual) / Math.log(all[i - 1])))
    .filter((r) => Number.isFinite(r) && r > 0);
  notes.push(`Newton on y' = y^2, h = 0.5: ${nonlinearStage.iterations} iterations, residuals ${nonlinearStage.residuals.map((r) => r.toExponential(1)).join(" ")}`);
  ok(nonlinearStage.iterations <= 8, `Newton must converge quickly (took ${nonlinearStage.iterations} iterations)`);
  ok(ratios.some((r) => r > 1.8), `log-residual ratios must show quadratic convergence (got ${ratios.map((r) => r.toFixed(2)).join(", ")})`);

  // 8. Boundary: h -> infinity in the L-stable methods still cannot overflow (they are rational).
  const huge = implicitEulerStep((_t, y) => [-y[0]], 0, [1], 1e9);
  const hugeSDIRK = sdirktStep((_t, y) => [-y[0]], 0, [1], 1e9);
  notes.push(`h = 1e9 with lambda = -1: implicit Euler ${huge[0].toExponential(2)}, SDIRK2 ${hugeSDIRK[0].toExponential(2)}`);
  ok(Math.abs(huge[0]) < 1e-6 && Math.abs(hugeSDIRK[0]) < 1e-6, "an enormous step must still decay to zero");
  const zero = implicitEulerStep(() => [0], 0, [7], 0.5);
  ok(zero[0] === 7, "a zero derivative must stay constant");

  // 9. A 2-state stiff system exercises the stage matrix and the numerical Jacobian:
  //    eigenvalues -1 (eigenvector [1,1]) and -999 (eigenvector [1,-1]).
  const stiffSystem = (_t, y) => [-500 * y[0] + 499 * y[1], 499 * y[0] - 500 * y[1]];
  const exactSystem = (t) => [0.5 * Math.exp(-t) + 0.5 * Math.exp(-999 * t), 0.5 * Math.exp(-t) - 0.5 * Math.exp(-999 * t)];
  const system = integrate(implicitEulerStep, stiffSystem, [1, 0], 0, 1, 20);
  notes.push(`2-state stiff system after 20 steps: [${system.map((v) => v.toExponential(3)).join(", ")}] vs exact [${exactSystem(1).map((v) => v.toExponential(3)).join(", ")}]`);
  ok(normInf(sub(system, exactSystem(1))) < 1e-2, "the stiff 2-state system must track its closed-form solution");

  // 10. Non-convergence must be reported instead of returning a silently wrong stage.
  throws(() => newtonStage((_t, y) => [y[0] * y[0]], 1, [0], 1, 1e6, [1], { maxIter: 3 }), Error, "an exploding stage must throw");

  // 11. Invalid input.
  throws(() => solveLinear([[0, 0], [0, 0]], [1, 1]), Error, "a singular stage matrix must throw");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { jacobian, newtonStage, solveLinear, implicitEulerStep, trapezoidalStep, sdirktStep, integrate, stability };

if (require.main === module) main();
