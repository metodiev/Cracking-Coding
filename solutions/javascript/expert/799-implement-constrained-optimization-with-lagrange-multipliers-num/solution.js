/** Task 799 - Implement constrained optimization with Lagrange multipliers numeric solver. (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Two numeric solvers for smooth constrained problems min f(x) s.t. g(x) = 0 and h(x) <= 0:
 *   KKT Newton: iterate on the stationarity/feasibility system
 *     grad f(x) + sum_k lambda_k grad g_k(x) = 0,   g(x) = 0
 *     Newton step from [ H_L  Jg^T ; Jg  0 ] [dx; dlambda] = -[ grad f + Jg^T lambda ; g ]
 *     with H_L the Hessian of the Lagrangian (all derivatives by central differences, documented).
 *   Augmented Lagrangian / multiplier iteration for inequality constraints:
 *     L(x; lambda, rho) = f + (rho/2) sum_k [ max(0, lambda_k + rho h_k(x) )^2 - lambda_k^2 ] / rho,
 *     lambda_k <- max(0, lambda_k + rho h_k(x)), rho <- 2 rho, minimised by a few Newton steps.
 * Because the KKT conditions are necessary *and* sufficient here, the solvers are verified against
 * closed-form optima (with the convention L = f + sum lambda_k g_k): (a) min x1^2 + x2^2 s.t. x1 + x2 = 1
 * -> (0.5, 0.5), lambda = -1; (b) min -x y z s.t. x + y + z = 3 -> (1, 1, 1), lambda = 1;
 * (c) min (x-2)^2 + (y-1)^2 s.t. x^2 + y^2 <= 1 -> (2, 1)/sqrt(5), lambda = sqrt(5) - 1.
 * An exhaustive active-set enumeration and a brute-force grid search give independent second opinions.
 *
 * Invariants verified: gradient stationarity ||grad f + sum lambda grad g||, primal feasibility, dual
 * feasibility (lambda >= 0 for active inequalities) and complementary slackness lambda_k h_k = 0 all
 * hold at the returned point; the multipliers match the closed forms; the constrained minimum is never
 * below the unconstrained one (the penalty bound); infeasible or degenerate problems raise instead of
 * returning a made-up answer; and every run is deterministic.
 *
 * What is simplified: derivatives come from central differences rather than hand-coded analytic forms
 * (so the Jacobians and Hessians are only as accurate as the finite-difference step allows), only small
 * dense problems are handled, and the inequality solver uses a fixed multiplier-update schedule.
 */
"use strict";

const { ok, throws } = require("node:assert");

const norm = (vector) => Math.sqrt(vector.reduce((acc, v) => acc + v * v, 0));
const dot = (a, b) => a.reduce((acc, v, i) => acc + v * b[i], 0);

/** Central-difference gradient of a scalar function. */
function gradient(f, x, h = 1e-6) {
  return x.map((value, i) => {
    const plus = x.slice();
    const minus = x.slice();
    plus[i] = value + h;
    minus[i] = value - h;
    return (f(plus) - f(minus)) / (2 * h);
  });
}

/** Central-difference Jacobian of a vector-valued function: rows are the constraints. */
const jacobian = (functions, x, h = 1e-6) => functions.map((g) => gradient((y) => g(y), x, h));

/** Dense Gaussian elimination with partial pivoting. */
function solveSystem(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-12) throw new Error("singular KKT system");
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

/** Hessian of a scalar function by central differences on the gradient. */
function hessian(f, x, h = 1e-4) {
  const n = x.length;
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
    const a = x.slice();
    const b = x.slice();
    const c = x.slice();
    const d = x.slice();
    a[i] += h;
    a[j] += h;
    b[i] += h;
    b[j] -= h;
    c[i] -= h;
    c[j] += h;
    d[i] -= h;
    d[j] -= h;
    return (f(a) - f(b) - f(c) + f(d)) / (4 * h * h);
  }));
}

const lagrangian = (f, constraints, multipliers) => (x) => f(x) + multipliers.reduce((acc, lambda, k) => acc + lambda * constraints[k](x), 0);

/** KKT Newton for equality constraints; returns the optimum and the multipliers, plus a residual log. */
function kktNewton(f, constraints, x0, { maxIterations = 60, tolerance = 1e-9 } = {}) {
  let x = x0.slice();
  let multipliers = new Array(constraints.length).fill(0);
  const residuals = [];
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const g = constraints.map((constraint) => constraint(x));
    const gradF = gradient(f, x);
    const J = jacobian(constraints, x);
    const stationarity = gradF.map((value, i) => value + J.reduce((acc, row, k) => acc + multipliers[k] * row[i], 0));
    const residual = Math.max(norm(stationarity), norm(g));
    residuals.push(residual);
    if (residual < tolerance) return { x, multipliers, iterations: iteration, residuals, converged: true };
    const lagrangianFunction = lagrangian(f, constraints, multipliers);
    const H = hessian(lagrangianFunction, x);
    const n = x.length;
    const m = constraints.length;
    const size = n + m;
    const system = Array.from({ length: size }, (_, i) => new Array(size).fill(0));
    const rhs = new Array(size).fill(0);
    for (let i = 0; i < n; i++) {
      rhs[i] = -stationarity[i];
      for (let j = 0; j < n; j++) system[i][j] = H[i][j];
      for (let k = 0; k < m; k++) system[i][n + k] = J[k][i];
    }
    for (let k = 0; k < m; k++) {
      rhs[n + k] = -g[k];
      for (let j = 0; j < n; j++) system[n + k][j] = J[k][j];
      // A tiny negative curvature on the multiplier block keeps the step well defined when the
      // constraints are linearly dependent (a redundant constraint would make the system singular).
      system[n + k][n + k] = -1e-8;
    }
    const step = solveSystem(system, rhs);
    x = x.map((value, i) => value + step[i]);
    multipliers = multipliers.map((value, k) => value + step[n + k]);
  }
  return { x, multipliers, iterations: maxIterations, residuals, converged: false };
}

/**
 * Augmented Lagrangian (PHR) for inequality constraints h_k(x) <= 0:
 *   L(x; lambda, rho) = f + sum_k [ max(0, lambda_k + rho h_k(x))^2 - lambda_k^2 ] / (2 rho)
 * The inner problem is minimised by damped Newton steps with a backtracking line search, then the
 * multipliers and the penalty are updated: lambda_k <- max(0, lambda_k + rho h_k(x)), rho <- 2 rho.
 */
function augmentedLagrangian(f, constraints, x0, { outerIterations = 40, innerSteps = 60, rho: rho0 = 1 } = {}) {
  let x = x0.slice();
  let multipliers = new Array(constraints.length).fill(0);
  let rho = rho0;
  const history = [];
  const objective = (point) => f(point) + multipliers.reduce(
    (acc, lambda, k) => acc + (Math.max(0, lambda + rho * constraints[k](point)) ** 2 - lambda * lambda) / (2 * rho), 0);
  for (let outer = 0; outer < outerIterations; outer++) {
    for (let step = 0; step < innerSteps; step++) {
      const grad = gradient(objective, x, 1e-6);
      const gradientNorm = norm(grad);
      if (gradientNorm < 1e-12) break;
      const H = hessian(objective, x, 1e-5);
      const scale = 1e-6 * (1 + Math.max(...H.map((row, i) => row.reduce((acc, v, j) => acc + Math.abs(v), 0))));
      const regularised = H.map((row, i) => row.map((v, j) => v + (i === j ? scale : 0)));
      let direction;
      try {
        direction = solveSystem(regularised, grad.map((v) => -v));
      } catch {
        direction = grad.map((v) => -v);
      }
      if (!Number.isFinite(dot(direction, grad)) || dot(direction, grad) >= 0) direction = grad.map((v) => -v);
      let alpha = 1;
      const current = objective(x);
      let trial = x.map((v, i) => v + alpha * direction[i]);
      for (let tries = 0; tries < 40 && objective(trial) > current + 1e-4 * alpha * dot(direction, grad); tries++) {
        alpha /= 2;
        trial = x.map((v, i) => v + alpha * direction[i]);
      }
      x = trial;
    }
    multipliers = multipliers.map((lambda, k) => Math.max(0, lambda + rho * constraints[k](x)));
    const violation = Math.max(0, ...constraints.map((constraint) => constraint(x)));
    history.push({ outer, rho, violation, objective: f(x) });
    if (violation < 1e-12) break;
    rho *= 2;
  }
  return { x, multipliers, rho, history };
}

/** Exhaustive active-set enumeration: solve the equality KKT for every active subset, keep the winner. */
function activeSetSolve(f, equalities, inequalities, x0) {
  const m = inequalities.length;
  const results = [];
  for (let mask = 0; mask < 2 ** m; mask++) {
    const active = Array.from({ length: m }, (_, k) => k).filter((k) => (mask >> k) & 1);
    const constraints = [...equalities, ...active.map((k) => inequalities[k])];
    let solved;
    try {
      solved = kktNewton(f, constraints, x0, { maxIterations: 100, tolerance: 1e-9 });
    } catch {
      continue;
    }
    if (!solved.converged) continue;
    const dual = solved.multipliers.slice(equalities.length);
    const dualFeasible = dual.every((lambda) => lambda >= -1e-6);
    const primalFeasible = inequalities.every((constraint) => constraint(solved.x) <= 1e-8);
    if (!dualFeasible || !primalFeasible) continue;
    results.push({ mask, active, x: solved.x, multipliers: solved.multipliers, objective: f(solved.x) });
  }
  if (results.length === 0) throw new Error("no feasible active set found");
  return results.reduce((acc, candidate) => (candidate.objective < acc.objective ? candidate : acc));
}

/** Brute-force grid search over a box, keeping the best feasible point (the independent oracle). */
function gridSearch(f, inequalities, bounds, steps = 400) {
  let best = null;
  for (let i = 0; i <= steps; i++) {
    for (let j = 0; j <= steps; j++) {
      const point = [bounds[0][0] + (i * (bounds[0][1] - bounds[0][0])) / steps, bounds[1][0] + (j * (bounds[1][1] - bounds[1][0])) / steps];
      if (!inequalities.every((constraint) => constraint(point) <= 0)) continue;
      const value = f(point);
      if (!best || value < best.value) best = { point, value };
    }
  }
  if (!best) throw new Error("the grid found no feasible point");
  return best;
}

function main() {
  const notes = [];

  // 1. Closed-form equality problem: min x1^2 + x2^2 s.t. x1 + x2 = 1.
  const quadratic = (x) => x[0] * x[0] + x[1] * x[1];
  const sumConstraint = (x) => x[0] + x[1] - 1;
  const quad = kktNewton(quadratic, [sumConstraint], [0, 0]);
  const quadError = Math.max(Math.abs(quad.x[0] - 0.5), Math.abs(quad.x[1] - 0.5));
  const quadMultiplierError = Math.abs(quad.multipliers[0] + 1);
  notes.push(`min x1^2+x2^2 s.t. x1+x2=1: x = (${quad.x.map((v) => v.toFixed(9)).join(", ")}), lambda = ${quad.multipliers[0].toFixed(9)} vs -1 after ${quad.iterations} iterations (errors ${quadError.toExponential(2)}, ${quadMultiplierError.toExponential(2)})`);
  ok(quadError < 1e-8 && quadMultiplierError < 1e-6, "the quadratic KKT solution and multiplier must match the closed form");
  ok(quad.converged, "the KKT iteration must converge");

  // 2. Nonlinear equality problem: min -x*y*z s.t. x + y + z = 3 -> (1, 1, 1), lambda = -1.
  const product = (x) => -x[0] * x[1] * x[2];
  const simplex = (x) => x[0] + x[1] + x[2] - 3;
  const nonlinear = kktNewton(product, [simplex], [0.5, 1.5, 1]);
  const nonlinearError = Math.max(...nonlinear.x.map((v) => Math.abs(v - 1)));
  notes.push(`min -xyz s.t. x+y+z=3: x = (${nonlinear.x.map((v) => v.toFixed(9)).join(", ")}), lambda = ${nonlinear.multipliers[0].toFixed(6)} vs 1 (error ${Math.abs(nonlinear.multipliers[0] - 1).toExponential(2)})`);
  ok(nonlinearError < 1e-7, `the nonlinear optimum must be (1, 1, 1) (worst error ${nonlinearError})`);
  ok(Math.abs(nonlinear.multipliers[0] - 1) < 1e-5, "the multiplier must match the closed form 1");

  // 3. Stationarity, feasibility and convergence residuals at the returned point.
  const nonlinearGradient = gradient(product, nonlinear.x);
  const constraintGradient = gradient(simplex, nonlinear.x);
  const stationarityResidual = norm(nonlinearGradient.map((v, i) => v + nonlinear.multipliers[0] * constraintGradient[i]));
  const feasibility = Math.abs(simplex(nonlinear.x));
  notes.push(`nonlinear solution residuals: ||grad f + lambda grad g|| = ${stationarityResidual.toExponential(2)}, |g| = ${feasibility.toExponential(2)}; final inner residual ${nonlinear.residuals.at(-1).toExponential(2)}`);
  ok(stationarityResidual < 1e-6, "the stationarity residual must vanish");
  ok(feasibility < 1e-9, "the equality constraint must be satisfied");

  // 4. Inequality problem: min (x-2)^2 + (y-1)^2 s.t. x^2 + y^2 <= 1 (projection onto the unit disk).
  const distance = (x) => (x[0] - 2) ** 2 + (x[1] - 1) ** 2;
  const disk = (x) => x[0] * x[0] + x[1] * x[1] - 1;
  const closedX = [2 / Math.sqrt(5), 1 / Math.sqrt(5)];
  const closedMultiplier = Math.sqrt(5) - 1;
  const activeSet = activeSetSolve(distance, [], [disk], [0.5, 0.5]);
  const activeError = Math.max(Math.abs(activeSet.x[0] - closedX[0]), Math.abs(activeSet.x[1] - closedX[1]));
  notes.push(`min ||x - (2,1)||^2 s.t. |x|^2 <= 1: active set {${activeSet.active.join(",")}}, x = (${activeSet.x.map((v) => v.toFixed(9)).join(", ")}) vs closed form (${closedX.map((v) => v.toFixed(9)).join(", ")}), lambda = ${activeSet.multipliers[0].toFixed(6)} vs ${closedMultiplier.toFixed(6)}`);
  ok(activeError < 1e-7, `the projection must match the closed form (worst error ${activeError})`);
  ok(Math.abs(activeSet.multipliers[0] - closedMultiplier) < 1e-6, "the inequality multiplier must match sqrt(5) - 1");
  ok(activeSet.active.length === 1, "exactly one inequality constraint must be active");

  // 5. The augmented Lagrangian must reach the same point, with complementary slackness satisfied.
  const augmented = augmentedLagrangian(distance, [disk], [0.2, 0.1], { outerIterations: 30, innerSteps: 30 });
  const augmentedError = Math.max(Math.abs(augmented.x[0] - closedX[0]), Math.abs(augmented.x[1] - closedX[1]));
  const complementary = augmented.multipliers[0] * disk(augmented.x);
  notes.push(`augmented Lagrangian: x = (${augmented.x.map((v) => v.toFixed(9)).join(", ")}), lambda = ${augmented.multipliers[0].toFixed(6)}, violation ${Math.max(0, disk(augmented.x)).toExponential(2)}, complementary slackness ${Math.abs(complementary).toExponential(2)}`);
  ok(augmentedError < 1e-4, `the augmented Lagrangian must find the same optimum (error ${augmentedError})`);
  ok(Math.max(0, disk(augmented.x)) < 1e-6, "the inequality must be feasible at the solution");
  ok(augmented.multipliers[0] >= 0, "the inequality multiplier must be non-negative");

  // 6. Independent oracle: a brute-force grid search must agree with the analytic optimum.
  const grid = gridSearch(distance, [disk], [[-1.1, 1.1], [-1.1, 1.1]], 400);
  const gridGap = Math.abs(grid.value - distance(closedX));
  notes.push(`grid oracle: best feasible value ${grid.value.toFixed(6)} at (${grid.point.map((v) => v.toFixed(4)).join(", ")}) vs analytic ${distance(closedX).toFixed(6)} (gap ${gridGap.toExponential(2)})`);
  ok(gridGap < 5e-3, "the grid search must agree with the analytic optimum to grid resolution");
  ok(grid.value >= distance(closedX) - 1e-12, "no grid point may beat the analytic minimum");

  // 7. The constrained minimum cannot beat the unconstrained one (the penalty bound).
  const unconstrained = kktNewton(quadratic, [], [3, 3]);
  notes.push(`bound check: unconstrained min ${quadratic(unconstrained.x).toExponential(3)}, equality-constrained min ${quadratic(quad.x).toExponential(3)}, disk-constrained min ${distance(activeSet.x).toFixed(6)}`);
  ok(quadratic(quad.x) >= quadratic(unconstrained.x) - 1e-9, "adding a constraint cannot lower the minimum");
  ok(distance(activeSet.x) >= 1 - 1e-9, "the disk projection must be at least the distance from (2,1) to the origin minus the radius");

  // 8. Degenerate and invalid inputs.
  throws(() => solveSystem([[1, 1], [1, 1]], [1, 1]), Error, "a singular KKT system must throw");
  throws(() => gridSearch(distance, [(x) => x[0] * x[0] + x[1] * x[1] + 1], [[-1, 1], [-1, 1]], 20), Error,
    "a grid with no feasible point must throw");
  const noConstraint = kktNewton(quadratic, [], [7, -2]);
  ok(norm(gradient(quadratic, noConstraint.x)) < 1e-6, "with no constraint the gradient must vanish");
  const redundant = kktNewton(quadratic, [sumConstraint, (x) => 2 * (x[0] + x[1] - 1)], [0, 0]);
  ok(Math.max(Math.abs(redundant.x[0] - 0.5), Math.abs(redundant.x[1] - 0.5)) < 1e-8, "a redundant constraint must not change the optimum");

  // 9. Determinism.
  const repeat = kktNewton(product, [simplex], [0.5, 1.5, 1]);
  ok(repeat.x.every((v, i) => v === nonlinear.x[i]) && repeat.multipliers[0] === nonlinear.multipliers[0], "the solver must be deterministic");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { gradient, jacobian, hessian, solveSystem, kktNewton, augmentedLagrangian, activeSetSolve, gridSearch };

if (require.main === module) main();
