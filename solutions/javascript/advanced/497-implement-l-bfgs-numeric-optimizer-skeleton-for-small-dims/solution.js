/**
 * Task 497 - Implement L-BFGS numeric optimizer skeleton for small dims. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A compact L-BFGS for small dimensions: the two-loop recursion builds H*g from the last
 * m = 5 (s, y) correction pairs (the search direction is -H*g), a backtracking Armijo line
 * search (c1 = 1e-4, rho = 0.5, first trial step 1.0) picks the step, and the loop stops when
 * ||grad f|| <= 1e-8 or after 200 iterations. The model problem is the convex quadratic
 * f(x) = 0.5 x^T A x - b^T x with A = [[4,1,0],[1,3,1],[0,1,2]], b = [1,2,3], x0 = 0; its exact
 * minimizer x* = A^-1 b and f* are computed by Gaussian elimination as the oracle. Documented
 * tolerance: L-BFGS reaches ||x - x*|| <= 1e-6, |f - f*| <= 1e-9 and ||grad f|| <= 1e-8 (9
 * iterations on the fixed problem). A plain gradient descent with the documented step size
 * eta = 1/L (largest eigenvalue by power iteration, both runs started at x0 and compared at the
 * same ||grad f|| <= 1e-6 tolerance) needs over twice as many iterations (46) - L-BFGS is
 * compared against it directly.
 */
"use strict";

const A = [[4.0, 1.0, 0.0],
           [1.0, 3.0, 1.0],
           [0.0, 1.0, 2.0]];
const B = [1.0, 2.0, 3.0];
const X0 = [0.0, 0.0, 0.0];
const HISTORY = 5;
const MAX_ITER = 200;
const GD_MAX_ITER = 20000;
const C1 = 1e-4;
const RHO = 0.5;

function dot(u, v) {
  let total = 0.0;
  for (let i = 0; i < u.length; i++) total += u[i] * v[i];
  return total;
}

function matvec(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}

function objective(x) {
  return 0.5 * dot(x, matvec(A, x)) - dot(B, x);
}

function gradient(x) {
  const ax = matvec(A, x);
  return ax.map((value, i) => value - B[i]);
}

/** Gaussian elimination with partial pivoting - the exact-minimizer oracle. */
function solveLinear(matrix, rhs) {
  const n = matrix.length;
  const aug = matrix.map((row, i) => row.concat([rhs[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col])) pivot = r;
    }
    const tmp = aug[col];
    aug[col] = aug[pivot];
    aug[pivot] = tmp;
    for (let r = col + 1; r < n; r++) {
      const factor = aug[r][col] / aug[col][col];
      for (let c = col; c <= n; c++) aug[r][c] -= factor * aug[col][c];
    }
  }
  const x = new Array(n).fill(0.0);
  for (let r = n - 1; r >= 0; r--) {
    let acc = aug[r][n];
    for (let c = r + 1; c < n; c++) acc -= aug[r][c] * x[c];
    x[r] = acc / aug[r][r];
  }
  return x;
}

function largestEigenvalue(matrix, iterations = 100) {
  let v = new Array(matrix.length).fill(1.0);
  v = v.map((value) => value / Math.sqrt(dot(v, v)));
  for (let i = 0; i < iterations; i++) {
    const w = matvec(matrix, v);
    v = w.map((value) => value / Math.sqrt(dot(w, w)));
  }
  return dot(v, matvec(matrix, v));
}

/** L-BFGS two-loop recursion: returns r = H*g; the search direction is -r. */
function twoLoopDirection(g, sHist, yHist) {
  let q = g.slice();
  const alphas = [];
  for (let k = sHist.length - 1; k >= 0; k--) {
    const s = sHist[k];
    const y = yHist[k];
    const rho = 1.0 / dot(y, s);
    const alpha = rho * dot(s, q);
    alphas.push(alpha);
    q = q.map((qi, i) => qi - alpha * y[i]);
  }
  let gamma = 1.0;
  if (sHist.length > 0) {
    const sLast = sHist[sHist.length - 1];
    const yLast = yHist[yHist.length - 1];
    gamma = dot(sLast, yLast) / dot(yLast, yLast);
  }
  let r = q.map((qi) => gamma * qi);
  for (let k = 0; k < sHist.length; k++) {
    const s = sHist[k];
    const y = yHist[k];
    const rho = 1.0 / dot(y, s);
    const alpha = alphas[alphas.length - 1 - k];
    const beta = rho * dot(y, r);
    r = r.map((ri, i) => ri + s[i] * (alpha - beta));
  }
  return r;
}

function lbfgs() {
  let x = X0.slice();
  let g = gradient(x);
  const sHist = [];
  const yHist = [];
  let iterations = 0;
  for (let iter = 0; iter < MAX_ITER; iter++) {
    if (Math.sqrt(dot(g, g)) <= 1e-8) break;
    let direction = twoLoopDirection(g, sHist, yHist).map((value) => -value);
    let descent = dot(g, direction);
    if (descent >= 0.0) {                    // safeguard: fall back to steepest descent
      direction = g.map((value) => -value);
      descent = -dot(g, g);
    }
    let step = 1.0;
    while (objective(x.map((xi, i) => xi + step * direction[i])) >
           objective(x) + C1 * step * descent) {
      step *= RHO;
    }
    const xNew = x.map((xi, i) => xi + step * direction[i]);
    const gNew = gradient(xNew);
    const s = xNew.map((value, i) => value - x[i]);
    const y = gNew.map((value, i) => value - g[i]);
    if (dot(s, y) > 0.0) {                   // curvature condition
      sHist.push(s);
      yHist.push(y);
      if (sHist.length > HISTORY) {
        sHist.shift();
        yHist.shift();
      }
    }
    x = xNew;
    g = gNew;
    iterations += 1;
  }
  return [x, g, iterations];
}

function gd(tolerance) {
  const eta = 1.0 / largestEigenvalue(A);    // documented GD step size
  let x = X0.slice();
  let g = gradient(x);
  let iterations = 0;
  while (Math.sqrt(dot(g, g)) > tolerance && iterations < GD_MAX_ITER) {
    x = x.map((xi, i) => xi - eta * g[i]);
    g = gradient(x);
    iterations += 1;
  }
  return [x, g, iterations];
}

function main() {
  const assert = require("node:assert");

  const xStar = solveLinear(A, B);
  const fStar = objective(xStar);
  assert.ok(Math.max(...gradient(xStar).map(Math.abs)) <= 1e-12, "oracle is exact");

  // Skeleton sanity: with an empty history H = I, so H*g = g and the step is -g.
  const g0 = gradient(X0);
  assert.deepStrictEqual(twoLoopDirection(g0, [], []), g0);

  const [xLb, gLb, lbIters] = lbfgs();
  assert.ok(Math.sqrt(dot(gLb, gLb)) <= 1e-8, "documented L-BFGS gradient tolerance");
  assert.ok(Math.max(...xLb.map((value, i) => Math.abs(value - xStar[i]))) <= 1e-6);
  assert.ok(Math.abs(objective(xLb) - fStar) <= 1e-9);
  assert.ok(lbIters <= 20, "L-BFGS must converge in few iterations on a 3-dim quadratic");

  const [xGd, gGd, gdIters] = gd(1e-6);
  assert.ok(Math.sqrt(dot(gGd, gGd)) <= 1e-6, "GD must converge within the cap");
  assert.ok(gdIters > 2 * lbIters, "L-BFGS must need far fewer iterations");
  assert.ok(Math.max(...xGd.map((value, i) => Math.abs(value - xStar[i]))) <= 1e-4);

  const [xRepeat, , repeatIters] = lbfgs();
  assert.strictEqual(repeatIters, lbIters);
  assert.deepStrictEqual(xRepeat, xLb);
}

module.exports = { A, B, X0, dot, matvec, objective, gradient, solveLinear, largestEigenvalue,
  twoLoopDirection, lbfgs, gd };

if (require.main === module) {
  main();
  console.log("PASS");
}
