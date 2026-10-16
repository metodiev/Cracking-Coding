/** Task 698 - Implement multigrid method outline for Poisson equation small grid. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Model: a geometric multigrid V-cycle for the 1D Poisson problem -u'' = f with Dirichlet boundary
 * conditions on n interior points, h = 1/(n+1), A = (1/h^2) tridiag(-1, 2, -1). The components are
 *
 *   smoother      weighted Jacobi, u <- u + omega D^-1 (f - A u) with omega = 2/3,
 *   residual      r = f - A u,
 *   restriction   R = 1/4 [1 2 1] (full weighting to the coarse grid of (n-1)/2 points),
 *   prolongation  P = 1/2 [1 2 1]^T (linear interpolation),
 *   coarse solve  the coarsest grid (n <= 3) is solved by the Thomas tridiagonal algorithm,
 *   V-cycle       nu1 pre-smooths, recurse on R r, prolongate and correct, nu2 post-smooths.
 * The coarse operator is the *Galerkin* product A_c = R A P, so the correction is consistent by
 * construction; with these weights it coincides exactly with the coarse-grid discretisation, which is
 * checked numerically instead of assumed.
 *
 * A scaled-down 2D version (5-point stencil, full-weighting restriction, bilinear prolongation, the
 * coarsest 3x3 grid solved densely) is included to show the same machinery in two dimensions; it is
 * deliberately small so the whole run stays fast.
 *
 * Simplified versus a production solver: only a single V-cycle (no F-cycles, no FMG), one smoother
 * (weighted Jacobi, no Gauss-Seidel/red-black ordering), direct solves are dense/Thomas because the
 * grids are small, and the 2D case is limited to a few levels.
 *
 * Verified: the Galerkin coarse operator matches the coarse Poisson stencil with a grid-independent
 * factor, the two-grid/V-cycle convergence factor is bounded well below one and nearly independent of
 * n, plain Jacobi degrades towards a factor of 1 as n grows (the reason multigrid exists), the number
 * of cycles to a fixed tolerance stays flat, the discrete solution matches the Thomas solve and the
 * analytic sine solution to O(h^2), the 2D V-cycle converges and matches a dense solve, and the
 * degenerate (n = 1, 3) and zero-right-hand-side cases are handled.
 */
"use strict";

function zeros(n) {
  return new Array(n).fill(0);
}

function norm2(vector) {
  let total = 0;
  for (const value of vector) total += value * value;
  return Math.sqrt(total);
}

function maxAbs(vector) {
  let worst = 0;
  for (const value of vector) worst = Math.max(worst, Math.abs(value));
  return worst;
}

/** A x for the 1D Poisson operator on n interior points. */
function applyPoisson(n, u) {
  const h = 1 / (n + 1);
  const scale = 1 / (h * h);
  const out = zeros(n);
  for (let i = 0; i < n; i++) {
    let value = 2 * u[i];
    if (i > 0) value -= u[i - 1];
    if (i + 1 < n) value -= u[i + 1];
    out[i] = scale * value;
  }
  return out;
}

function residual(n, f, u) {
  const au = applyPoisson(n, u);
  return f.map((value, index) => value - au[index]);
}

/** Weighted Jacobi smoothing performed in place. */
function smooth(n, f, u, sweeps, omega = 2 / 3) {
  const h = 1 / (n + 1);
  const diagonal = 2 / (h * h);
  for (let sweep = 0; sweep < sweeps; sweep++) {
    const next = u.slice();
    for (let i = 0; i < n; i++) {
      let value = 2 * u[i];
      if (i > 0) value -= u[i - 1];
      if (i + 1 < n) value -= u[i + 1];
      next[i] = u[i] + (omega * (f[i] - diagonal * 0.5 * value)) / diagonal;
    }
    for (let i = 0; i < n; i++) u[i] = next[i];
  }
  return u;
}

/** R r: full weighting from n fine points to (n-1)/2 coarse points. */
function restrict1D(n, fine) {
  const coarse = (n - 1) / 2;
  if (!Number.isInteger(coarse) || coarse < 1) throw new RangeError("the fine grid must be 2m+1 points");
  const out = zeros(coarse);
  for (let i = 0; i < coarse; i++) {
    const centre = 2 * i + 1;
    const left = centre - 1;
    const right = centre + 1;
    out[i] = 0.25 * (left >= 0 ? fine[left] : 0) + 0.5 * fine[centre]
      + 0.25 * (right < n ? fine[right] : 0);
  }
  return out;
}

/** P e: linear interpolation from the coarse grid back to the fine grid. */
function prolong1D(coarse, fine) {
  const n = 2 * coarse + 1;
  const out = zeros(n);
  for (let i = 0; i < coarse; i++) {
    out[2 * i + 1] += fine[i];
    out[2 * i] += 0.5 * fine[i];
    if (2 * i + 2 < n) out[2 * i + 2] += 0.5 * fine[i];
  }
  return out;
}

/** The Galerkin coarse operator A_c = R A P as a dense matrix (small grids only). */
function galerkinCoarse(n) {
  const coarse = (n - 1) / 2;
  const matrix = Array.from({ length: coarse }, () => zeros(coarse));
  for (let j = 0; j < coarse; j++) {
    const basis = zeros(coarse);
    basis[j] = 1;
    const fine = prolong1D(coarse, basis);
    const image = applyPoisson(n, fine);
    const restricted = restrict1D(n, image);
    for (let i = 0; i < coarse; i++) matrix[i][j] = restricted[i];
  }
  return matrix;
}

/** Dense Poisson matrix, used for the coarse-operator comparison and the 2D reference solve. */
function densePoisson(n) {
  const h = 1 / (n + 1);
  const scale = 1 / (h * h);
  const matrix = Array.from({ length: n }, () => zeros(n));
  for (let i = 0; i < n; i++) {
    matrix[i][i] = 2 * scale;
    if (i > 0) matrix[i][i - 1] = -scale;
    if (i + 1 < n) matrix[i][i + 1] = -scale;
  }
  return matrix;
}

/** Thomas algorithm: exact solve of the 1D Poisson system, also the coarsest-grid solver. */
function thomasSolve(f) {
  const n = f.length;
  const h = 1 / (n + 1);
  const scale = 1 / (h * h);
  const c = new Array(n).fill(0);
  const d = new Array(n).fill(0);
  const x = zeros(n);
  c[0] = -scale / (2 * scale);
  d[0] = f[0] / (2 * scale);
  for (let i = 1; i < n; i++) {
    const denominator = 2 * scale + scale * c[i - 1];      // sub-diagonal is -scale
    c[i] = i + 1 < n ? -scale / denominator : 0;
    d[i] = (f[i] + scale * d[i - 1]) / denominator;
  }
  x[n - 1] = d[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = d[i] - c[i] * x[i + 1];
  return x;
}

/** One V-cycle: pre-smooth, recurse on the coarse residual, correct, post-smooth. */
function vcycle1D(n, f, u, { preSmooths = 2, postSmooths = 2, coarsest = 3 } = {}) {
  if (n <= coarsest) {
    const exact = thomasSolve(f);
    for (let i = 0; i < n; i++) u[i] = exact[i];
    return u;
  }
  smooth(n, f, u, preSmooths);
  const r = residual(n, f, u);
  const coarseRhs = restrict1D(n, r);
  const coarseCorrection = zeros(coarseRhs.length);
  vcycle1D(coarseRhs.length, coarseRhs, coarseCorrection, { preSmooths, postSmooths, coarsest });
  const correction = prolong1D(coarseRhs.length, coarseCorrection);
  for (let i = 0; i < n; i++) u[i] += correction[i];
  smooth(n, f, u, postSmooths);
  return u;
}

function multigridSolve(n, f, cycles, options = {}) {
  const u = zeros(n);
  const history = [norm2(residual(n, f, u))];
  for (let cycle = 0; cycle < cycles; cycle++) {
    vcycle1D(n, f, u, options);
    history.push(norm2(residual(n, f, u)));
  }
  return { u, history };
}

function jacobiSolve(n, f, cycles, omega = 2 / 3) {
  const u = zeros(n);
  const history = [norm2(residual(n, f, u))];
  for (let cycle = 0; cycle < cycles; cycle++) {
    smooth(n, f, u, 1, omega);
    history.push(norm2(residual(n, f, u)));
  }
  return { u, history };
}

/** Geometric mean of the residual reduction per cycle over a run. */
function convergenceFactor(history) {
  const steps = history.length - 1;
  if (steps <= 0) return 1;
  const ratio = history[history.length - 1] / history[0];
  if (!(ratio > 0)) return 0;                          // already at (or below) the starting residual
  return ratio ** (1 / steps);
}

// --- a small two-dimensional V-cycle ---------------------------------------------------------

function index2D(i, j, n) {
  return i * n + j;
}

function applyPoisson2D(n, u) {
  const h = 1 / (n + 1);
  const scale = 1 / (h * h);
  const out = zeros(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let value = 4 * u[index2D(i, j, n)];
      value -= i > 0 ? u[index2D(i - 1, j, n)] : 0;
      value -= i + 1 < n ? u[index2D(i + 1, j, n)] : 0;
      value -= j > 0 ? u[index2D(i, j - 1, n)] : 0;
      value -= j + 1 < n ? u[index2D(i, j + 1, n)] : 0;
      out[index2D(i, j, n)] = scale * value;
    }
  }
  return out;
}

function residual2D(n, f, u) {
  const au = applyPoisson2D(n, u);
  return f.map((value, index) => value - au[index]);
}

function smooth2D(n, f, u, sweeps, omega = 2 / 3) {
  const h = 1 / (n + 1);
  const scale = 1 / (h * h);
  const diagonal = 4 * scale;
  for (let sweep = 0; sweep < sweeps; sweep++) {
    const next = u.slice();
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let stencil = 4 * u[index2D(i, j, n)];
        stencil -= i > 0 ? u[index2D(i - 1, j, n)] : 0;
        stencil -= i + 1 < n ? u[index2D(i + 1, j, n)] : 0;
        stencil -= j > 0 ? u[index2D(i, j - 1, n)] : 0;
        stencil -= j + 1 < n ? u[index2D(i, j + 1, n)] : 0;
        const index = index2D(i, j, n);
        next[index] = u[index] + (omega * (f[index] - scale * stencil)) / diagonal;
      }
    }
    for (let i = 0; i < u.length; i++) u[i] = next[i];
  }
  return u;
}

/** Full weighting (1/16 * [1 2 1; 2 4 2; 1 2 1]) from n x n to ((n-1)/2) x ((n-1)/2). */
function restrict2D(n, fine) {
  const coarse = (n - 1) / 2;
  const out = zeros(coarse * coarse);
  const weights = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
  for (let i = 0; i < coarse; i++) {
    for (let j = 0; j < coarse; j++) {
      let total = 0;
      for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
          const si = 2 * i + 1 + di;
          const sj = 2 * j + 1 + dj;
          if (si < 0 || sj < 0 || si >= n || sj >= n) continue;
          total += weights[di + 1][dj + 1] * fine[index2D(si, sj, n)];
        }
      }
      out[index2D(i, j, coarse)] = total / 16;
    }
  }
  return out;
}

/** Bilinear interpolation from the coarse grid to the fine grid. */
function prolong2D(coarse, fine) {
  const n = 2 * coarse + 1;
  const out = zeros(n * n);
  for (let i = 0; i < coarse; i++) {
    for (let j = 0; j < coarse; j++) {
      const value = fine[index2D(i, j, coarse)];
      const centre = [2 * i + 1, 2 * j + 1];
      for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
          const fi = centre[0] + di;
          const fj = centre[1] + dj;
          if (fi < 0 || fj < 0 || fi >= n || fj >= n) continue;
          const weight = (di === 0 ? 1 : 0.5) * (dj === 0 ? 1 : 0.5);
          out[index2D(fi, fj, n)] += weight * value;
        }
      }
    }
  }
  return out;
}

/** Dense Gaussian elimination with partial pivoting (the 2D reference solve). */
function denseSolve(matrix, b) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  const rhs = b.slice();
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    [a[column], a[pivot]] = [a[pivot], a[column]];
    [rhs[column], rhs[pivot]] = [rhs[pivot], rhs[column]];
    for (let row = column + 1; row < n; row++) {
      const factor = a[row][column] / a[column][column];
      if (factor === 0) continue;
      for (let k = column; k < n; k++) a[row][k] -= factor * a[column][k];
      rhs[row] -= factor * rhs[column];
    }
  }
  const x = zeros(n);
  for (let row = n - 1; row >= 0; row--) {
    let total = rhs[row];
    for (let column = row + 1; column < n; column++) total -= a[row][column] * x[column];
    x[row] = total / a[row][row];
  }
  return x;
}

function densePoisson2D(n) {
  const size = n * n;
  const matrix = Array.from({ length: size }, () => zeros(size));
  const h = 1 / (n + 1);
  const scale = 1 / (h * h);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const row = index2D(i, j, n);
      matrix[row][row] = 4 * scale;
      const neighbours = [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]];
      for (const [ni, nj] of neighbours) {
        if (ni < 0 || nj < 0 || ni >= n || nj >= n) continue;
        matrix[row][index2D(ni, nj, n)] = -scale;
      }
    }
  }
  return matrix;
}

function vcycle2D(n, f, u, { preSmooths = 2, postSmooths = 2, coarsest = 3 } = {}) {
  if (n <= coarsest) {
    const exact = denseSolve(densePoisson2D(n), f);
    for (let i = 0; i < u.length; i++) u[i] = exact[i];
    return u;
  }
  smooth2D(n, f, u, preSmooths);
  const r = residual2D(n, f, u);
  const coarseRhs = restrict2D(n, r);
  const coarse = (n - 1) / 2;
  const coarseCorrection = zeros(coarse * coarse);
  vcycle2D(coarse, coarseRhs, coarseCorrection, { preSmooths, postSmooths, coarsest });
  const correction = prolong2D(coarse, coarseCorrection);
  for (let i = 0; i < u.length; i++) u[i] += correction[i];
  smooth2D(n, f, u, postSmooths);
  return u;
}

function multigridSolve2D(n, f, cycles, options = {}) {
  const u = zeros(n * n);
  const history = [norm2(residual2D(n, f, u))];
  for (let cycle = 0; cycle < cycles; cycle++) {
    vcycle2D(n, f, u, options);
    history.push(norm2(residual2D(n, f, u)));
  }
  return { u, history };
}

module.exports = {
  zeros, norm2, maxAbs, applyPoisson, residual, smooth, restrict1D, prolong1D, galerkinCoarse,
  densePoisson, thomasSolve, vcycle1D, multigridSolve, jacobiSolve, convergenceFactor,
  applyPoisson2D, residual2D, smooth2D, restrict2D, prolong2D, denseSolve, densePoisson2D,
  vcycle2D, multigridSolve2D, index2D,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- the Galerkin coarse operator -----------------------------------------------------------
  for (const n of [7, 15, 31]) {
    const coarse = (n - 1) / 2;
    const galerkin = galerkinCoarse(n);
    const direct = densePoisson(coarse);
    let worst = 0;
    for (let i = 0; i < coarse; i++) {
      for (let j = 0; j < coarse; j++) worst = Math.max(worst, Math.abs(galerkin[i][j] - direct[i][j]));
    }
    assert.ok(worst < 1e-9, `R A P is exactly the coarse Poisson stencil (n=${n}, worst ${worst})`);
    assert.ok(Math.abs(galerkin[0][1] + galerkin[0][0] / 2) < 1e-9, "the stencil is [-1, 2, -1] scaled");
    assert.ok(galerkin[0].every((value, index) => index <= 1 || Math.abs(value) < 1e-12),
      "and it is tridiagonal");
    assert.ok(galerkin.every((row, index) => row[index] > 0), "with a positive diagonal");
  }

  // --- restriction and prolongation are adjoint-ish and shape preserving ---------------------
  const fineExample = [1, 2, 3, 4, 5, 6, 7];
  const restrictedExample = restrict1D(7, fineExample);
  assert.deepStrictEqual(restrictedExample, [0.25 * 1 + 0.5 * 2 + 0.25 * 3, 0.25 * 3 + 0.5 * 4 + 0.25 * 5,
    0.25 * 5 + 0.5 * 6 + 0.25 * 7]);
  assert.deepStrictEqual(prolong1D(3, [1, 1, 1]), [0.5, 1, 1, 1, 1, 1, 0.5]);
  assert.throws(() => restrict1D(8, new Array(8).fill(0)), RangeError);
  assert.deepStrictEqual(restrict1D(7, new Array(7).fill(0)), [0, 0, 0]);

  // --- convergence factors --------------------------------------------------------------------
  const multigridFactors = [];
  const cyclesNeeded = [];
  for (const n of [15, 31, 63]) {
    const f = Array.from({ length: n }, (_, i) => Math.sin(((i + 1) * Math.PI) / (n + 1)));
    const run = multigridSolve(n, f, 10);
    const factor = convergenceFactor(run.history);
    multigridFactors.push(factor);
    assert.ok(factor < 0.1, `a V-cycle reduces the residual fast for n=${n} (factor ${factor})`);
    const long = multigridSolve(n, f, 20);
    const needed = long.history.findIndex((value, index) => index > 0 && value <= 1e-8 * long.history[0]);
    cyclesNeeded.push(needed);
    assert.ok(needed > 0 && needed <= 8, `n=${n} reaches a 1e-8 reduction in ${needed} V-cycles`);
    assert.ok(long.history[6] / long.history[0] < 1e-6, `six cycles give at least six digits for n=${n}`);
  }
  assert.ok(Math.max(...multigridFactors) / Math.min(...multigridFactors) < 2.5,
    `the multigrid rate hardly depends on n (${multigridFactors.map((v) => v.toFixed(3)).join(", ")})`);
  assert.ok(Math.max(...cyclesNeeded) - Math.min(...cyclesNeeded) <= 2,
    `the cycle count to a fixed tolerance stays flat (${cyclesNeeded.join(",")})`);

  const jacobiFactors = [15, 31, 63].map((n) => convergenceFactor(jacobiSolve(n,
    Array.from({ length: n }, (_, i) => Math.sin(((i + 1) * Math.PI) / (n + 1))), 20).history));
  assert.ok(jacobiFactors[2] > 0.99, `plain Jacobi stalls as n grows (${jacobiFactors[2]})`);
  assert.ok(jacobiFactors[2] > jacobiFactors[0], "and gets worse with n");
  assert.ok(multigridFactors[2] < jacobiFactors[2] / 10, "the multigrid factor is an order better");

  // --- accuracy against the direct solve and the analytic solution ----------------------------
  const n = 31;
  const h = 1 / (n + 1);
  const analytic = (x) => Math.sin(Math.PI * x);
  const rhs = Array.from({ length: n }, (_, i) => Math.PI ** 2 * analytic((i + 1) * h));
  const direct = thomasSolve(rhs);
  const solved = multigridSolve(n, rhs, 8);
  assert.ok(maxAbs(solved.u.map((value, index) => value - direct[index])) < 1e-8,
    "the V-cycle solution matches the exact discrete solution");
  const discreteError = maxAbs(solved.u.map((value, index) => value - analytic((index + 1) * h)));
  const expectedError = (Math.PI ** 2) * h * h / 12;
  assert.ok(Math.abs(discreteError - expectedError) < 0.25 * expectedError,
    `the discretisation error follows the O(h^2) constant (${discreteError} vs ${expectedError})`);
  assert.ok(maxAbs(residual(n, rhs, solved.u)) < 1e-6, "and the residual is small");
  assert.ok(solved.history[1] < solved.history[0], "the first cycle already reduces the residual");
  assert.ok(convergenceFactor(solved.history) < 0.3);

  // --- degenerate sizes -----------------------------------------------------------------------
  for (const n of [1, 3]) {
    const f = Array.from({ length: n }, (_, i) => (i + 1) * 2);
    const exact = thomasSolve(f);
    const viaCycle = multigridSolve(n, f, 1).u;
    assert.ok(maxAbs(viaCycle.map((value, index) => value - exact[index])) < 1e-9,
      `the coarsest grid is solved exactly for n=${n}`);
  }
  const zeroRun = multigridSolve(15, zeros(15), 3);
  assert.ok(maxAbs(zeroRun.u) < 1e-15, "a zero right-hand side stays zero");
  assert.strictEqual(multigridSolve(15, zeros(15), 3).history[0], 0);
  assert.strictEqual(convergenceFactor([0, 0, 0]), 0);
  assert.strictEqual(convergenceFactor([1]), 1);
  assert.deepStrictEqual(prolong1D(1, [4]), [2, 4, 2]);
  assert.deepStrictEqual(restrict1D(3, [2, 4, 2]), [3]);

  // --- the two-dimensional V-cycle -------------------------------------------------------------
  const n2 = 7;
  const h2 = 1 / (n2 + 1);
  const rhs2 = Array.from({ length: n2 * n2 }, (_, index) => {
    const i = Math.floor(index / n2) + 1;
    const j = (index % n2) + 1;
    return 2 * Math.PI ** 2 * Math.sin(Math.PI * i * h2) * Math.sin(Math.PI * j * h2);
  });
  const run2D = multigridSolve2D(n2, rhs2, 10);
  const factor2D = convergenceFactor(run2D.history);
  assert.ok(factor2D < 0.4, `the 2D V-cycle converges (factor ${factor2D})`);
  assert.ok(run2D.history[10] / run2D.history[0] < 1e-6, "ten cycles give six digits");
  const direct2D = denseSolve(densePoisson2D(n2), rhs2);
  assert.ok(maxAbs(run2D.u.map((value, index) => value - direct2D[index])) < 1e-3,
    "and it agrees with the dense solve");
  assert.ok(norm2(residual2D(n2, rhs2, run2D.u)) < 1e-4, "the 2D residual is small");
  assert.deepStrictEqual(restrict2D(3, new Array(9).fill(1))[0], (1 + 2 + 1 + 2 + 4 + 2 + 1 + 2 + 1) / 16,
    "full weighting sums to one");
  const constantProlongation = prolong2D(3, new Array(9).fill(1));
  assert.strictEqual(constantProlongation[index2D(3, 3, 7)], 1, "a coarse point maps to itself with weight one");
  assert.strictEqual(constantProlongation[index2D(1, 1, 7)], 1);
  assert.strictEqual(constantProlongation[index2D(0, 0, 7)], 0.25, "a corner only sees one coarse neighbour");
  for (let i = 1; i <= 5; i++) {
    for (let j = 1; j <= 5; j++) {
      assert.strictEqual(constantProlongation[index2D(i, j, 7)], 1, "interior interpolation weights sum to one");
    }
  }
  assert.strictEqual(constantProlongation[index2D(0, 3, 7)], 0.5, "an edge point sees half of one coarse value");
  assert.strictEqual(prolong2D(3, new Array(9).fill(0)).every((value) => value === 0), true);
  const constantLaplacian = applyPoisson2D(3, new Array(9).fill(1));
  assert.ok(Math.abs(constantLaplacian[index2D(1, 1, 3)]) < 1e-12,
    "the interior point of a constant field sees a zero Laplacian");
  assert.ok(Math.abs(constantLaplacian[index2D(0, 0, 3)]) > 0,
    "but a corner point feels the Dirichlet zero boundary");

  // --- determinism -----------------------------------------------------------------------------
  const replay = multigridSolve(n, rhs, 8);
  assert.deepStrictEqual(replay.u, solved.u);
  assert.deepStrictEqual(replay.history, solved.history);
  assert.deepStrictEqual(multigridSolve2D(n2, rhs2, 3).u, multigridSolve2D(n2, rhs2, 3).u);

  console.log(`1d factors=${multigridFactors.map((v) => v.toFixed(4)).join(",")} jacobi=${jacobiFactors[2].toFixed(4)}`);
  console.log(`cycles=${cyclesNeeded.join(",")} 2d factor=${factor2D.toFixed(4)}`);
  console.log("PASS");
}
