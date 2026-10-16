/** Task 697 - Implement iterative linear solver (CG) with preconditioning demo. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Model: the conjugate gradient method for symmetric positive definite systems, with and without a
 * preconditioner, plus its preconditioned form (PCG) and the machinery needed to certify it:
 *
 *     r_0 = b - A x_0,  z_0 = M^-1 r_0,  p_0 = z_0
 *     alpha_k = (r_k, z_k) / (p_k, A p_k)
 *     x_{k+1} = x_k + alpha_k p_k,  r_{k+1} = r_k - alpha_k A p_k,
 *     beta_k = (r_{k+1}, z_{k+1}) / (r_k, z_k),  p_{k+1} = z_{k+1} + beta_k p_k
 * with the Jacobi preconditioner M = diag(A) (a plain reciprocal of the diagonal).
 *
 * The test problems are the 1D Poisson matrix A = (1/h^2) tridiag(-1, 2, -1) (its eigenvalues are
 * known in closed form, so the condition number and therefore CG's theoretical iteration bound are
 * known too), a badly scaled diagonal matrix, and a small hand-written SPD matrix whose solution is
 * known. Correctness is pinned by an independent dense LU solver with partial pivoting: the CG
 * iterate must reach the direct solution, and the residual must shrink below the requested tolerance.
 *
 * Verified: symmetry detection and rejection of non-symmetric input, agreement with the direct solve,
 * the theoretical iteration bound sqrt(kappa)/2 * ln(2/eps) + 1 for Poisson, the A-norm error
 * decreasing monotonically (the defining minimisation property of CG), mutual orthogonality of the
 * residuals, one-iteration convergence of Jacobi-PCG on a diagonal matrix, fewer iterations with
 * preconditioning on Poisson than without, refusal to converge on an indefinite matrix, exactness on
 * identity/1x1/zero right-hand sides, and determinism.
 */
"use strict";

function zeros(n) {
  return new Array(n).fill(0);
}

function copy(vector) {
  return vector.slice();
}

function dot(left, right) {
  let total = 0;
  for (let i = 0; i < left.length; i++) total += left[i] * right[i];
  return total;
}

function norm2(vector) {
  return Math.sqrt(dot(vector, vector));
}

/** y = A x for a dense row-major matrix. */
function matVec(matrix, vector) {
  const n = matrix.length;
  const out = zeros(n);
  for (let i = 0; i < n; i++) {
    let total = 0;
    const row = matrix[i];
    for (let j = 0; j < n; j++) total += row[j] * vector[j];
    out[i] = total;
  }
  return out;
}

function subtract(left, right) {
  return left.map((value, index) => value - right[index]);
}

function residual(matrix, x, b) {
  return subtract(b, matVec(matrix, x));
}

/** 1D Poisson matrix with Dirichlet boundary conditions on n interior points. */
function buildPoisson(n) {
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

/** Closed-form eigenvalues of the 1D Poisson matrix: 4/h^2 sin^2(k pi h / 2). */
function poissonEigenvalues(n) {
  const h = 1 / (n + 1);
  const values = [];
  for (let k = 1; k <= n; k++) values.push((4 / (h * h)) * Math.sin((k * Math.PI * h) / 2) ** 2);
  return values;
}

function conditionNumber(n) {
  const values = poissonEigenvalues(n);
  return values[values.length - 1] / values[0];
}

function isSymmetric(matrix, tolerance = 0) {
  const n = matrix.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(matrix[i][j] - matrix[j][i]) > tolerance) return false;
    }
  }
  return true;
}

/** max|A_ij - A_ji| / max|A_ij|: the symmetry defect relative to the matrix scale. */
function symmetryDefect(matrix) {
  const n = matrix.length;
  let worst = 0;
  let scale = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      worst = Math.max(worst, Math.abs(matrix[i][j] - matrix[j][i]));
      scale = Math.max(scale, Math.abs(matrix[i][j]));
    }
  }
  return scale === 0 ? 0 : worst / scale;
}

/** Jacobi preconditioner: solve M z = r with M = diag(A). */
function jacobiPreconditioner(matrix) {
  const inverseDiagonal = matrix.map((row, index) => {
    if (row[index] === 0) throw new RangeError("Jacobi preconditioning needs a non-zero diagonal");
    return 1 / row[index];
  });
  return (r) => r.map((value, index) => value * inverseDiagonal[index]);
}

const identityPreconditioner = (r) => r.slice();

/** Conjugate gradient (preconditioned when a preconditioner is given). */
function conjugateGradient(matrix, b, options = {}) {
  const { tolerance = 1e-10, maxIterations = 10 * matrix.length, preconditioner = identityPreconditioner } = options;
  if (symmetryDefect(matrix) > 1e-12) throw new Error("CG requires a symmetric matrix");
  const n = matrix.length;
  const x = zeros(n);
  let r = residual(matrix, x, b);
  let z = preconditioner(r);
  let p = z.slice();
  let rz = dot(r, z);
  const residuals = [norm2(r)];
  const toleranceAbsolute = tolerance * Math.max(norm2(b), 1);
  let iterations = 0;

  while (iterations < maxIterations) {
    const ap = matVec(matrix, p);
    const denominator = dot(p, ap);
    if (denominator === 0) break;
    const alpha = rz / denominator;
    for (let i = 0; i < n; i++) x[i] += alpha * p[i];
    for (let i = 0; i < n; i++) r[i] -= alpha * ap[i];
    iterations++;
    residuals.push(norm2(r));
    if (norm2(r) <= toleranceAbsolute) break;
    z = preconditioner(r);
    const rzNext = dot(r, z);
    const beta = rzNext / rz;
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    rz = rzNext;
  }
  return { x, iterations, residuals, converged: norm2(r) <= toleranceAbsolute };
}

/** Independent dense solver: Gaussian elimination with partial pivoting. */
function directSolve(matrix, b) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  const rhs = b.slice();
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    if (Math.abs(a[pivot][column]) < 1e-300) throw new Error("singular matrix");
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

/**
 * Raw CG steps, exposing the iterate and residual history - used to check the minimisation property
 * (the A-norm of the error never grows) and the orthogonality of the residuals.
 */
function cgIterates(matrix, b, count) {
  const n = matrix.length;
  const x = zeros(n);
  let r = residual(matrix, x, b);
  let p = r.slice();
  let rr = dot(r, r);
  const iterates = [x.slice()];
  const residuals = [r.slice()];
  for (let k = 0; k < count; k++) {
    const ap = matVec(matrix, p);
    const denominator = dot(p, ap);
    if (denominator === 0) break;
    const alpha = rr / denominator;
    for (let i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * ap[i]; }
    const rrNext = dot(r, r);
    const beta = rrNext / rr;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    rr = rrNext;
    iterates.push(x.slice());
    residuals.push(r.slice());
  }
  return { iterates, residuals };
}

/** sqrt(x^T A x): the energy norm CG minimises. */
function energyNorm(matrix, vector) {
  return Math.sqrt(dot(vector, matVec(matrix, vector)));
}

function maxDifference(left, right) {
  let worst = 0;
  for (let i = 0; i < left.length; i++) worst = Math.max(worst, Math.abs(left[i] - right[i]));
  return worst;
}

module.exports = {
  zeros, copy, dot, norm2, matVec, subtract, residual, buildPoisson, poissonEigenvalues,
  conditionNumber, isSymmetric, symmetryDefect, jacobiPreconditioner, identityPreconditioner,
  conjugateGradient, cgIterates, directSolve, energyNorm, maxDifference,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- symmetry detection ---------------------------------------------------------------------
  const poisson8 = buildPoisson(8);
  assert.strictEqual(isSymmetric(poisson8, 0), true, "the Poisson matrix is symmetric");
  assert.strictEqual(isSymmetric(poisson8, 1e-12), true);
  const asymmetric = [[1, 2], [1, 1]];
  assert.strictEqual(isSymmetric(asymmetric), false);
  assert.throws(() => conjugateGradient(asymmetric, [1, 1]), Error, "CG rejects a non-symmetric matrix");
  assert.throws(() => jacobiPreconditioner([[0, 0], [0, 1]]), RangeError);
  assert.deepStrictEqual(poisson8[0].slice(0, 3), [2 * 81, -81, 0], "the stencil is (1/h^2)[-1, 2, -1]");

  // --- closed-form spectrum and condition number ----------------------------------------------
  const eigenvalues = poissonEigenvalues(8);
  assert.strictEqual(eigenvalues.length, 8);
  assert.ok(Math.abs(eigenvalues[0] - 4 * 81 * Math.sin(Math.PI / 18) ** 2) < 1e-9);
  assert.ok(eigenvalues.every((value, index) => index === 0 || value > eigenvalues[index - 1]),
    "the eigenvalues are increasing");
  assert.ok(Math.abs(conditionNumber(8) - 1 / Math.tan(Math.PI / 18) ** 2) < 1e-9,
    "kappa = cot^2(pi h / 2) for the 1D Poisson matrix");
  assert.ok(conditionNumber(64) > 10 * conditionNumber(8), "longer intervals are worse conditioned");

  // --- a small hand-checkable system ----------------------------------------------------------
  const handMatrix = [[4, 1], [1, 3]];
  const handRhs = [1, 2];
  const handExact = directSolve(handMatrix, handRhs);
  assert.ok(Math.abs(handExact[0] - 1 / 11) < 1e-12, "the analytic solution x = (1/11, 7/11)");
  assert.ok(Math.abs(handExact[1] - 7 / 11) < 1e-12);
  const hand = conjugateGradient(handMatrix, handRhs, { tolerance: 1e-12 });
  assert.ok(maxDifference(hand.x, handExact) < 1e-9, "CG reaches the direct solution");
  assert.ok(hand.iterations <= 2, "a 2x2 SPD system takes at most two directions");

  // --- Poisson: direct agreement, iteration bound, A-norm monotonicity, orthogonality ---------
  const n = 16;
  const matrix = buildPoisson(n);
  const exactRhs = Array.from({ length: n }, (_, i) => Math.sin(((i + 1) * Math.PI) / (n + 1)) ** 2);
  const direct = directSolve(matrix, exactRhs);
  const cg = conjugateGradient(matrix, exactRhs, { tolerance: 1e-12 });
  assert.ok(maxDifference(cg.x, direct) < 1e-8, `CG matches the direct solve (${maxDifference(cg.x, direct)})`);
  assert.ok(norm2(residual(matrix, cg.x, exactRhs)) < 1e-9 * norm2(exactRhs), "and the residual is tiny");
  assert.ok(cg.converged);

  const kappa = conditionNumber(n);
  const bound = Math.ceil(0.5 * Math.sqrt(kappa) * Math.log(2 / 1e-12) + 1);
  assert.ok(cg.iterations <= bound, `CG respects the theoretical bound (${cg.iterations} <= ${bound})`);
  assert.ok(cg.iterations <= n + 2, "and converges in about n steps");

  // the A-norm of the error decreases monotonically, which is what CG minimises
  const { iterates, residuals: residualHistory } = cgIterates(matrix, exactRhs, 6);
  const aNorms = iterates.map((x) => energyNorm(matrix, subtract(direct, x)));
  for (let k = 1; k < aNorms.length; k++) {
    assert.ok(aNorms[k] <= aNorms[k - 1] + 1e-12 * aNorms[0], `A-norm error decreases at step ${k}`);
  }
  assert.ok(aNorms[0] > aNorms[aNorms.length - 1] * 2, `and the decrease is large (${aNorms[0]} -> ${aNorms[aNorms.length - 1]})`);
  assert.strictEqual(aNorms.length, 7);

  // residuals are mutually orthogonal: the defining CG identity r_i . r_j = 0
  for (let i = 0; i < residualHistory.length; i++) {
    for (let j = i + 1; j < residualHistory.length; j++) {
      const relative = Math.abs(dot(residualHistory[i], residualHistory[j]))
        / (norm2(residualHistory[i]) * norm2(residualHistory[j]));
      assert.ok(relative < 1e-8, `residuals ${i} and ${j} are orthogonal (${relative})`);
    }
  }
  assert.strictEqual(residualHistory.length, 7);

  // --- preconditioning -------------------------------------------------------------------------
  const diagonal = Array.from({ length: 9 }, () => zeros(9));
  for (let i = 0; i < 9; i++) diagonal[i][i] = 2 ** (2 * i);
  const diagonalRhs = new Array(9).fill(1);
  const plainDiagonal = conjugateGradient(diagonal, diagonalRhs, { tolerance: 1e-10, maxIterations: 200 });
  const jackobiDiagonal = conjugateGradient(diagonal, diagonalRhs, {
    tolerance: 1e-10, maxIterations: 200, preconditioner: jacobiPreconditioner(diagonal),
  });
  assert.strictEqual(plainDiagonal.converged, true);
  assert.ok(plainDiagonal.iterations > 8, `the identity preconditioner needs many steps (${plainDiagonal.iterations})`);
  assert.strictEqual(jackobiDiagonal.iterations, 1, "Jacobi makes a diagonal system a single step");
  assert.ok(maxDifference(jackobiDiagonal.x, directSolve(diagonal, diagonalRhs)) < 1e-9);

  // On a *constant* diagonal the Jacobi preconditioner is a scalar multiple of the identity, so it
  // cannot change the CG iterates at all - that is a fact worth pinning down. The interesting case
  // is a matrix whose diagonal varies, which is what moving Jacobi helps with.
  const constantDiagonal = conjugateGradient(matrix, exactRhs, {
    tolerance: 1e-10, preconditioner: jacobiPreconditioner(matrix),
  });
  const plainConstant = conjugateGradient(matrix, exactRhs, { tolerance: 1e-10 });
  assert.strictEqual(constantDiagonal.iterations, plainConstant.iterations,
    "Jacobi on a constant-diagonal matrix is a no-op");
  assert.ok(maxDifference(constantDiagonal.x, plainConstant.x) < 1e-9);

  const n32 = 32;
  const poisson32 = buildPoisson(n32);
  const scaling = Array.from({ length: n32 }, (_, i) => 1 + (i / (n32 - 1)) * 9);
  const scaledPoisson = poisson32.map((row, i) => row.map((value, j) => value * scaling[i] * scaling[j]));
  const scaledRhs = Array.from({ length: n32 }, (_, i) => Math.sin(((i + 1) * Math.PI) / (n32 + 1)) ** 2);
  assert.ok(symmetryDefect(scaledPoisson) < 1e-15, "the scaled matrix is symmetric to round-off");
  assert.strictEqual(symmetryDefect(poisson32), 0);
  const plainScaled = conjugateGradient(scaledPoisson, scaledRhs, { tolerance: 1e-10, maxIterations: 500 });
  const jacobiScaled = conjugateGradient(scaledPoisson, scaledRhs, {
    tolerance: 1e-10, maxIterations: 500, preconditioner: jacobiPreconditioner(scaledPoisson),
  });
  const scaledDirect = directSolve(scaledPoisson, scaledRhs);
  assert.strictEqual(jacobiScaled.converged, true);
  assert.ok(jacobiScaled.iterations < plainScaled.iterations,
    `Jacobi cuts the iteration count on a varying diagonal (${jacobiScaled.iterations} < ${plainScaled.iterations})`);
  assert.ok(maxDifference(jacobiScaled.x, scaledDirect) < 1e-6, "and still lands on the right solution");
  assert.ok(maxDifference(plainScaled.x, scaledDirect) < 1e-6);
  assert.ok(norm2(residual(scaledPoisson, jacobiScaled.x, scaledRhs)) < 1e-6 * norm2(scaledRhs));

  // --- edge cases ------------------------------------------------------------------------------
  const zeroRhs = zeros(n);
  const trivial = conjugateGradient(matrix, zeroRhs, { tolerance: 1e-10 });
  assert.strictEqual(trivial.iterations, 0, "a zero right-hand side needs no iterations");
  assert.deepStrictEqual(trivial.x, zeros(n));
  assert.deepStrictEqual(trivial.residuals, [0]);
  const identity = [[1, 0], [0, 1]];
  const identitySolve = conjugateGradient(identity, [3, -4], { tolerance: 1e-12 });
  assert.strictEqual(identitySolve.iterations, 1, "an identity system is solved in one step");
  assert.deepStrictEqual(identitySolve.x, [3, -4]);
  const oneByOne = conjugateGradient([[5]], [10], { tolerance: 1e-12 });
  assert.deepStrictEqual(oneByOne.x, [2]);
  assert.strictEqual(oneByOne.residuals.length, 2);
  // An indefinite matrix can break the method down: here the very first search direction is
  // orthogonal in the A inner product (p . A p = 0), so CG cannot even take a step.
  const degenerate = conjugateGradient([[1, 0], [0, -1]], [1, 1], { tolerance: 1e-10, maxIterations: 5 });
  assert.strictEqual(degenerate.converged, false, "CG does not converge when p . A p vanishes");
  assert.strictEqual(degenerate.iterations, 0, "it stops before the first step");
  assert.deepStrictEqual(degenerate.x, [0, 0]);
  assert.strictEqual(degenerate.residuals.length, 1);
  assert.ok(subtract && copy([1, 2]).length === 2);

  // --- determinism ------------------------------------------------------------------------------
  const again = conjugateGradient(matrix, exactRhs, { tolerance: 1e-12 });
  assert.deepStrictEqual(again.x, cg.x);
  assert.deepStrictEqual(again.residuals, cg.residuals);

  console.log(`poisson n=${n} kappa=${kappa.toFixed(1)} cg=${cg.iterations} pcg=${constantDiagonal.iterations}`);
  console.log(`scaled n=${n32} cg=${plainScaled.iterations} pcg=${jacobiScaled.iterations} diag pcg=${jackobiDiagonal.iterations}`);
  console.log("PASS");
}
