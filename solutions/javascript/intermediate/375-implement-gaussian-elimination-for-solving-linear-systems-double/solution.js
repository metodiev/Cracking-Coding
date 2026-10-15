/**
 * Task 375 - Gaussian elimination with partial pivoting (doubles). (Tier: Intermediate, CTCI Ch 6)
 *
 * Solve A x = b for a small dense double matrix.  Forward elimination with partial
 * pivoting (swap in the row with the largest |pivot|), then back substitution.
 * Documented conventions:
 *   * a pivot column whose remaining entries are all <= EPS in absolute value is
 *     treated as a free column - the system then has no unique solution;
 *   * after elimination, a row with an all-zero coefficient part but |rhs| > EPS is
 *     reported "inconsistent";
 *   * otherwise a rank-deficient consistent system is reported "infinite";
 *   * results are computed in doubles, so verification by substitution uses the
 *     documented tolerance TOL = 1e-9 (absolute) - the small hand-checked systems
 *     below have exact integer solutions that the elimination reproduces to well
 *     within that tolerance.
 * Sentinel: the status string ("unique", "infinite", "inconsistent") plus x = null for
 * anything that is not "unique".
 *
 * Includes systems that require a pivot swap (leading zero in the first column) and a
 * 4x4 diagonally dominant system, plus LCG cross-checks: a random integer matrix A and
 * integer x* give b = A x*; the solver must recover x* whenever the elimination reports
 * a unique solution.
 */
"use strict";

const EPS = 1e-12;      // pivot/inconsistency threshold for the elimination
const TOL = 1e-9;       // tolerance for verifying a solution by substitution

class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/** Returns {status, x} with status in {"unique", "infinite", "inconsistent"}. */
function solve(matrix, rhs) {
  const n = matrix.length;
  const augmented = matrix.map((row, i) => [...row.map(Number), Number(rhs[i])]);

  let rank = 0;
  for (let col = 0; col < n; col++) {
    let pivotRow = -1;
    let pivotSize = 0;
    for (let row = rank; row < n; row++) {
      if (Math.abs(augmented[row][col]) > pivotSize) {
        pivotSize = Math.abs(augmented[row][col]);
        pivotRow = row;
      }
    }
    if (pivotRow === -1 || pivotSize <= EPS) continue;              // free column
    [augmented[rank], augmented[pivotRow]] = [augmented[pivotRow], augmented[rank]];
    for (let row = rank + 1; row < n; row++) {
      const factor = augmented[row][col] / augmented[rank][col];
      if (factor !== 0) {
        for (let k = col; k <= n; k++) augmented[row][k] -= factor * augmented[rank][k];
      }
    }
    rank++;
  }

  for (let row = rank; row < n; row++) {
    if (Math.abs(augmented[row][n]) > EPS) return { status: "inconsistent", x: null };
  }
  if (rank < n) return { status: "infinite", x: null };

  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let total = augmented[row][n];
    for (let col = row + 1; col < n; col++) total -= augmented[row][col] * x[col];
    x[row] = total / augmented[row][row];
  }
  return { status: "unique", x };
}

/** max |A x - b| - the independent substitution check. */
function residual(matrix, x, rhs) {
  let worst = 0;
  for (let i = 0; i < matrix.length; i++) {
    let total = 0;
    for (let j = 0; j < matrix[i].length; j++) total += matrix[i][j] * x[j];
    worst = Math.max(worst, Math.abs(total - rhs[i]));
  }
  return worst;
}

function lcgSystem(lcg) {
  const size = 2 + lcg.next(3);
  const matrix = [];
  const intMatrix = [];
  for (let i = 0; i < size; i++) {
    const row = [];
    const intRow = [];
    for (let j = 0; j < size; j++) {
      const value = lcg.next(11) - 5;
      intRow.push(value);
      row.push(value);
    }
    matrix.push(row);
    intMatrix.push(intRow);
  }
  const planted = [];
  for (let i = 0; i < size; i++) planted.push(lcg.next(11) - 5);
  const rhs = [];
  for (let i = 0; i < size; i++) {
    let total = 0;
    for (let j = 0; j < size; j++) total += intMatrix[i][j] * planted[j];
    rhs.push(total);
  }
  return { matrix, rhs, planted };
}

module.exports = { solve, residual };

if (require.main === module) {
  const assert = require("node:assert");

  // 2x2, unique integer solution (2, 1)
  let result = solve([[1, 1], [1, -1]], [3, 1]);
  assert.strictEqual(result.status, "unique", "2x2 unique");
  assert.ok(Math.abs(result.x[0] - 2) < TOL && Math.abs(result.x[1] - 1) < TOL, "2x2 solution");

  // needs a pivot swap: first pivot is zero
  result = solve([[0, 1], [1, 0]], [2, 3]);
  assert.strictEqual(result.status, "unique", "pivot swap unique");
  assert.ok(Math.abs(result.x[0] - 3) < TOL && Math.abs(result.x[1] - 2) < TOL, "pivot swap solution");

  // 3x3 permutation system: x = (1, 2, 3)
  result = solve([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [3, 2, 1]);
  assert.strictEqual(result.status, "unique", "permutation unique");
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(result.x[i] - (i + 1)) < TOL, `permutation ${i}`);

  // classic 3x3 with solution (5, 3, -2): checked by hand
  const matrix3 = [[1, 1, 1], [0, 2, 5], [2, 5, -1]];
  const rhs3 = [6, -4, 27];
  assert.ok(5 + 3 - 2 === 6 && 2 * 3 + 5 * -2 === -4 && 2 * 5 + 5 * 3 + 2 === 27, "hand check 3x3");
  result = solve(matrix3, rhs3);
  assert.strictEqual(result.status, "unique", "3x3 unique");
  assert.ok(Math.abs(result.x[0] - 5) < TOL && Math.abs(result.x[1] - 3) < TOL
    && Math.abs(result.x[2] + 2) < TOL, "3x3 solution");
  assert.ok(residual(matrix3, result.x, rhs3) < TOL, "3x3 residual");

  // 4x4 diagonally dominant with solution (1, 2, 3, 4)
  const matrix4 = [[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]];
  const rhs4 = [11, 12, 13, 14];
  result = solve(matrix4, rhs4);
  assert.strictEqual(result.status, "unique", "4x4 unique");
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(result.x[i] - (i + 1)) < TOL, `4x4 ${i}`);
  assert.ok(residual(matrix4, result.x, rhs4) < TOL, "4x4 residual");

  // rank-deficient consistent -> infinite
  result = solve([[1, 1], [2, 2]], [1, 2]);
  assert.ok(result.status === "infinite" && result.x === null, "infinite sentinel");
  // rank-deficient inconsistent -> inconsistent
  result = solve([[1, 1], [2, 2]], [1, 2.5]);
  assert.ok(result.status === "inconsistent" && result.x === null, "inconsistent sentinel");
  // three-way dependent 3x3: row3 = row1 + row2
  result = solve([[1, 0, 0], [0, 1, 0], [1, 1, 0]], [1, 2, 3]);
  assert.ok(result.status === "infinite" && result.x === null, "3x3 dependent infinite");
  result = solve([[1, 0, 0], [0, 1, 0], [1, 1, 0]], [1, 2, 4]);
  assert.ok(result.status === "inconsistent" && result.x === null, "3x3 dependent inconsistent");

  // --- LCG cross-checks: b = A x* must recover x* -----------------------
  const lcg = new Lcg(37513);
  let uniqueCount = 0;
  for (let trial = 0; trial < 80; trial++) {
    const { matrix, rhs, planted } = lcgSystem(lcg);
    result = solve(matrix, rhs);
    if (result.status !== "unique") continue;      // singular random matrix; skipped
    uniqueCount++;
    for (let i = 0; i < planted.length; i++) {
      assert.ok(Math.abs(result.x[i] - planted[i]) < 1e-6, `recover planted ${trial}`);
    }
    assert.ok(residual(matrix, result.x, rhs) < TOL, `random residual ${trial}`);
  }
  assert.ok(uniqueCount >= 40, `enough unique systems, got ${uniqueCount}`);
  console.log("PASS");
}
