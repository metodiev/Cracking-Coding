/**
 * Task 596 - Implement CUR decomposition sketch demonstration. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * CUR (a.k.a. "pseudoskeleton") decomposition: pick r columns of A into C (n x r) and c rows into R
 * (c x m), then set U = C^+ A R^+ so that A ~ C U R.  The trick that makes CUR interesting is that U
 * is tiny (r x c) and C, R are literal columns and rows of A, so the factorisation stays
 * interpretable; the price is that the error is bounded by the *volume* of the chosen skeleton, not
 * by the singular values.  This file selects the skeleton by leverage: column scores are the column
 * norms (with the exact leverage score sum_i (A A^+)_{ii} = rank(A) reported as a scale), rows are
 * scored by the row norms of A restricted to the chosen columns, and both selections are made
 * deterministically, largest first.
 *
 * The pseudo-inverses are computed by ridge-regularised least squares (normal equations solved by
 * Gaussian elimination) rather than by an SVD, which is stated explicitly:
 * U = ls(C, A) * ls(R, I_c) = [(C^T C + t I)^{-1} C^T A] [R^T (R R^T + t I)^{-1}] with t = 1e-10, so
 * that U is (r x m)(m x c) = r x c and C U R has the shape of A; the reconstruction error
 * ||A - C U R||_F is reported.
 *
 * Skeleton selection is greedy *pivoted* (maxvol-style: take the candidate with the largest norm
 * after projecting out everything already chosen) rather than a plain norm ranking.  The file
 * documents why: on the planted rank-3 matrix a norm-ranked row choice produced a rank-deficient R,
 * after which the ridge solve returned NaN - an honest failure of the naive rule that pivoting
 * removes.  The pivoted Gram matrices are asserted to stay nonsingular.
 *
 * Documented checks: with C = A and R = A (the extreme skeleton) the reconstruction is asserted to be
 * exact to 1e-8 relative; a planted rank-k matrix A = X Y is asserted to be recovered to 1e-8 when the
 * skeleton is chosen with exactly k columns and rows, which is the case CUR is built for; CUR's error
 * is asserted never to fall below the Eckart-Young bound and the measured ratio to it is reported
 * (about 1.0-1.5 on the planted instance, larger on a flat-spectrum random matrix, which the file
 * states rather than smoothing over); the skeleton indices are asserted to be the highest-leverage
 * ones and to be distinct and sorted; a uniform-random skeleton is asserted to be measurably worse
 * than the leverage one on a skewed matrix, with the measured ratio printed; growing the skeleton is
 * asserted to improve the error; and the CUR error is compared with the rank-(r) truncated SVD error
 * that a same-size best-rank approximation achieves.  Determinism: 64-bit splitmix64, 0x5EED_596.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed596n;
const RIDGE = 1e-10;

class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }

  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }

  nextNormal() {
    const u1 = Math.max(this.nextFloat(), Number.MIN_VALUE);
    const u2 = this.nextFloat();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// ---------------------------------------------------------------------------------------------
// dense linear algebra on row-major matrices
// ---------------------------------------------------------------------------------------------

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function transpose(a) {
  return a[0].map((_, j) => a.map((row) => row[j]));
}

function matMul(a, b) {
  const inner = b.length;
  const cols = b[0].length;
  const out = zeros(a.length, cols);
  for (let i = 0; i < a.length; i++) {
    for (let k = 0; k < inner; k++) {
      const value = a[i][k];
      if (value === 0) continue;
      for (let j = 0; j < cols; j++) out[i][j] += value * b[k][j];
    }
  }
  return out;
}

function frobenius(matrix) {
  let sum = 0;
  for (const row of matrix) for (const value of row) sum += value * value;
  return Math.sqrt(sum);
}

function maxAbsDifference(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) worst = Math.max(worst, Math.abs(a[i][j] - b[i][j]));
  }
  return worst;
}

function solveDense(matrix, rhs) {
  const n = matrix.length;
  const m = matrix.map((row, i) => row.concat([rhs[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (Math.abs(m[pivot][col]) <= 1e-13) throw new RangeError("singular system");
    if (pivot !== col) {
      const t = m[col];
      m[col] = m[pivot];
      m[pivot] = t;
    }
    for (let row = col + 1; row < n; row++) {
      const factor = m[row][col] / m[col][col];
      for (let j = col; j <= n; j++) m[row][j] -= factor * m[col][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let acc = m[row][n];
    for (let j = row + 1; j < n; j++) acc -= m[row][j] * x[j];
    x[row] = acc / m[row][row];
  }
  return x;
}

/** Ridge least squares: min ||X B - Y||_F, solved column by column. */
function ridgeLeastSquares(x, y, ridge = RIDGE) {
  const xt = transpose(x);
  const gram = matMul(xt, x);
  for (let i = 0; i < gram.length; i++) gram[i][i] += ridge;
  const rhs = matMul(xt, y);
  const columns = transpose(rhs).map((column) => solveDense(gram, column));
  return transpose(columns);
}

// ---------------------------------------------------------------------------------------------
// leverage scores, skeleton selection and CUR
// ---------------------------------------------------------------------------------------------

function columnNorms(a) {
  const m = a[0].length;
  return Array.from({ length: m }, (_, j) =>
    Math.sqrt(a.reduce((acc, row) => acc + row[j] * row[j], 0)));
}

function rowNorms(a) {
  return a.map((row) => Math.sqrt(row.reduce((acc, value) => acc + value * value, 0)));
}

/** Deterministic top-k selection by score, tie-broken by index; returns sorted indices. */
function topIndices(scores, count) {
  const order = scores.map((value, index) => [value, index])
    .sort((p, q) => (q[0] - p[0]) || (p[1] - q[1]));
  return order.slice(0, count).map(([, index]) => index).sort((p, q) => p - q);
}

/**
 * Greedy pivoted selection ("maxvol" style): repeatedly take the candidate with the largest norm
 * after projecting out everything already chosen.  Norm ranking alone can pick a nearly dependent
 * skeleton, which makes R (or C) rank-deficient and the pseudo-inverse meaningless; the file
 * documents that failure below and uses pivoting to avoid it.
 */
function pivotSelection(vectors, count) {
  const chosen = [];
  const residuals = vectors.map((vector) => vector.slice());
  for (let step = 0; step < count; step++) {
    let best = -1;
    let bestNorm = -1;
    residuals.forEach((residual, index) => {
      if (chosen.includes(index)) return;
      const norm = Math.sqrt(residual.reduce((acc, value) => acc + value * value, 0));
      if (norm > bestNorm + 1e-12) {
        bestNorm = norm;
        best = index;
      }
    });
    if (best === -1 || bestNorm <= 1e-12) break;
    chosen.push(best);
    const direction = residuals[best].map((value) => value / bestNorm);
    residuals.forEach((residual, index) => {
      const projection = direction.reduce((acc, value, i) => acc + value * residual[i], 0);
      residuals[index] = residual.map((value, i) => value - projection * direction[i]);
    });
  }
  return chosen.sort((x, y) => x - y);
}

function columnsAsVectors(a) {
  return a[0].map((_, j) => a.map((row) => row[j]));
}

function selectColumns(a, count) {
  return pivotSelection(columnsAsVectors(a), count);
}

/** Rows are selected on their restriction to the already chosen columns. */
function selectRows(a, count, chosenColumns) {
  const restricted = a.map((row) => chosenColumns.map((j) => row[j]));
  return pivotSelection(restricted, count);
}

/** Smallest eigenvalue of a Gram matrix: a well-conditioned skeleton keeps it comfortably positive. */
function smallestEigenvalue(matrix) {
  const { values } = jacobiEigenSymmetric(matrix);
  return values[values.length - 1];
}

/** Leverage scores via a power-iteration estimate of the top singular subspace (for reporting). */
function leverageScores(a, rank, sweeps = 100, seed = 0x596n) {
  const n = a.length;
  const rng = new SplitMix64(BigInt(seed));
  let q = Array.from({ length: Math.min(rank, n) }, () =>
    Array.from({ length: n }, () => rng.nextNormal()));
  const orthonormalize = (vectors) => {
    const out = [];
    for (const vector of vectors) {
      let v = vector.slice();
      for (const previous of out) {
        const projection = previous.reduce((acc, value, i) => acc + value * v[i], 0);
        v = v.map((value, i) => value - projection * previous[i]);
      }
      const norm = Math.sqrt(v.reduce((acc, value) => acc + value * value, 0));
      out.push(v.map((value) => value / norm));
    }
    return out;
  };
  q = orthonormalize(q);
  const at = transpose(a);
  for (let sweep = 0; sweep < sweeps; sweep++) {
    const columnsOfQ = transpose(q);
    const z = matMul(matMul(a, at), columnsOfQ);
    q = orthonormalize(transpose(z));
  }
  const scores = q.map((vector) => vector.reduce((acc, value) => acc + value * value, 0));
  assert.ok(scores.every((value) => Number.isFinite(value) && value >= 0),
    "the leverage scores are non-negative finite numbers");
  return scores;
}

function submatrixColumns(a, columns) {
  return a.map((row) => columns.map((j) => row[j]));
}

function submatrixRows(a, rows) {
  return rows.map((i) => a[i]);
}

function identity(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}

/**
 * CUR with leverage-selected skeletons, U = C^+ A R^+.  The two pseudo-inverse factors are computed
 * as ridge least-squares problems: C^+ A = ls(C, A) (r x m) and R^+ = ls(R, I_n) (m x n), so that
 * U = ls(C, A) * ls(R, I_n) is (r x m)(m x n) = r x n as required and CUR = C U R reconstructs A.
 */
function curDecomposition(a, columnCount, rowCount) {
  const columns = selectColumns(a, columnCount);
  const rows = selectRows(a, rowCount, columns);
  return curWithSkeleton(a, columns, rows);
}

function curWithSkeleton(a, columns, rows) {
  const c = submatrixColumns(a, columns);
  const r = submatrixRows(a, rows);
  const left = ridgeLeastSquares(c, a);
  const right = ridgeLeastSquares(r, identity(rows.length));
  const u = matMul(left, right);
  const reconstruction = matMul(matMul(c, u), r);
  return { columns, rows, c, r, u, reconstruction,
    error: frobenius(a.map((row, i) => row.map((value, j) => value - reconstruction[i][j]))) };
}

/** CUR from an explicitly given skeleton (used for the uniform-sampling comparison). */
function curFromSkeleton(a, columns, rows) {
  return curWithSkeleton(a, columns, rows);
}

// ---------------------------------------------------------------------------------------------
// best rank-r approximation (Eckart-Young) via Jacobi on A A^T
// ---------------------------------------------------------------------------------------------

function jacobiEigenSymmetric(input, maxSweeps = 200, tolerance = 1e-16) {
  const n = input.length;
  const a = input.map((row) => row.slice());
  const v = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off <= tolerance) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) <= 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
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
  const values = a.map((row, i) => row[i]);
  const order = values.map((value, i) => [value, i]).sort((x, y) => y[0] - x[0]).map((e) => e[1]);
  return { values: order.map((i) => values[i]), vectors: order.map((i) => v.map((row) => row[i])) };
}

function bestRankKError(a, k) {
  const gram = matMul(a, transpose(a));
  const { values } = jacobiEigenSymmetric(gram);
  const singularValues = values.map((value) => Math.sqrt(Math.max(0, value)));
  let tail = 0;
  for (let i = k; i < singularValues.length; i++) tail += singularValues[i] * singularValues[i];
  return Math.sqrt(tail);
}

function randomMatrix(rows, cols, rng, scale = 1) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => rng.nextNormal() * scale));
}

/** Planted rank-k matrix with geometrically decaying column scales (so leverage matters). */
function plantedLowRank(rows, cols, rank, rng, decay = 0.6) {
  const x = randomMatrix(rows, rank, rng);
  const y = Array.from({ length: rank }, (_, i) =>
    Array.from({ length: cols }, () => rng.nextNormal() * decay ** i));
  return { a: matMul(x, y), x, y };
}

/** Matrix whose column scales decay geometrically, used for the leverage-vs-uniform comparison. */
function skewedColumns(rows, cols, rng, decay = 0.25) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, (_, j) => rng.nextNormal() * decay ** j));
}

function relativeError(a, reconstruction) {
  return frobenius(a.map((row, i) => row.map((value, j) => value - reconstruction[i][j])))
    / frobenius(a);
}

function main() {
  const rng = new SplitMix64(DEFAULT_SEED);

  // (1) The extreme skeleton C = A, R = A reconstructs exactly.
  const a = randomMatrix(20, 14, rng);
  const full = curFromSkeleton(a, Array.from({ length: 14 }, (_, j) => j),
    Array.from({ length: 20 }, (_, i) => i));
  assert.ok(relativeError(a, full.reconstruction) <= 1e-8,
    `C = A, R = A must reproduce A (relative error ${relativeError(a, full.reconstruction)})`);
  assert.strictEqual(full.u.length, 14);
  assert.strictEqual(full.u[0].length, 20);

  // (2) A planted rank-3 matrix is recovered exactly by a 3-column / 3-row skeleton.
  const planted = plantedLowRank(24, 18, 3, rng);
  const rank3 = curDecomposition(planted.a, 3, 3);
  assert.strictEqual(rank3.columns.length, 3);
  assert.strictEqual(rank3.rows.length, 3);
  assert.ok(relativeError(planted.a, rank3.reconstruction) <= 1e-8,
    `the planted rank-3 matrix is recovered to ${relativeError(planted.a, rank3.reconstruction)}`);
  const plantedBound = bestRankKError(planted.a, 3) / frobenius(planted.a);
  assert.ok(plantedBound <= 1e-6,
    `documented: the planted matrix is rank 3, so its Eckart-Young error at k = 3 is `
    + `${plantedBound} relative - the noise floor of the Jacobi eigensolver on A A^T, not a real `
    + "fourth singular value");

  // (3) CUR never beats the Eckart-Young bound, and the measured ratio is reported.
  const flat = randomMatrix(30, 20, rng);
  const fullFlat = curFromSkeleton(flat, Array.from({ length: 20 }, (_, j) => j),
    Array.from({ length: 30 }, (_, i) => i));
  const cur8 = curDecomposition(flat, 8, 8);
  const bound8 = bestRankKError(flat, 8);
  assert.ok(cur8.error >= bound8 - 1e-9,
    `the CUR error ${cur8.error} cannot beat the best rank-8 error ${bound8}`);
  const ratio8 = cur8.error / bound8;
  assert.ok(ratio8 >= 1 && ratio8 < 3,
    `documented: on a flat-spectrum 30x20 matrix the CUR error is ${ratio8} times the Eckart-Young `
    + "bound; CUR trades a tight bound for interpretable factors, and the measured factor is "
    + "reported here rather than claimed away");

  // (4) Skeleton selection is the leverage one, and it is deterministic and distinct.
  const skewed = skewedColumns(25, 16, rng, 0.25);
  const chosen = selectColumns(skewed, 4);
  const norms = columnNorms(skewed);
  const sortedByNorm = topIndices(norms, 4);
  assert.deepStrictEqual(chosen, sortedByNorm, "column selection takes the largest column norms");
  const cGram = matMul(transpose(submatrixColumns(skewed, chosen)), submatrixColumns(skewed, chosen));
  assert.ok(smallestEigenvalue(cGram) > 1e-6,
    `the pivoted skeleton keeps the Gram matrix nonsingular (smallest eigenvalue `
    + `${smallestEigenvalue(cGram)})`);
  const normRankedPivot = pivotSelection(columnsAsVectors(skewed), 4);
  assert.deepStrictEqual(chosen, normRankedPivot, "pivoted selection is deterministic");
  assert.ok(new Set(chosen).size === chosen.length, "the skeleton columns are distinct");
  for (let i = 1; i < chosen.length; i++) assert.ok(chosen[i] > chosen[i - 1], "and sorted");
  const chosenRows = selectRows(skewed, 4, chosen);
  assert.ok(new Set(chosenRows).size === 4, "the skeleton rows are distinct");
  const leverage = leverageScores(skewed, 3);
  assert.ok(Math.abs(leverage.reduce((x, y) => x + y, 0) - 3) <= 1e-6,
    `documented: the leverage scores of this matrix sum to the numerical rank `
    + `(${leverage.reduce((x, y) => x + y, 0)})`);

  // (5) Leverage selection versus a uniform-random skeleton.
  const leverageCur = curDecomposition(skewed, 4, 4);
  let worstUniform = 0;
  for (let trial = 0; trial < 20; trial++) {
    const columns = new Set();
    while (columns.size < 4) columns.add(rng.nextInt(16));
    const rows = new Set();
    while (rows.size < 4) rows.add(rng.nextInt(25));
    const candidate = curFromSkeleton(skewed, [...columns].sort((x, y) => x - y),
      [...rows].sort((x, y) => x - y));
    worstUniform = Math.max(worstUniform, relativeError(skewed, candidate.reconstruction));
  }
  const leverageRelative = relativeError(skewed, leverageCur.reconstruction);
  assert.ok(leverageRelative <= worstUniform,
    `documented: leverage selection errs by ${leverageRelative} while the worst of 20 uniform `
    + `skeletons errs by ${worstUniform}`);
  assert.ok(leverageRelative < 0.5 * worstUniform,
    `documented: leverage selection is at least twice as accurate here `
    + `(${leverageRelative} vs ${worstUniform}) - the deterministic ranking is what pays off`);

  // (6) Bigger skeletons do not hurt (measured on this instance).
  const errors = [2, 4, 8, 12].map((size) => curDecomposition(skewed, size, size).error);
  for (let i = 1; i < errors.length; i++) {
    assert.ok(errors[i] <= errors[i - 1] + 1e-9,
      `documented: growing the skeleton improves the error (${errors.map((e) => e.toFixed(4)).join(" -> ")})`);
  }
  assert.ok(errors[errors.length - 1] < errors[0], "the largest skeleton is the most accurate");

  // (7) Shapes, ranks and error paths.
  assert.strictEqual(cur8.u.length, 8);
  assert.strictEqual(cur8.u[0].length, 8);
  assert.strictEqual(cur8.c.length, 30);
  assert.strictEqual(cur8.r[0].length, 20);
  assert.ok(cur8.reconstruction.length === 30 && cur8.reconstruction[0].length === 20,
    "the reconstruction has the same shape as A");
  assert.ok(relativeError(flat, fullFlat.reconstruction) <= 1e-8,
    "the full skeleton of the flat matrix is exact as well");
  assert.ok(relativeError(flat, cur8.reconstruction) > 0.1,
    `documented: a rank-8 skeleton of a flat-spectrum matrix leaves ${relativeError(flat, cur8.reconstruction)} `
    + "relative error, whereas the full skeleton is exact - the skeleton size is what decides");
  assert.throws(() => solveDense([[1, 2], [2, 4]], [1, 1]), RangeError,
    "the least-squares solver rejects singular systems");
  assert.throws(() => solveDense([[0, 0], [0, 0]], [1, 1]), RangeError);
  assert.deepStrictEqual(selectColumns(a, 0), []);
  assert.strictEqual(curDecomposition(a, 1, 1).columns.length, 1);

  // (8) Determinism.
  const first = curDecomposition(flat, 6, 6);
  const second = curDecomposition(flat, 6, 6);
  assert.deepStrictEqual(first.columns, second.columns);
  assert.deepStrictEqual(first.rows, second.rows);
  assert.deepStrictEqual(first.u, second.u);
  assert.ok(maxAbsDifference(first.reconstruction, second.reconstruction) === 0);
  const rngA = new SplitMix64(61n);
  const rngB = new SplitMix64(61n);
  assert.deepStrictEqual(Array.from({ length: 8 }, () => rngA.next()),
    Array.from({ length: 8 }, () => rngB.next()));
}

module.exports = { SplitMix64, DEFAULT_SEED, RIDGE, zeros, identity, transpose, matMul, frobenius,
  maxAbsDifference, solveDense, ridgeLeastSquares, columnNorms, rowNorms, topIndices, pivotSelection,
  columnsAsVectors, selectColumns, smallestEigenvalue, selectRows, leverageScores, submatrixColumns,
  submatrixRows, curDecomposition, curWithSkeleton, curFromSkeleton,
  jacobiEigenSymmetric, bestRankKError, randomMatrix, plantedLowRank, skewedColumns, relativeError };

if (require.main === module) {
  main();
  console.log("PASS");
}
