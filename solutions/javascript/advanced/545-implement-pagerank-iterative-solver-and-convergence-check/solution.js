/**
 * Task 545 - Implement pagerank iterative solver and convergence check. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * PageRank by the damped power method: with the row-stochastic transition matrix P built from the
 * edge list, the Google matrix is G = alpha * P + (1 - alpha)/n * J and the iteration is
 * x_{k+1} = G^T x_k, i.e. each vertex hands alpha of its mass to its out-neighbours and the
 * remaining (1 - alpha) is spread uniformly by the teleport term.  The solver records the L1
 * difference ||x_{k+1} - x_k||_1 per step, stops when it drops below a documented tolerance, and
 * reports whether it converged inside the iteration cap.  All test graphs are simple digraphs
 * without dangling vertices (out-degree >= 1 everywhere), so the transition matrix is stochastic
 * on its own and the dangling question is left to task 546; the self-test asserts that property.
 *
 * Documented checks: the iterate matches the exact solution of (I - alpha P^T) x = (1-alpha)/n * 1
 * computed by Gauss-Jordan elimination (<= 1e-9); the ranks sum to one and satisfy the eigenvalue
 * equation ||G^T x - x||_1 <= tolerance; the residual history is non-increasing and shrinks by at
 * least the factor alpha per step, which is the standard contraction argument
 * ||G^T(x - y)||_1 <= alpha ||x - y||_1 - asserted directly on random probability vectors; the
 * limit is independent of the starting vector (uniform, a point mass, and LCG vectors); alpha = 0
 * gives exactly the uniform distribution in a single step; a directed cycle is uniform; two
 * disjoint cycles each keep exactly their share of the total mass; and the iteration cap is shown
 * to bite by asking for one iteration only.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function l1Difference(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += Math.abs(a[i] - b[i]);
  return total;
}

function maxDifference(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

/**
 * Row-stochastic transition matrix.  `mode` only matters for out-degree-0 vertices; the graphs
 * used by this task never have any, which is asserted in the tests.
 */
function transitionMatrix(n, edges, mode = "uniform") {
  const out = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) out[u].push(v);
  const P = Array.from({ length: n }, () => new Array(n).fill(0));
  const dangling = [];
  for (let u = 0; u < n; u++) {
    if (out[u].length === 0) {
      dangling.push(u);
      if (mode === "uniform") {
        for (let v = 0; v < n; v++) P[u][v] = 1 / n;
      } else if (mode === "self") {
        P[u][u] = 1;
      } else if (mode !== "drop") {
        throw new Error(`unknown dangling mode ${mode}`);
      }
    } else {
      for (const v of out[u]) P[u][v] += 1 / out[u].length;
    }
  }
  return { P, out, dangling };
}

/** One application of the Google matrix transposed: x -> alpha P^T x + (1 - alpha)/n. */
function applyGoogle(P, x, alpha) {
  const n = x.length;
  const out = new Array(n).fill((1 - alpha) / n);
  for (let u = 0; u < n; u++) {
    const share = alpha * x[u];
    if (share === 0) continue;
    const row = P[u];
    for (let v = 0; v < n; v++) {
      if (row[v] !== 0) out[v] += share * row[v];
    }
  }
  return out;
}

/** Gauss-Jordan elimination with partial pivoting; exact enough for tiny systems. */
function solveLinearSystem(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => row.concat([rhs[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    }
    if (Math.abs(a[pivot][col]) < 1e-300) throw new Error("singular system");
    const swap = a[col];
    a[col] = a[pivot];
    a[pivot] = swap;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = a[r][col] / a[col][col];
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) a[r][c] -= factor * a[col][c];
    }
  }
  const x = new Array(n);
  for (let i = 0; i < n; i++) x[i] = a[i][n] / a[i][i];
  return x;
}

/** Exact PageRank: solve (I - alpha P^T) x = (1 - alpha)/n * 1. */
function pageRankExact(n, edges, alpha = 0.85, mode = "uniform") {
  const { P } = transitionMatrix(n, edges, mode);
  const matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0) - alpha * P[j][i]));
  const rhs = new Array(n).fill((1 - alpha) / n);
  return solveLinearSystem(matrix, rhs);
}

function pageRankIterative(n, edges, options = {}) {
  const alpha = options.alpha === undefined ? 0.85 : options.alpha;
  const tol = options.tol === undefined ? 1e-12 : options.tol;
  const maxIter = options.maxIter === undefined ? 500 : options.maxIter;
  const mode = options.mode === undefined ? "uniform" : options.mode;
  const { P, out, dangling } = transitionMatrix(n, edges, mode);
  let x = options.start ? options.start.slice() : new Array(n).fill(1 / n);
  const history = [];
  let converged = false;
  let iterations = 0;
  for (let step = 1; step <= maxIter; step++) {
    const next = applyGoogle(P, x, alpha);
    history.push(l1Difference(next, x));
    x = next;
    iterations = step;
    if (history[history.length - 1] <= tol) {
      converged = true;
      break;
    }
  }
  const residual = l1Difference(applyGoogle(P, x, alpha), x);
  return { x, history, converged, iterations, residual, P, out, dangling, alpha, tol };
}

function randomGraph(n, percent, seed) {
  const edges = [];
  let s = seed % 65537;
  for (let u = 0; u < n; u++) {
    for (let v = 0; v < n; v++) {
      if (u === v) continue;
      s = lcgNext(s);
      if (s % 100 < percent) edges.push([u, v]);
    }
  }
  return { edges, seed: s };
}

/** Digraph with guaranteed out-degree >= 1 everywhere. */
function noDanglingGraph(n, percent, seed) {
  const { edges } = randomGraph(n, percent, seed);
  const hasOut = new Array(n).fill(false);
  for (const [u] of edges) hasOut[u] = true;
  for (let u = 0; u < n; u++) {
    if (!hasOut[u]) edges.push([u, (u + 1) % n]);
  }
  return { edges, n };
}

function randomProbabilityVector(n, seed) {
  const values = [];
  let s = seed % 65537;
  for (let i = 0; i < n; i++) {
    s = lcgNext(s);
    values.push(s % 100 + 1);
  }
  const total = values.reduce((a, b) => a + b, 0);
  return values.map((value) => value / total);
}

function cycleGraph(n) {
  const edges = [];
  for (let i = 0; i < n; i++) edges.push([i, (i + 1) % n]);
  return edges;
}

function main() {
  const assert = require("node:assert");

  // (1) Closed forms: alpha = 0 is uniform at once, a directed cycle is uniform, 2-cycle is 1/2.
  const nothing = pageRankIterative(5, cycleGraph(5), { alpha: 0 });
  assert.deepStrictEqual(nothing.x, [0.2, 0.2, 0.2, 0.2, 0.2]);
  assert.strictEqual(nothing.iterations, 1, "alpha = 0 converges in one step");
  assert.ok(nothing.converged);
  const cycle = pageRankIterative(4, cycleGraph(4), { alpha: 0.85 });
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(cycle.x[i] - 0.25) <= 1e-12,
    "a directed cycle is perfectly symmetric");
  assert.strictEqual(cycle.iterations, 1);
  const twoCycle = pageRankIterative(2, [[0, 1], [1, 0]]);
  assert.ok(maxDifference(twoCycle.x, [0.5, 0.5]) <= 1e-12);

  // (2) Iteration == exact linear solve, ranks sum to one, eigen-residual within tolerance.
  for (let seed = 1; seed <= 8; seed++) {
    const n = 6 + (seed % 4);
    const { edges } = noDanglingGraph(n, 35 + 5 * (seed % 3), seed * 19);
    const run = pageRankIterative(n, edges, { alpha: 0.85 });
    assert.deepStrictEqual(run.dangling, [], `seed ${seed}: no dangling vertex by construction`);
    assert.ok(run.out.every((list) => list.length >= 1), `seed ${seed}: out-degree >= 1`);
    assert.ok(run.converged, `seed ${seed}: the power method must converge`);
    const sum = run.x.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) <= 1e-12, `seed ${seed}: ranks sum to one`);
    assert.ok(run.x.every((value) => value > 0 && value < 1), `seed ${seed}: strictly inside (0,1)`);
    assert.ok(run.residual <= 1e-12, `seed ${seed}: residual ${run.residual} <= tolerance`);
    const exact = pageRankExact(n, edges, 0.85);
    assert.ok(maxDifference(run.x, exact) <= 1e-9, `seed ${seed}: matches the linear solve`);
    const applied = applyGoogle(run.P, run.x, run.alpha);
    assert.ok(l1Difference(applied, run.x) <= 1e-12, `seed ${seed}: eigenvector of G^T`);
    assert.ok(run.iterations <= 200, `seed ${seed}: converges in a sane number of steps`);
  }

  // (3) Convergence is geometric with the damping factor: history non-increasing, ratio <= alpha.
  for (let seed = 11; seed <= 14; seed++) {
    const n = 8;
    const { edges } = noDanglingGraph(n, 30, seed * 29);
    const run = pageRankIterative(n, edges, { alpha: 0.85 });
    for (let k = 1; k < run.history.length; k++) {
      assert.ok(run.history[k] <= run.history[k - 1] + 1e-15,
        `seed ${seed}: residual must not grow at step ${k}`);
      if (run.history[k - 1] > 1e-9) {
        assert.ok(run.history[k] <= 0.85 * run.history[k - 1] + 1e-15,
          `seed ${seed}: step ${k} must shrink by at least alpha`);
      }
    }
  }

  // (4) Contraction of the Google matrix in the L1 norm, checked on random probability vectors.
  const contractionEdges = noDanglingGraph(7, 40, 4242).edges;
  const contractionP = transitionMatrix(7, contractionEdges).P;
  for (let seed = 1; seed <= 6; seed++) {
    const x = randomProbabilityVector(7, seed * 13);
    const y = randomProbabilityVector(7, seed * 37 + 5);
    const distance = l1Difference(x, y);
    const moved = l1Difference(applyGoogle(contractionP, x, 0.85),
      applyGoogle(contractionP, y, 0.85));
    assert.ok(moved <= 0.85 * distance + 1e-12,
      `seed ${seed}: ||G^T(x-y)||_1 <= alpha ||x-y||_1 (${moved} <= ${0.85 * distance})`);
  }

  // (5) The limit does not depend on the starting vector.
  const startEdges = noDanglingGraph(6, 45, 777).edges;
  const fromUniform = pageRankIterative(6, startEdges, { alpha: 0.85 }).x;
  const delta = new Array(6).fill(0);
  delta[2] = 1;
  const fromDelta = pageRankIterative(6, startEdges, { alpha: 0.85, start: delta }).x;
  const fromRandom = pageRankIterative(6, startEdges,
    { alpha: 0.85, start: randomProbabilityVector(6, 99) }).x;
  assert.ok(maxDifference(fromUniform, fromDelta) <= 1e-9);
  assert.ok(maxDifference(fromUniform, fromRandom) <= 1e-9);
  assert.ok(maxDifference(fromDelta, pageRankExact(6, startEdges, 0.85)) <= 1e-9);
  assert.ok(maxDifference(fromUniform, pageRankExact(6, startEdges, 0.85)) <= 1e-9);

  // (6) Mass is conserved per closed component: two disjoint cycles split 3 : 3.
  const twoComponents = [0, 1, 2].map((i) => [i, (i + 1) % 3])
    .concat([3, 4, 5].map((i) => [i, 3 + ((i + 1) % 3)]));
  const split = pageRankIterative(6, twoComponents, { alpha: 0.85 });
  const firstComponent = split.x.slice(0, 3).reduce((a, b) => a + b, 0);
  const secondComponent = split.x.slice(3).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(firstComponent - 0.5) <= 1e-12 && Math.abs(secondComponent - 0.5) <= 1e-12,
    "each closed component keeps exactly its share of the mass");

  // (7) The iteration cap really bites, and lower tolerances need more steps.
  const capped = pageRankIterative(9, noDanglingGraph(9, 35, 5).edges,
    { alpha: 0.85, maxIter: 1, tol: 1e-15 });
  assert.strictEqual(capped.converged, false);
  assert.strictEqual(capped.iterations, 1);
  assert.ok(capped.residual > 1e-15);
  const casual = pageRankIterative(9, noDanglingGraph(9, 35, 5).edges,
    { alpha: 0.85, tol: 1e-4 });
  const precise = pageRankIterative(9, noDanglingGraph(9, 35, 5).edges,
    { alpha: 0.85, tol: 1e-14 });
  assert.ok(casual.iterations < precise.iterations, "a tighter tolerance costs more iterations");
  assert.ok(maxDifference(casual.x, precise.x) <= 1e-3);

  // (8) Determinism.
  assert.deepStrictEqual(pageRankIterative(6, startEdges), pageRankIterative(6, startEdges));
  assert.deepStrictEqual(pageRankExact(6, startEdges), pageRankExact(6, startEdges));
}

module.exports = { lcgNext, l1Difference, maxDifference, transitionMatrix, applyGoogle,
  solveLinearSystem, pageRankExact, pageRankIterative, randomGraph, noDanglingGraph,
  randomProbabilityVector, cycleGraph };

if (require.main === module) {
  main();
  console.log("PASS");
}
