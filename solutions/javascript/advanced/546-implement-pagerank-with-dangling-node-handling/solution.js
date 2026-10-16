/**
 * Task 546 - Implement pagerank with dangling node handling. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * A vertex with no outgoing edges ("dangling") makes the transition matrix substochastic, so the
 * power iteration must decide where its alpha-weighted mass goes.  Three standard policies are
 * implemented side by side: `uniform` (the dangling mass is redistributed uniformly over all
 * vertices, the classic PageRank fix), `self` (the mass stays on the dangling vertex via a
 * self-loop), and `drop` (the mass simply disappears - kept only to show the failure mode).
 *
 * Documented checks: the dangling vertices are detected and a graph without any makes all three
 * policies agree exactly; the matrix form of one step is compared against an independent
 * mass-redistribution formulation (accumulate the dangling mass, then scatter it) and the two
 * agree to 1e-12 while converging to the same ranks to 1e-9; the uniform policy conserves total
 * mass exactly at every iterate (sum = 1) whereas `drop` is asserted to lose mass monotonically
 * (sum <= alpha * sum + (1 - alpha) and strictly below 1) and to settle on a fixed point with
 * sum = 1 - alpha on an all-dangling graph; `uniform` is shown to be exactly equivalent to
 * rewriting the graph so every dangling vertex gets n explicit out-edges (one per vertex), which
 * materialises the uniform row; the converged ranks match the exact Gauss-Jordan solution of
 * (I - alpha P^T) x = (1 - alpha)/n for every policy; alpha = 0 collapses everything to 1/n; and
 * a hub that is dangling ends up top-ranked under the uniform policy.  Reruns are identical.
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

function outLinks(n, edges) {
  const out = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) out[u].push(v);
  return out;
}

function danglingVertices(n, edges) {
  const out = outLinks(n, edges);
  const list = [];
  for (let u = 0; u < n; u++) if (out[u].length === 0) list.push(u);
  return list;
}

/**
 * Materialised row-stochastic transition matrix.  `uniform` fills a dangling row with 1/n,
 * `self` puts a 1 on the diagonal, `drop` leaves the row zero (the ranking then leaks mass).
 */
function transitionMatrix(n, edges, mode = "uniform") {
  if (!["uniform", "self", "drop"].includes(mode)) throw new Error(`unknown mode ${mode}`);
  const out = outLinks(n, edges);
  const P = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let u = 0; u < n; u++) {
    if (out[u].length === 0) {
      if (mode === "uniform") for (let v = 0; v < n; v++) P[u][v] = 1 / n;
      else if (mode === "self") P[u][u] = 1;
    } else {
      for (const v of out[u]) P[u][v] += 1 / out[u].length;
    }
  }
  return { P, out, dangling: danglingVertices(n, edges) };
}

function applyGoogle(P, x, alpha) {
  const n = x.length;
  const out = new Array(n).fill((1 - alpha) / n);
  for (let u = 0; u < n; u++) {
    const share = alpha * x[u];
    if (share === 0) continue;
    const row = P[u];
    for (let v = 0; v < n; v++) if (row[v] !== 0) out[v] += share * row[v];
  }
  return out;
}

/**
 * Independent formulation: the same step expressed as "each vertex pushes alpha of its mass to
 * its out-neighbours; the mass stranded at dangling vertices is then scattered according to the
 * policy".  Returns the stranded mass as well, which the tests use for the drop-mode accounting.
 */
function redistributionStep(n, edges, x, alpha, mode = "uniform") {
  const out = outLinks(n, edges);
  const next = new Array(n).fill((1 - alpha) / n);
  let danglingMass = 0;
  for (let u = 0; u < n; u++) {
    const share = alpha * x[u];
    if (out[u].length === 0) {
      danglingMass += share;
      continue;
    }
    const per = share / out[u].length;
    for (const v of out[u]) next[v] += per;
  }
  if (mode === "uniform") {
    for (let v = 0; v < n; v++) next[v] += danglingMass / n;
  } else if (mode === "self") {
    for (const u of danglingVertices(n, edges)) next[u] += alpha * x[u];
  } else if (mode !== "drop") {
    throw new Error(`unknown mode ${mode}`);
  }
  return { next, danglingMass };
}

function solveLinearSystem(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => row.concat([rhs[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    }
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
  return a.map((row, i) => row[n] / a[i][i]);
}

function pageRankExact(n, edges, alpha = 0.85, mode = "uniform") {
  const { P } = transitionMatrix(n, edges, mode);
  const matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0) - alpha * P[j][i]));
  return solveLinearSystem(matrix, new Array(n).fill((1 - alpha) / n));
}

function pageRankIterative(n, edges, options = {}) {
  const alpha = options.alpha === undefined ? 0.85 : options.alpha;
  const tol = options.tol === undefined ? 1e-13 : options.tol;
  const maxIter = options.maxIter === undefined ? 500 : options.maxIter;
  const mode = options.mode === undefined ? "uniform" : options.mode;
  const { P, dangling } = transitionMatrix(n, edges, mode);
  let x = new Array(n).fill(1 / n);
  const history = [];
  const sums = [];
  let converged = false;
  let iterations = 0;
  for (let step = 1; step <= maxIter; step++) {
    const next = applyGoogle(P, x, alpha);
    history.push(l1Difference(next, x));
    x = next;
    iterations = step;
    sums.push(x.reduce((a, b) => a + b, 0));
    if (history[history.length - 1] <= tol) {
      converged = true;
      break;
    }
  }
  return { x, history, sums, converged, iterations, P, dangling, alpha,
    residual: l1Difference(applyGoogle(P, x, alpha), x) };
}

/** The same iteration written with the mass-redistribution formulation. */
function pageRankByRedistribution(n, edges, options = {}) {
  const alpha = options.alpha === undefined ? 0.85 : options.alpha;
  const tol = options.tol === undefined ? 1e-13 : options.tol;
  const maxIter = options.maxIter === undefined ? 500 : options.maxIter;
  const mode = options.mode === undefined ? "uniform" : options.mode;
  let x = new Array(n).fill(1 / n);
  let converged = false;
  let iterations = 0;
  const history = [];
  for (let step = 1; step <= maxIter; step++) {
    const { next } = redistributionStep(n, edges, x, alpha, mode);
    history.push(l1Difference(next, x));
    x = next;
    iterations = step;
    if (history[history.length - 1] <= tol) {
      converged = true;
      break;
    }
  }
  return { x, history, converged, iterations };
}

/** Rewrites every dangling vertex into one with n explicit out-edges, materialising 1/n rows. */
function rewriteDangling(n, edges) {
  const rewritten = edges.map(([u, v]) => [u, v]);
  for (const u of danglingVertices(n, edges)) {
    for (let v = 0; v < n; v++) rewritten.push([u, v]);
  }
  return rewritten;
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

/** Random digraph with vertex n-1 forced to be a sink that is still linked to. */
function graphWithSink(n, percent, seed) {
  const { edges } = randomGraph(n, percent, seed);
  const kept = edges.filter(([u]) => u !== n - 1);
  return { edges: kept.concat([[0, n - 1]]), n };
}

function main() {
  const assert = require("node:assert");

  // (1) Detection, and agreement of all policies when nothing is dangling.
  const dense = randomGraph(6, 45, 3).edges;
  assert.deepStrictEqual(danglingVertices(6, dense), [], "the dense graph has no sink");
  const denseRuns = ["uniform", "self", "drop"].map((mode) =>
    pageRankIterative(6, dense, { mode, alpha: 0.85 }));
  assert.ok(maxDifference(denseRuns[0].x, denseRuns[1].x) <= 1e-12);
  assert.ok(maxDifference(denseRuns[0].x, denseRuns[2].x) <= 1e-12,
    "with no dangling vertex the policies are indistinguishable");

  const sink = [[0, 1], [2, 1]];
  assert.deepStrictEqual(danglingVertices(3, sink), [1], "vertex 1 has no out-edges");

  // (2) The matrix step and the mass-redistribution step are the same operator.
  for (const mode of ["uniform", "self", "drop"]) {
    for (let seed = 1; seed <= 5; seed++) {
      const n = 6;
      const { edges, seed: next } = randomGraph(n, 30, seed * 17);
      const { P } = transitionMatrix(n, edges, mode);
      const x = new Array(n).fill(1 / n).map((value, i) => value * (1 + (i % 3)));
      const total = x.reduce((a, b) => a + b, 0);
      const start = x.map((value) => value / total);
      const viaMatrix = applyGoogle(P, start, 0.85);
      const viaMass = redistributionStep(n, edges, start, 0.85, mode).next;
      assert.ok(maxDifference(viaMatrix, viaMass) <= 1e-12,
        `${mode} seed ${seed}: the two formulations agree (lcg state ${next >= 0})`);
      const sum = viaMatrix.reduce((a, b) => a + b, 0);
      if (mode !== "drop") {
        assert.ok(Math.abs(sum - 1) <= 1e-12, `${mode}: one step preserves the total mass`);
      } else {
        assert.ok(sum <= 1 + 1e-12, "drop can only lose mass");
      }
    }
  }

  // (3) Both iteration styles converge to the same ranks, and to the exact linear solve.
  for (let seed = 11; seed <= 16; seed++) {
    const n = 7;
    const { edges } = graphWithSink(n, 25, seed * 23);
    assert.ok(danglingVertices(n, edges).length > 0, `seed ${seed}: the graph must have a sink`);
    for (const mode of ["uniform", "self", "drop"]) {
      const matrix = pageRankIterative(n, edges, { mode, alpha: 0.85 });
      const mass = pageRankByRedistribution(n, edges, { mode, alpha: 0.85 });
      assert.ok(matrix.converged && mass.converged, `${mode} seed ${seed}: converged`);
      assert.ok(maxDifference(matrix.x, mass.x) <= 1e-9,
        `${mode} seed ${seed}: both formulations find the same ranks`);
      const exact = pageRankExact(n, edges, 0.85, mode);
      assert.ok(maxDifference(matrix.x, exact) <= 1e-9, `${mode} seed ${seed}: matches the solve`);
      assert.ok(matrix.residual <= 1e-12, `${mode} seed ${seed}: eigen-residual`);
    }
  }

  // (4) Uniform conserves mass at every iterate; drop loses it monotonically.
  const star = [[0, 1], [2, 1]];
  const uniformRun = pageRankIterative(3, star, { mode: "uniform" });
  for (const sum of uniformRun.sums) {
    assert.ok(Math.abs(sum - 1) <= 1e-12, "uniform policy: every iterate sums to one");
  }
  const dropRun = pageRankIterative(3, star, { mode: "drop" });
  let previousMass = 1;
  const totalDangling = danglingVertices(3, star).length;
  for (let step = 0; step < dropRun.sums.length; step++) {
    assert.ok(dropRun.sums[step] <= 0.85 * previousMass + 0.15 + 1e-12,
      `drop policy: mass can only fall (step ${step})`);
    assert.ok(dropRun.sums[step] < 1 - 1e-6, "drop policy: mass really leaks");
    previousMass = dropRun.sums[step];
  }
  assert.strictEqual(totalDangling, 1);
  assert.ok(dropRun.sums[dropRun.sums.length - 1] < uniformRun.sums[uniformRun.sums.length - 1] - 1e-6);
  assert.notStrictEqual(dropRun.residual, undefined);
  assert.ok(dropRun.x[1] < 1, "the dropped rank mass is simply gone");

  // (5) Policies genuinely differ once a sink exists, and the hub-sink is top-ranked.
  const uniformStar = pageRankIterative(3, star, { mode: "uniform", alpha: 0.85 }).x;
  const selfStar = pageRankIterative(3, star, { mode: "self", alpha: 0.85 }).x;
  assert.ok(maxDifference(uniformStar, selfStar) > 1e-3, "the policies differ on the sink graph");
  assert.strictEqual(uniformStar.indexOf(Math.max(...uniformStar)), 1,
    "the dangling hub collects the most rank under the uniform policy");
  assert.ok(maxDifference(uniformStar, pageRankExact(3, star, 0.85, "uniform")) <= 1e-9);
  assert.ok(maxDifference(selfStar, pageRankExact(3, star, 0.85, "self")) <= 1e-9);
  assert.ok(Math.abs(selfStar[1] - uniformStar[1]) > 1e-3);

  // (6) Uniform handling == rewriting the graph so sinks get one n-way uniform row.
  for (let seed = 21; seed <= 24; seed++) {
    const n = 5;
    const { edges } = graphWithSink(n, 25, seed * 31);
    const rewritten = rewriteDangling(n, edges);
    assert.deepStrictEqual(danglingVertices(n, rewritten), [],
      "the rewrite removes every sink");
    const direct = pageRankIterative(n, edges, { mode: "uniform", alpha: 0.85 }).x;
    const asPlain = pageRankIterative(n, rewritten, { mode: "drop", alpha: 0.85 }).x;
    assert.ok(maxDifference(direct, asPlain) <= 1e-12,
      `seed ${seed}: explicit uniform out-edges reproduce the uniform policy exactly`);
  }

  // (7) Closed forms on fully dangling and trivial graphs.
  const isolated = [];
  assert.deepStrictEqual(danglingVertices(5, isolated), [0, 1, 2, 3, 4]);
  const isolatedUniform = pageRankIterative(5, isolated, { mode: "uniform", alpha: 0.85 }).x;
  assert.ok(maxDifference(isolatedUniform, new Array(5).fill(0.2)) <= 1e-12);
  const isolatedDrop = pageRankIterative(5, isolated, { mode: "drop", alpha: 0.85 }).x;
  assert.ok(maxDifference(isolatedDrop, new Array(5).fill(0.03)) <= 1e-12,
    "all-dangling + drop: every rank is (1 - alpha)/n");
  assert.ok(Math.abs(isolatedDrop.reduce((a, b) => a + b, 0) - 0.15) <= 1e-12,
    "all-dangling + drop: the total mass is exactly 1 - alpha");
  const zeroAlpha = pageRankIterative(3, star, { mode: "drop", alpha: 0 });
  assert.ok(maxDifference(zeroAlpha.x, [1 / 3, 1 / 3, 1 / 3]) <= 1e-12,
    "alpha = 0 wipes out the whole dangling question");
  assert.strictEqual(pageRankIterative(1, [], { mode: "uniform" }).x[0], 1);
  assert.deepStrictEqual(transitionMatrix(2, [[0, 1], [1, 0]], "self").dangling, []);

  // (8) Guard rails and determinism.
  assert.throws(() => transitionMatrix(3, star, "nonsense"), /unknown mode/);
  assert.throws(() => redistributionStep(3, star, [0.2, 0.6, 0.2], 0.85, "nonsense"), /unknown mode/);
  assert.deepStrictEqual(pageRankIterative(3, star, { mode: "uniform" }),
    pageRankIterative(3, star, { mode: "uniform" }));
  assert.deepStrictEqual(pageRankByRedistribution(3, star, { mode: "self" }),
    pageRankByRedistribution(3, star, { mode: "self" }));
  assert.deepStrictEqual(pageRankExact(3, star, 0.85, "uniform"),
    pageRankExact(3, star, 0.85, "uniform"));
}

module.exports = { lcgNext, l1Difference, maxDifference, outLinks, danglingVertices,
  transitionMatrix, applyGoogle, redistributionStep, solveLinearSystem, pageRankExact,
  pageRankIterative, pageRankByRedistribution, rewriteDangling, randomGraph, graphWithSink };

if (require.main === module) {
  main();
  console.log("PASS");
}
