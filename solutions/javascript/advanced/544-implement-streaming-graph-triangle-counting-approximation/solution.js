/**
 * Task 544 - Implement streaming graph triangle counting approximation. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * One-pass streaming triangle counting over an edge stream.  The exact counter keeps only the
 * seen part of the graph: adjacency sets for the vertices touched so far, and when an edge (u, v)
 * arrives it adds |N(u) ∩ N(v)| (intersecting the smaller side into the larger one) to the running
 * total.  Every triangle is closed exactly once - by its last-arriving edge, whose two endpoints
 * then both already know the third vertex - so the running count after prefix i equals the exact
 * triangle count of that prefix, which the self-test verifies against a fresh O(n^3) brute force
 * for every single prefix.  A second, in-memory "forward" counter orients each edge from the
 * higher-degree endpoint to the lower one (ties by id) and sums |N+(u) ∩ N+(v)| over oriented
 * edges; it is asserted to agree with the brute force and to need fewer membership tests than the
 * unordered per-edge intersection on a dense graph.
 *
 * The approximation part is DOULION-style sparsification: every edge is kept independently with
 * probability p by a deterministic LCG, the exact streaming counter runs on the surviving stream,
 * and T̂ = (#sampled triangles) / p^3.  Because the exact counter is the identity for p = 1, the
 * test asserts equality there; asserts that the sampled count matches a second, independently
 * built sparsified graph; asserts the empirical mean over many fixed seeds is close to the true
 * count (unbiasedness) for p = 0.6 and 0.8; asserts the estimator variance shrinks as p grows;
 * and documents the heavy tail by showing p = 0.2 produces zero estimates while still averaging
 * in the right order of magnitude.  Reruns with the same seeds are bit-identical.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

/** Deterministic graph on n vertices; edges only from lower to higher index. */
function randomGraph(n, percent, seed) {
  const edges = [];
  let s = seed % 65537;
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      s = lcgNext(s);
      if (s % 100 < percent) edges.push([u, v]);
    }
  }
  return { edges, seed: s };
}

function completeGraph(n) {
  const edges = [];
  for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) edges.push([u, v]);
  return edges;
}

/** Ground truth: enumerate all triples (fine for n <= 10). */
function bruteTriangles(n, edges) {
  const adj = Array.from({ length: n }, () => new Array(n).fill(false));
  for (const [u, v] of edges) {
    adj[u][v] = true;
    adj[v][u] = true;
  }
  let count = 0;
  for (let a = 0; a < n; a++) {
    for (let b = a + 1; b < n; b++) {
      if (!adj[a][b]) continue;
      for (let c = b + 1; c < n; c++) {
        if (adj[a][c] && adj[b][c]) count += 1;
      }
    }
  }
  return count;
}

/**
 * Exact single-pass streaming counter.  `keep` (optional) decides per edge, by 1-based stream
 * position, whether the edge is part of the stream the counter sees - that is how sparsification
 * is layered on top without giving the counter any look-ahead.
 */
function streamingTriangleCount(n, edges, keep = null) {
  const seen = Array.from({ length: n }, () => new Set());
  const running = [];
  const stats = { intersections: 0, edgesStreamed: 0, edgesKept: 0 };
  let triangles = 0;
  edges.forEach(([u, v], index) => {
    stats.edgesStreamed += 1;
    if (keep !== null && !keep(u, v, index + 1)) {
      running.push(triangles);
      return;
    }
    stats.edgesKept += 1;
    const left = seen[u];
    const right = seen[v];
    const [small, large] = left.size <= right.size ? [left, right] : [right, left];
    let closed = 0;
    for (const w of small) {
      stats.intersections += 1;
      if (large.has(w)) closed += 1;
    }
    triangles += closed;
    seen[u].add(v);
    seen[v].add(u);
    running.push(triangles);
  });
  return { triangles, running, stats, seen };
}

/** In-memory heavy-light ("forward") counter: orient heavier -> lighter, then intersect. */
function orderedTriangleCount(n, edges) {
  const degree = new Array(n).fill(0);
  for (const [u, v] of edges) {
    degree[u] += 1;
    degree[v] += 1;
  }
  const rank = (vertex) => degree[vertex] * n + vertex;
  const forward = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) {
    if (rank(u) > rank(v)) forward[u].push(v);
    else forward[v].push(u);
  }
  for (const list of forward) list.sort((a, b) => a - b);
  const stats = { intersections: 0 };
  let triangles = 0;
  for (let u = 0; u < n; u++) {
    const other = new Set(forward[u]);
    for (const v of forward[u]) {
      const target = forward[v];
      for (const w of target) {
        stats.intersections += 1;
        if (other.has(w)) triangles += 1;
      }
    }
  }
  return { triangles, stats, forward };
}

/**
 * Same graph, but every edge intersecting its full adjacency list (no ordering, no size trick).
 * Each triangle is seen once per edge, so the raw total is three times the answer.
 */
function unorderedTriangleCount(n, edges) {
  const adj = Array.from({ length: n }, () => new Set());
  for (const [u, v] of edges) {
    adj[u].add(v);
    adj[v].add(u);
  }
  const stats = { intersections: 0 };
  let raw = 0;
  for (const [u, v] of edges) {
    for (const w of adj[u]) {
      stats.intersections += 1;
      if (w !== u && w !== v && adj[v].has(w)) raw += 1;
    }
  }
  return { triangles: raw / 3, stats };
}

/** Independent LCG sparsification of the stream; returns the kept edges and the per-edge flags. */
function sparsify(edges, p, seed) {
  const threshold = Math.round(p * 1000);
  let s = seed % 65537;
  const flags = [];
  const kept = [];
  for (const edge of edges) {
    s = lcgNext(s);
    const keep = (s % 1000) < threshold;
    flags.push(keep);
    if (keep) kept.push(edge);
  }
  return { flags, kept, seed: s };
}

/** DOULION estimate: run the streaming counter on the sampled stream and rescale by p^3. */
function sparsifiedEstimate(n, edges, p, seed) {
  const { flags, kept } = sparsify(edges, p, seed);
  const streamed = streamingTriangleCount(n, edges, (u, v, index) => flags[index - 1]);
  const direct = streamingTriangleCount(n, kept);
  return { count: streamed.triangles, estimate: streamed.triangles / p ** 3,
    directCount: direct.triangles, keptCount: kept.length, stats: streamed.stats };
}

function estimatorStats(n, edges, p, seeds) {
  const values = seeds.map((seed) => sparsifiedEstimate(n, edges, p, seed).estimate);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / values.length;
  return { values, mean, variance, stddev: Math.sqrt(variance) };
}

function seedRange(from, to) {
  const seeds = [];
  for (let seed = from; seed <= to; seed++) seeds.push(seed);
  return seeds;
}

function main() {
  const assert = require("node:assert");

  // (1) Hand-verified exact counts.
  assert.strictEqual(bruteTriangles(3, completeGraph(3)), 1);
  assert.strictEqual(streamingTriangleCount(3, completeGraph(3)).triangles, 1);
  assert.strictEqual(bruteTriangles(4, completeGraph(4)), 4);
  assert.strictEqual(streamingTriangleCount(4, completeGraph(4)).triangles, 4);
  assert.strictEqual(orderedTriangleCount(4, completeGraph(4)).triangles, 4);
  assert.strictEqual(streamingTriangleCount(5, completeGraph(5)).triangles, 10);
  assert.strictEqual(orderedTriangleCount(5, completeGraph(5)).triangles, 10);
  assert.strictEqual(streamingTriangleCount(4, [[0, 1], [1, 2], [2, 3], [3, 0]]).triangles, 0,
    "a 4-cycle has no triangle");
  assert.strictEqual(streamingTriangleCount(4, [[0, 1], [1, 2], [2, 3]]).triangles, 0, "a path");
  const k33 = [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]];
  assert.strictEqual(streamingTriangleCount(6, k33).triangles, 0, "K3,3 is triangle free");
  assert.strictEqual(streamingTriangleCount(1, []).triangles, 0);
  assert.strictEqual(streamingTriangleCount(2, [[0, 1]]).triangles, 0, "no third vertex");

  // (2) Three independent counters agree with brute force on LCG graphs.
  for (let seed = 1; seed <= 8; seed++) {
    const n = 9;
    const { edges } = randomGraph(n, 40 + 5 * (seed % 3), seed * 17);
    const truth = bruteTriangles(n, edges);
    assert.strictEqual(streamingTriangleCount(n, edges).triangles, truth,
      `seed ${seed}: streaming counter`);
    assert.strictEqual(orderedTriangleCount(n, edges).triangles, truth,
      `seed ${seed}: heavy-light counter`);
    assert.strictEqual(unorderedTriangleCount(n, edges).triangles, truth,
      `seed ${seed}: unordered counter`);
    assert.ok(truth > 0, `seed ${seed}: the test graph should contain triangles`);
  }

  // (3) The stream really is a stream: after every prefix the running count equals the exact
  //     triangle count of that prefix subgraph.
  for (let seed = 11; seed <= 13; seed++) {
    const n = 8;
    const { edges } = randomGraph(n, 55, seed * 23);
    const { running, stats } = streamingTriangleCount(n, edges);
    assert.strictEqual(running.length, edges.length);
    assert.strictEqual(stats.edgesStreamed, edges.length);
    assert.strictEqual(stats.edgesKept, edges.length);
    for (let i = 1; i <= edges.length; i++) {
      assert.strictEqual(running[i - 1], bruteTriangles(n, edges.slice(0, i)),
        `seed ${seed}: running count after prefix ${i}`);
    }
    assert.strictEqual(running[running.length - 1], bruteTriangles(n, edges));
    assert.ok(stats.intersections > 0, "the counter does perform set intersections");
  }

  // (4) The heavy-light orientation saves membership tests on a dense graph.
  const dense = completeGraph(12);
  const ordered = orderedTriangleCount(12, dense);
  const unordered = unorderedTriangleCount(12, dense);
  assert.strictEqual(ordered.triangles, 220, "C(12,3)");
  assert.strictEqual(streamingTriangleCount(12, dense).triangles, 220);
  assert.ok(ordered.stats.intersections < unordered.stats.intersections,
    `orienting by degree must cut the work (${ordered.stats.intersections} < ` +
    `${unordered.stats.intersections})`);

  // (5) p = 1 reproduces the exact count, and the two sparsified paths agree.
  const graph = { n: 10, edges: randomGraph(10, 60, 99).edges };
  const exact = bruteTriangles(graph.n, graph.edges);
  for (let seed = 1; seed <= 5; seed++) {
    const estimate = sparsifiedEstimate(graph.n, graph.edges, 1.0, seed);
    assert.strictEqual(estimate.count, exact, "keeping everything is the exact counter");
    assert.strictEqual(estimate.estimate, exact);
    assert.strictEqual(estimate.keptCount, graph.edges.length);
  }
  assert.ok(exact > 20, "the test graph is triangle-rich");

  // (6) Unbiasedness and variance shrink as p grows (all with fixed seeds).
  const seeds = seedRange(1, 60);
  const mid = estimatorStats(graph.n, graph.edges, 0.6, seeds);
  const high = estimatorStats(graph.n, graph.edges, 0.8, seeds);
  const low = estimatorStats(graph.n, graph.edges, 0.4, seeds);
  assert.ok(Math.abs(mid.mean - exact) <= 0.25 * exact,
    `p=0.6 mean ${mid.mean.toFixed(2)} must be close to ${exact}`);
  assert.ok(Math.abs(high.mean - exact) <= 0.25 * exact,
    `p=0.8 mean ${high.mean.toFixed(2)} must be close to ${exact}`);
  assert.ok(high.variance < mid.variance, "larger p: smaller variance");
  assert.ok(mid.variance < low.variance, "smaller p: larger variance");

  // (7) The deterministic sampler is cross-checked by counting the kept graph directly, and the
  //     heavy tail of small-p sampling is documented rather than hidden.
  for (let seed = 1; seed <= 10; seed++) {
    const run = sparsifiedEstimate(graph.n, graph.edges, 0.5, seed);
    assert.strictEqual(run.count, run.directCount,
      "streaming the sampled edges equals counting the sampled graph");
    assert.strictEqual(run.estimate, run.count / 0.125);
  }
  const tiny = estimatorStats(graph.n, graph.edges, 0.2, seeds);
  assert.ok(tiny.values.some((value) => value === 0),
    "at p = 0.2 many samples contain no triangle at all");
  assert.ok(tiny.mean >= exact / 4 && tiny.mean <= 4 * exact,
    `even the heavy-tailed p=0.2 mean stays in the right order of magnitude (${tiny.mean.toFixed(2)})`);
  assert.ok(tiny.variance > high.variance);

  // (8) Determinism: same seeds, same numbers; different seeds, different samples.
  assert.deepStrictEqual(sparsifiedEstimate(graph.n, graph.edges, 0.6, 7),
    sparsifiedEstimate(graph.n, graph.edges, 0.6, 7));
  assert.deepStrictEqual(estimatorStats(graph.n, graph.edges, 0.6, seeds),
    estimatorStats(graph.n, graph.edges, 0.6, seeds));
  const first = sparsify(graph.edges, 0.6, 1);
  const second = sparsify(graph.edges, 0.6, 2);
  assert.notDeepStrictEqual(first.flags, second.flags, "different seeds sample differently");
  assert.strictEqual(first.kept.length, first.flags.filter(Boolean).length);
}

module.exports = { lcgNext, randomGraph, completeGraph, bruteTriangles, streamingTriangleCount,
  orderedTriangleCount, unorderedTriangleCount, sparsify, sparsifiedEstimate, estimatorStats,
  seedRange };

if (require.main === module) {
  main();
  console.log("PASS");
}
