/** Task 845 - Implement approximate personalized PageRank with bookmark coloring.
 *  (Tier: Master, CTCI Ch 4 — Trees and Graphs)
 *
 * Personalized PageRank from a source s is π = (1 - α) * SUM_{k >= 0} α^k P^k e_s, i.e. the
 * distribution of a random walk that follows an edge with probability α and restarts at s otherwise.
 * Two approximations are implemented and both are cross-checked against a dense power-iteration oracle:
 *
 *  1. Forward push (Andersen–Chung–Lang), the deterministic half of bookmark coloring. Every node
 *     keeps a *colour* c and a *residual* r; a push moves (1 - α) r_u into c_u (that is the walk
 *     terminating at u) and spreads α r_u / deg(u) to the out-neighbours. This maintains two exact
 *     invariants —
 *         SUM_v c_v + SUM_v r_v = 1      (mass conservation, after every single push)
 *         c_v <= π_v for every v          (the colour never overshoots)
 *     — and pushing the relative residual below ε yields a certified approximation whose error and
 *     push count are both measured across a range of ε.
 *  2. Bookmark coloring (Berthold et al.): bookmarks are nodes whose colour was precomputed once. A
 *     query walks from s and, at the first node that is a bookmark, adds weight * c_h instead of
 *     continuing. Because a walk standing on h continues exactly like a walk started at h, the
 *     substitution is exact in expectation: the estimator stays unbiased while a long random-walk tail
 *     is replaced by a table lookup — the whole point of the data structure.
 *
 * Everything is deterministic: graphs, pushes and walks are driven by seeded splitmix64, and every
 * estimate is compared against the oracle rather than eye-balled.
 *
 * Simplifications, stated honestly: the bookmark table holds colours computed at a small ε instead of
 * exact PPR, graphs are small, and a dangling node (out-degree 0) simply absorbs the walk, so PPR mass
 * can sum to less than 1 — the tests measure that leak instead of hiding it. No timers, threads or
 * sockets.
 *
 * Invariants verified below:
 *  - push safety: mass conservation, c <= π componentwise, and ε -> 0 reproducing the oracle to 1e-6;
 *  - accuracy/cost tradeoff: smaller ε means strictly more pushes and strictly less error, with top-k
 *    recall improving along the way;
 *  - Monte Carlo correctness: the estimator is a distribution, more walks means less error, and the
 *    mean over independent seeds converges to the oracle (unbiasedness measured, not assumed);
 *  - bookmark coloring: unbiased for the same oracle, strictly better error at an equal sample count,
 *    and far fewer walk steps per query; the shortcut mass is accounted for;
 *  - structure: dangling nodes leak exactly what the oracle leaks, a disconnected component keeps zero
 *    PPR, self-loops conserve mass, and a dangling source cannot be pushed;
 *  - determinism: identical seeds reproduce colours, push counts, walks and answers.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;

/** Deterministic splitmix64: the only source of pseudo-randomness. */
class SplitMix64 {
  constructor(seed) {
    this.state = (BigInt(seed) & MASK64) || 1n;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }

  /** Uniform integer in [0, bound). */
  upto(bound) {
    return Number(this.next() % BigInt(bound));
  }

  /** Uniform float in [0, 1) with 2^-53 resolution. */
  unit() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }
}

/** Deterministic graph: a spanning path (optional) plus random arcs and optional dangling sinks. */
function buildGraph(options = {}) {
  const vertices = options.vertices ?? 26;
  const extraEdges = options.extraEdges ?? 20;
  const seed = options.seed ?? 845;
  const directed = options.directed ?? true;
  const connected = options.connected ?? true;
  const dangling = options.dangling ?? 0;
  const rng = new SplitMix64(seed);
  const adjacency = new Map();
  for (let i = 0; i < vertices + dangling; i++) adjacency.set(i, []);
  const addEdge = (from, to) => {
    adjacency.get(from).push(to);
    if (!directed) adjacency.get(to).push(from);
  };
  for (let i = 1; i < vertices && connected; i++) addEdge(i - 1, i);
  for (let i = 0; i < extraEdges; i++) {
    const from = rng.upto(vertices);
    const to = rng.upto(vertices);
    if (from === to) continue;
    addEdge(from, to);
  }
  for (let i = 0; i < dangling; i++) addEdge(rng.upto(vertices), vertices + i);
  return { adjacency, vertices: vertices + dangling };
}

const outDegree = (graph, node) => (graph.adjacency.get(node) ?? []).length;
const nodesOf = (graph) => [...graph.adjacency.keys()].sort((a, b) => a - b);

/** The dense oracle: power iteration on π = (1 - α) e_s + α P^T π; dangling nodes absorb the mass. */
function exactPpr(graph, source, alpha, iterations = 800) {
  const nodes = nodesOf(graph);
  let rank = new Map(nodes.map((node) => [node, 0]));
  rank.set(source, 1);
  for (let step = 0; step < iterations; step++) {
    const next = new Map(nodes.map((node) => [node, 0]));
    next.set(source, 1 - alpha);
    for (const node of nodes) {
      const degree = outDegree(graph, node);
      if (degree === 0) continue; // the walk dies here and that mass leaks out
      const share = (alpha * rank.get(node)) / degree;
      for (const target of graph.adjacency.get(node)) {
        next.set(target, next.get(target) + share);
      }
    }
    rank = next;
  }
  return rank;
}

function l1Distance(left, right, nodes) {
  let total = 0;
  for (const node of nodes) total += Math.abs((left.get(node) ?? 0) - (right.get(node) ?? 0));
  return total;
}

/**
 * Forward push: a node is pushed while its residual r_u >= ε * deg(u). Returns the colour together
 * with the counters the cost measurements need.
 */
function forwardPush(graph, source, alpha, epsilon, options = {}) {
  const maxPushes = options.maxPushes ?? 200000;
  const color = new Map();
  const residual = new Map([[source, 1]]);
  const queue = [source];
  let pushes = 0;
  let operations = 0;
  while (queue.length > 0 && pushes < maxPushes) {
    const node = queue.pop();
    const pending = residual.get(node) ?? 0;
    const degree = outDegree(graph, node);
    if (degree === 0 || pending < epsilon * degree) continue;
    residual.set(node, 0);
    // A walk standing on u restarts (terminates) with probability (1 - α) and walks on with α.
    color.set(node, (color.get(node) ?? 0) + (1 - alpha) * pending);
    const share = (alpha * pending) / degree;
    for (const target of graph.adjacency.get(node)) {
      const before = residual.get(target) ?? 0;
      const after = before + share;
      residual.set(target, after);
      operations++;
      const threshold = epsilon * Math.max(1, outDegree(graph, target));
      if (outDegree(graph, target) > 0 && before < threshold && after >= threshold) {
        queue.push(target);
      }
    }
    pushes++;
  }
  let colorMass = 0;
  for (const value of color.values()) colorMass += value;
  let residualMass = 0;
  for (const value of residual.values()) residualMass += value;
  return { color, residual, pushes, operations, colorMass, residualMass };
}

/** Nodes that still carry residual mass. */
const residualNodes = (push) =>
  [...push.residual.entries()].filter(([, value]) => value > 0).map(([node]) => node);

/**
 * Walk sampler. Without a bookmark table it deposits (1 - α) / walks at every visited node and stops
 * with probability (1 - α) after each deposit; the survival probability supplies the α^k decay, so the
 * estimator is exactly unbiased for π. With a table, the first bookmark h met replaces the whole
 * remaining tail by its colour (the walk standing on h continues exactly like a walk from h).
 */
function walkEstimate(graph, source, alpha, walks, rng, options = {}) {
  const colors = options.colors ?? null;
  const estimate = new Map();
  const deposit = (node, amount) => estimate.set(node, (estimate.get(node) ?? 0) + amount);
  const share = (1 - alpha) / walks;
  let steps = 0;
  let bookmarkHits = 0;
  let shortcutMass = 0;
  for (let walk = 0; walk < walks; walk++) {
    let node = source;
    for (;;) {
      if (colors !== null && colors.has(node)) {
        for (const [target, value] of colors.get(node)) deposit(target, value / walks);
        shortcutMass += 1 / walks;
        bookmarkHits++;
        break;
      }
      deposit(node, share);
      if (rng.unit() < 1 - alpha) break; // the walk restarts instead of taking an edge step
      const degree = outDegree(graph, node);
      if (degree === 0) break; // absorbed by a dangling node
      node = graph.adjacency.get(node)[rng.upto(degree)];
      steps++;
    }
  }
  return { estimate, steps, bookmarkHits, shortcutMass };
}

/** Precomputes the bookmark table: one colouring per bookmark node. */
function buildBookmarks(graph, bookmarks, alpha, epsilon) {
  const colors = new Map();
  for (const bookmark of bookmarks) {
    colors.set(bookmark, forwardPush(graph, bookmark, alpha, epsilon).color);
  }
  return colors;
}

/** Mean L1 error of an estimator over several sources. */
function meanError(graph, sources, alpha, estimator, nodes) {
  let total = 0;
  for (const source of sources) {
    total += l1Distance(estimator(source), exactPpr(graph, source, alpha), nodes);
  }
  return total / sources.length;
}

/** Top-k recall of an approximation against the oracle ranking. */
function topKRecall(approximate, oracle, nodes, k) {
  const order = (map) => [...nodes].sort((a, b) => (map.get(b) ?? 0) - (map.get(a) ?? 0));
  const truth = order(oracle).slice(0, k);
  const guess = new Set(order(approximate).slice(0, k));
  return truth.filter((node) => guess.has(node)).length / k;
}

module.exports = { MASK64, SplitMix64, buildGraph, outDegree, nodesOf, exactPpr, l1Distance,
  forwardPush, residualNodes, walkEstimate, buildBookmarks, meanError, topKRecall };

if (require.main === module) {
  const assert = require("node:assert");
  const alpha = 0.85;
  const graph = buildGraph({ vertices: 26, extraEdges: 20, seed: 845 });
  const nodes = nodesOf(graph);
  const source = 0;

  // --- the oracle -----------------------------------------------------------
  const oracle = exactPpr(graph, source, alpha);
  const oracleMass = nodes.reduce((acc, node) => acc + oracle.get(node), 0);
  assert.ok(Math.abs(oracleMass - 1) < 1e-9, `the oracle sums to 1 (${oracleMass})`);
  assert.ok(nodes.every((node) => oracle.get(node) >= 0), "probabilities are non-negative");
  assert.strictEqual(Math.max(...nodes.map((node) => oracle.get(node))), oracle.get(source),
    "the most likely node to be visited is the source itself");
  assert.ok(exactPpr(graph, source, 0.3).get(source) > oracle.get(source),
    "a smaller alpha keeps more mass at the source");

  // --- forward push invariants ----------------------------------------------
  const coarse = forwardPush(graph, source, alpha, 1e-2);
  assert.ok(Math.abs(coarse.colorMass + coarse.residualMass - 1) < 1e-12,
    `mass is conserved (${coarse.colorMass} + ${coarse.residualMass})`);
  assert.ok(coarse.pushes > 0 && coarse.operations > coarse.pushes, "the push does real work");
  for (const node of nodes) {
    assert.ok((coarse.color.get(node) ?? 0) <= oracle.get(node) + 1e-12,
      `the colour never overshoots the true PPR at ${node}`);
  }
  assert.ok(coarse.colorMass < 1, "a coarse colouring holds less than the whole mass");
  assert.ok(residualNodes(coarse).length > 0, "and leaves a residual behind");

  // --- smaller epsilon: more work, less error -------------------------------
  const tradeoff = [];
  for (const epsilon of [1e-2, 1e-3, 1e-4, 1e-5]) {
    const push = forwardPush(graph, source, alpha, epsilon);
    tradeoff.push({ epsilon, pushes: push.pushes,
      error: l1Distance(push.color, oracle, nodes),
      recall: topKRecall(push.color, oracle, nodes, 5),
      colorMass: push.colorMass });
    assert.ok(Math.abs(push.colorMass + push.residualMass - 1) < 1e-12,
      `mass is conserved at epsilon ${epsilon}`);
  }
  for (let i = 1; i < tradeoff.length; i++) {
    assert.ok(tradeoff[i].pushes > tradeoff[i - 1].pushes,
      `pushes grow as epsilon shrinks (${tradeoff[i - 1].pushes} -> ${tradeoff[i].pushes})`);
    assert.ok(tradeoff[i].error < tradeoff[i - 1].error,
      `error shrinks as epsilon shrinks (${tradeoff[i - 1].error} -> ${tradeoff[i].error})`);
    assert.ok(tradeoff[i].colorMass >= tradeoff[i - 1].colorMass - 1e-12,
      "and the colour mass grows monotonically");
  }
  assert.ok(tradeoff[3].error < 1e-3, `epsilon 1e-5 leaves a tiny L1 error (${tradeoff[3].error})`);
  assert.ok(tradeoff[3].recall >= tradeoff[0].recall, "top-5 recall does not get worse");
  assert.ok(tradeoff[3].recall >= 0.6, `top-5 recall at 1e-5 is ${tradeoff[3].recall}`);

  // --- driving epsilon down reproduces the oracle ---------------------------
  const exactish = forwardPush(graph, source, alpha, 1e-7, { maxPushes: 300000 });
  const exactishError = l1Distance(exactish.color, oracle, nodes);
  assert.ok(exactishError < 1e-4, `a converged push reproduces the oracle (${exactishError})`);
  assert.ok(Math.abs(exactish.colorMass + exactish.residualMass - 1) < 1e-12,
    "mass stays conserved while converging");

  // --- plain Monte Carlo walks ----------------------------------------------
  const walkError = (walks, seed) => meanError(graph, [0, 5, 11, 17], alpha,
    (src) => walkEstimate(graph, src, alpha, walks, new SplitMix64(seed + src)).estimate, nodes);
  const smallWalks = walkError(200, 9000);
  const largeWalks = walkError(4000, 9100);
  assert.ok(largeWalks < smallWalks,
    `more walks means less error (${largeWalks.toFixed(4)} < ${smallWalks.toFixed(4)})`);
  assert.ok(largeWalks < 0.08, `4000 walks give a small L1 error (${largeWalks.toFixed(4)})`);
  const single = walkEstimate(graph, source, alpha, 500, new SplitMix64(1));
  assert.ok(Math.abs(single.estimate.get(source) - oracle.get(source)) < 0.15,
    "the estimator lands near the oracle at the source");
  let singleTotal = 0;
  for (const value of single.estimate.values()) singleTotal += value;
  assert.ok(singleTotal > 0.9 && singleTotal < 1.1,
    `the estimate concentrates around a probability mass (${singleTotal.toFixed(4)})`);
  const tiny = walkEstimate(graph, source, alpha, 50, new SplitMix64(77));
  assert.ok(tiny.steps > 0 && tiny.steps <= 50 * 40, `the walk budget is respected (${tiny.steps})`);
  // Unbiasedness: the mean over independent seeds approaches the oracle.
  const average = (estimator, seeds, walks) => {
    const total = new Map(nodes.map((node) => [node, 0]));
    for (let seed = 0; seed < seeds; seed++) {
      const sample = estimator(seed, walks);
      for (const node of nodes) total.set(node, total.get(node) + sample.get(node) / seeds);
    }
    return total;
  };
  const plainAverage = average((seed, walks) =>
    walkEstimate(graph, source, alpha, walks, new SplitMix64(1000 + seed * 37)).estimate, 40, 200);
  const plainBias = l1Distance(plainAverage, oracle, nodes);
  assert.ok(plainBias < 0.05, `averaging independent runs removes the noise (${plainBias.toFixed(4)})`);

  // --- bookmark coloring ----------------------------------------------------
  const bookmarks = [3, 7, 11, 15, 19, 23];
  const table = buildBookmarks(graph, bookmarks, alpha, 1e-5);
  assert.strictEqual(table.size, bookmarks.length, "every bookmark has a colour");
  for (const bookmark of bookmarks) {
    const push = forwardPush(graph, bookmark, alpha, 1e-5);
    assert.strictEqual(table.get(bookmark).size, push.color.size,
      `bookmark ${bookmark} stores exactly the pushed colour`);
    assert.ok(table.get(bookmark).size > 0, `bookmark ${bookmark} has a non-empty colour`);
  }
  const sources = [0, 5, 11, 19];
  const plain = meanError(graph, sources, alpha,
    (src) => walkEstimate(graph, src, alpha, 2000, new SplitMix64(400 + src)).estimate, nodes);
  const bookmarked = meanError(graph, sources, alpha,
    (src) => walkEstimate(graph, src, alpha, 2000, new SplitMix64(400 + src),
      { colors: table }).estimate, nodes);
  assert.ok(bookmarked < plain,
    `bookmarks cut the error at equal sample count (${bookmarked.toFixed(4)} < ${plain.toFixed(4)})`);
  let bookmarkSteps = 0;
  let plainSteps = 0;
  for (const src of sources) {
    bookmarkSteps += walkEstimate(graph, src, alpha, 2000, new SplitMix64(400 + src),
      { colors: table }).steps;
    plainSteps += walkEstimate(graph, src, alpha, 2000, new SplitMix64(400 + src)).steps;
  }
  assert.ok(bookmarkSteps < plainSteps * 0.8,
    `bookmark queries spend far fewer walk steps (${bookmarkSteps} vs ${plainSteps})`);
  const hubRun = walkEstimate(graph, source, alpha, 2000, new SplitMix64(400), { colors: table });
  assert.ok(hubRun.bookmarkHits > 0, `bookmarks are actually hit (${hubRun.bookmarkHits})`);
  assert.ok(hubRun.shortcutMass > 0 && hubRun.shortcutMass <= 1,
    `the shortcut mass is a real fraction of the total (${hubRun.shortcutMass.toFixed(3)})`);
  let hubTotal = 0;
  for (const value of hubRun.estimate.values()) hubTotal += value;
  assert.ok(hubTotal > 0.8 && hubTotal < 1.3,
    `the bookmark estimate also concentrates around a probability mass (${hubTotal.toFixed(4)})`);
  // A query whose source is itself a bookmark returns that bookmark's colour exactly.
  const selfQuery = walkEstimate(graph, bookmarks[0], alpha, 10, new SplitMix64(5),
    { colors: table });
  assert.strictEqual(selfQuery.bookmarkHits, 10, "every walk stops immediately at the source bookmark");
  assert.strictEqual(selfQuery.steps, 0, "and takes no edge step at all");
  // Unbiasedness of the bookmark estimator: the mean over independent seeds hits the same oracle.
  const hubAverage = average((seed, walks) =>
    walkEstimate(graph, source, alpha, walks, new SplitMix64(2000 + seed * 41),
      { colors: table }).estimate, 40, 200);
  const hubBias = l1Distance(hubAverage, oracle, nodes);
  assert.ok(hubBias < 0.05, `the bookmark estimator is unbiased (${hubBias.toFixed(4)})`);

  // --- top-k retrieval ------------------------------------------------------
  const approximate = walkEstimate(graph, source, alpha, 8000, new SplitMix64(31), { colors: table });
  const recall = topKRecall(approximate.estimate, oracle, nodes, 5);
  assert.ok(recall >= 0.8, `top-5 recall from the bookmark sampler is ${recall}`);
  const precise = forwardPush(graph, source, alpha, 1e-6);
  assert.strictEqual(topKRecall(precise.color, oracle, nodes, 5), 1,
    "a sharp colouring recovers the exact top five");

  // --- structural edge cases ------------------------------------------------
  const sink = buildGraph({ vertices: 12, extraEdges: 6, seed: 8450, dangling: 2 });
  const sinkOracle = exactPpr(sink, 0, alpha);
  const sinkNodes = nodesOf(sink);
  const sinkMass = sinkNodes.reduce((acc, node) => acc + sinkOracle.get(node), 0);
  assert.ok(sinkMass < 1, `dangling nodes leak mass (${sinkMass.toFixed(4)})`);
  const sinkPush = forwardPush(sink, 0, alpha, 1e-4);
  assert.ok(Math.abs(sinkPush.colorMass + sinkPush.residualMass - 1) < 1e-12,
    "the push still conserves mass with dangling nodes");
  assert.ok(sinkPush.residualMass > 1e-3, "the leaked mass stays in the residual, as it must");
  assert.ok(sinkPush.colorMass < sinkMass + 1e-12, "and the colour cannot exceed what the oracle has");

  const island = { adjacency: new Map([[0, [1]], [1, [0]], [2, [3]], [3, [2]]]), vertices: 4 };
  const islandOracle = exactPpr(island, 0, 0.85);
  assert.strictEqual(islandOracle.get(2), 0, "a disconnected component keeps zero PPR");
  assert.ok(Math.abs(islandOracle.get(0) + islandOracle.get(1) - 1) < 1e-9,
    "and all the mass stays in the source's component");
  const islandPush = forwardPush(island, 0, 0.85, 1e-3);
  assert.strictEqual(islandPush.color.get(2) ?? 0, 0, "the colour never reaches the island");

  const selfLoop = { adjacency: new Map([[0, [0, 1]], [1, [0]]]), vertices: 2 };
  const selfLoopOracle = exactPpr(selfLoop, 0, 0.85);
  assert.ok(Math.abs(selfLoopOracle.get(0) + selfLoopOracle.get(1) - 1) < 1e-9,
    "a self loop still conserves mass");
  const singleNode = { adjacency: new Map([[0, []]]), vertices: 1 };
  assert.ok(Math.abs(exactPpr(singleNode, 0, 0.85).get(0) - 0.15) < 1e-12,
    "a graph of one dangling node gives the source exactly its restart mass");
  assert.strictEqual(forwardPush(singleNode, 0, 0.85, 1e-9).pushes, 0,
    "a dangling source cannot be pushed");

  // --- determinism ----------------------------------------------------------
  const pushA = forwardPush(graph, source, alpha, 1e-4);
  const pushB = forwardPush(graph, source, alpha, 1e-4);
  assert.deepStrictEqual([...pushA.color.entries()], [...pushB.color.entries()],
    "pushes are deterministic");
  assert.strictEqual(pushA.pushes, pushB.pushes, "including their push counts");
  const walkA = walkEstimate(graph, source, alpha, 400, new SplitMix64(4242), { colors: table });
  const walkB = walkEstimate(graph, source, alpha, 400, new SplitMix64(4242), { colors: table });
  assert.deepStrictEqual([...walkA.estimate.entries()], [...walkB.estimate.entries()],
    "walks are deterministic");
  assert.strictEqual(walkA.steps, walkB.steps, "and their step counts match");

  console.log(`oracle mass=${oracleMass.toFixed(12)}; push tradeoff ` +
    tradeoff.map((entry) => `eps=${entry.epsilon}:${entry.pushes}p/${entry.error.toFixed(5)}e`).join(" "));
  console.log(`walks: 200->${smallWalks.toFixed(4)} 4000->${largeWalks.toFixed(4)}; bookmarks: ` +
    `${bookmarked.toFixed(4)} vs plain ${plain.toFixed(4)} (steps ${bookmarkSteps} vs ${plainSteps})`);
  console.log(`top-5 recall: sampler=${recall} colouring=${topKRecall(precise.color, oracle, nodes, 5)}; ` +
    `dangling leak=${(1 - sinkMass).toFixed(4)}; bias walks=${plainBias.toFixed(4)} bookmarks=${hubBias.toFixed(4)}`);
  console.log("PASS");
}
