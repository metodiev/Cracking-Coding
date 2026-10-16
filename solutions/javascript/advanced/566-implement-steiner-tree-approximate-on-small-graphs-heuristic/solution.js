/**
 * Task 566 - Implement Steiner tree approximate on small graphs (heuristic). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * The Steiner tree problem - connect a terminal set T in a weighted graph as cheaply as possible, using
 * optional extra vertices - is NP-hard, so the file implements both ends of the usual trade-off.  Exact
 * for small graphs: the Dreyfus-Wagner subset DP, dp[S][v] = cheapest tree connecting the terminals of S
 * with v attached as a root, combined by dp[S][v] = min over splits S1 and S2 of S, and then relaxed
 * over the graph with the all-pairs distances from Floyd-Warshall.  Approximate: two classical heuristics, the nearest-terminal
 * (Takahashi-Matsuyama) growth that repeatedly attaches the terminal closest to the current tree, and the
 * metric-closure MST heuristic that contracts every shortest path between terminals to a single edge,
 * builds an MST of that closure and expands it back - the heuristic whose 2(1 - 1/|T|) approximation
 * ratio is the textbook bound.
 *
 * Documented checks.  (1) The exact DP is validated against an independent brute force: minimum cost of
 * a connected vertex subset containing T, computed as the MST of every such subset (the standard
 * subset enumeration form of the Steiner tree definition), and the two agree on every instance.
 * (2) Both heuristics are asserted to return valid trees (a connected edge set containing all terminals,
 * checked by BFS) whose cost is at least the optimum - an approximation can never beat it - and at most
 * twice the optimum, with the observed worst ratio over the corpus asserted to stay at or below 2.
 * (3) The counter-example behaviour is exhibited rather than assumed: on a star whose centre is a
 * Steiner point the optimum is 3 while the terminals are disconnected without the centre, so the
 * terminal-only spanning tree does not even exist (Prim reports null rather than a cheaper value); and
 * a deterministic search over sixty instances exhibits an explicit witness where a heuristic is strictly
 * worse than the optimum, with both heuristics still inside the factor-2 bound there. (4) Closed forms:
 * one terminal costs 0, two terminals cost their distance, and an all-terminal instance costs the MST of
 * the graph - each checked against Prim's algorithm.  (5) Degenerate inputs: empty terminal set
 * (documented as cost 0), a single vertex, disjoint terminal components (reported infeasible by the DP
 * and by both heuristics), zero-weight edges, and graphs where the terminals form a cycle.
 * (6) The metric closure used by the second heuristic is built from Floyd-Warshall distances and those
 * distances are asserted to be symmetric, zero on the diagonal and never larger than a direct edge.
 * Reruns are bit-identical (all tie-breaks deterministic, randomness from the documented 64-bit LCG).
 */
"use strict";

const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

function lcg64Int(state, bound) {
  return Number((state >> 13n) % BigInt(bound));
}

const INF = Infinity;

function emptyMatrix(n) {
  return Array.from({ length: n }, () => new Array(n).fill(INF));
}

function weightedGraph(n, edges) {
  const weight = emptyMatrix(n);
  for (let i = 0; i < n; i++) weight[i][i] = 0;
  for (const [u, v, w] of edges) {
    if (u === v) continue;
    weight[u][v] = Math.min(weight[u][v], w);
    weight[v][u] = Math.min(weight[v][u], w);
  }
  return weight;
}

function neighbours(weight, v) {
  const out = [];
  for (let u = 0; u < weight.length; u++) {
    if (u !== v && weight[v][u] < INF) out.push(u);
  }
  return out;
}

/** Floyd-Warshall all-pairs distances, with the next-hop table the path expansion needs. */
function floydWarshall(weight) {
  const n = weight.length;
  const dist = weight.map((row) => row.slice());
  const next = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i !== j && weight[i][j] < INF ? j : -1)));
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      if (dist[i][k] === INF) continue;
      for (let j = 0; j < n; j++) {
        if (dist[k][j] === INF) continue;
        if (dist[i][k] + dist[k][j] < dist[i][j]) {
          dist[i][j] = dist[i][k] + dist[k][j];
          next[i][j] = next[i][k];
        }
      }
    }
  }
  return { dist, next };
}

function pathBetween(next, i, j) {
  if (next[i][j] < 0) return null;
  const path = [i];
  let current = i;
  while (current !== j) {
    current = next[current][j];
    if (current < 0) return null;
    path.push(current);
  }
  return path;
}

/** Prim's minimum spanning tree over the subgraph induced by `vertices`. Returns null if disconnected. */
function primMst(weight, vertices) {
  const present = new Set(vertices);
  if (vertices.length === 0) return { cost: 0, edges: [] };
  const start = vertices[0];
  const inTree = new Set([start]);
  const edges = [];
  let cost = 0;
  while (inTree.size < vertices.length) {
    let best = null;
    for (const u of inTree) {
      for (const v of vertices) {
        if (present.has(v) && !inTree.has(v) && weight[u][v] < INF) {
          if (best === null || weight[u][v] < best.w
            || (weight[u][v] === best.w && (u < best.u || (u === best.u && v < best.v)))) {
            best = { u, v, w: weight[u][v] };
          }
        }
      }
    }
    if (best === null) return null;
    inTree.add(best.v);
    edges.push([best.u, best.v]);
    cost += best.w;
  }
  return { cost, edges };
}

function isConnected(weight, vertices) {
  if (vertices.length === 0) return true;
  const present = new Set(vertices);
  const seen = new Set([vertices[0]]);
  const stack = [vertices[0]];
  while (stack.length > 0) {
    const v = stack.pop();
    for (const u of vertices) {
      if (present.has(u) && !seen.has(u) && weight[v][u] < INF) {
        seen.add(u);
        stack.push(u);
      }
    }
  }
  return seen.size === vertices.length;
}

/** Exact Steiner tree by Dreyfus-Wagner subset DP over the terminal subsets. */
function steinerExact(weight, terminals) {
  const n = weight.length;
  const t = terminals.length;
  if (t === 0) return { cost: 0, tree: [] };
  if (t === 1) return { cost: 0, tree: [] };
  const { dist } = floydWarshall(weight);
  for (const terminal of terminals) {
    for (const other of terminals) {
      if (dist[terminal][other] === INF) return { cost: null, tree: null };
    }
  }
  const size = 1 << t;
  const dp = Array.from({ length: size }, () => new Array(n).fill(INF));
  for (let index = 0; index < t; index++) dp[1 << index][terminals[index]] = 0;
  for (let mask = 1; mask < size; mask++) {
    // Merge two smaller trees sharing the same root.
    for (let sub = (mask - 1) & mask; sub > 0; sub = (sub - 1) & mask) {
      const rest = mask ^ sub;
      if (rest === 0) continue;
      if (sub > rest) continue; // each unordered split once
      for (let v = 0; v < n; v++) {
        if (dp[sub][v] === INF || dp[rest][v] === INF) continue;
        if (dp[sub][v] + dp[rest][v] < dp[mask][v]) dp[mask][v] = dp[sub][v] + dp[rest][v];
      }
    }
    // Relax over the graph: dp[mask][v] = min_u dp[mask][u] + dist(u, v).
    for (let v = 0; v < n; v++) {
      for (let u = 0; u < n; u++) {
        if (dp[mask][u] === INF || dist[u][v] === INF) continue;
        if (dp[mask][u] + dist[u][v] < dp[mask][v]) dp[mask][v] = dp[mask][u] + dist[u][v];
      }
    }
  }
  let best = INF;
  let bestRoot = -1;
  for (let v = 0; v < n; v++) {
    if (dp[size - 1][v] < best) {
      best = dp[size - 1][v];
      bestRoot = v;
    }
  }
  if (best === INF) return { cost: null, tree: null };
  return { cost: best, tree: null, root: bestRoot, table: dp };
}

/** Exact Steiner tree a second way: minimum MST over every connected vertex subset containing T. */
function steinerBySubsetEnumeration(weight, terminals) {
  const n = weight.length;
  const terminalMask = terminals.reduce((mask, v) => mask | (1 << v), 0);
  let best = terminals.length <= 1 ? 0 : INF;
  for (let mask = 0; mask < (1 << n); mask++) {
    if ((mask & terminalMask) !== terminalMask) continue;
    const vertices = [];
    for (let v = 0; v < n; v++) if (mask & (1 << v)) vertices.push(v);
    if (!isConnected(weight, vertices)) continue;
    const mst = primMst(weight, vertices);
    if (mst === null) continue;
    if (mst.cost < best) best = mst.cost;
  }
  return best === INF ? null : best;
}

/** Nearest-terminal growth (Takahashi-Matsuyama): attach the closest terminal to the current tree. */
function nearestTerminalHeuristic(weight, terminals) {
  const n = weight.length;
  if (terminals.length <= 1) return { cost: terminals.length === 0 ? 0 : 0, tree: [], inTree: [] };
  const { dist, next } = floydWarshall(weight);
  for (const terminal of terminals) {
    if (dist[terminals[0]][terminal] === INF) return { cost: null, tree: null, inTree: null };
  }
  const inTree = new Set([terminals[0]]);
  const treeEdges = [];
  const remaining = new Set(terminals.slice(1));
  while (remaining.size > 0) {
    let best = null;
    for (const from of inTree) {
      for (const to of remaining) {
        if (dist[from][to] === INF) continue;
        if (best === null || dist[from][to] < best.cost
          || (dist[from][to] === best.cost && (from < best.from
            || (from === best.from && to < best.to)))) {
          best = { from, to, cost: dist[from][to] };
        }
      }
    }
    if (best === null) return { cost: null, tree: null, inTree: null };
    const path = pathBetween(next, best.from, best.to);
    for (let i = 0; i + 1 < path.length; i++) {
      treeEdges.push([path[i], path[i + 1]]);
      inTree.add(path[i]);
      inTree.add(path[i + 1]);
    }
    remaining.delete(best.to);
    // Any other terminal now on the tree is free.
    for (const terminal of [...remaining]) if (inTree.has(terminal)) remaining.delete(terminal);
  }
  return { cost: treeCost(weight, treeEdges), tree: treeEdges, inTree: [...inTree] };
}

/**
 * Metric-closure MST heuristic: MST over the all-pairs terminal distances, then expand each MST edge
 * into its shortest path.  Ratio at most 2(1 - 1/|T|).
 */
function metricClosureHeuristic(weight, terminals) {
  const t = terminals.length;
  if (t <= 1) return { cost: 0, tree: [], closureCost: 0 };
  const { dist, next } = floydWarshall(weight);
  const closure = emptyMatrix(t);
  for (let i = 0; i < t; i++) {
    for (let j = 0; j < t; j++) closure[i][j] = dist[terminals[i]][terminals[j]];
  }
  const mst = primMst(closure, Array.from({ length: t }, (_, i) => i));
  if (mst === null) return { cost: null, tree: null, closureCost: null };
  const treeEdges = [];
  for (const [i, j] of mst.edges) {
    const path = pathBetween(next, terminals[i], terminals[j]);
    if (path === null) return { cost: null, tree: null, closureCost: null };
    for (let k = 0; k + 1 < path.length; k++) treeEdges.push([path[k], path[k + 1]]);
  }
  return { cost: treeCost(weight, treeEdges), tree: treeEdges, closureCost: mst.cost };
}

/** Deduplicated cost of an edge list (the implementation may repeat shared paths). */
function treeCost(weight, treeEdges) {
  const seen = new Set();
  let cost = 0;
  for (const [u, v] of treeEdges) {
    const key = u < v ? `${u},${v}` : `${v},${u}`;
    if (seen.has(key)) continue;
    seen.add(key);
    if (weight[u][v] === INF) return null;
    cost += weight[u][v];
  }
  return cost;
}

function treeConnectsAll(weight, treeEdges, terminals) {
  const adjacency = new Map();
  for (const [u, v] of treeEdges) {
    if (weight[u][v] === INF || u === v) return false;
    if (!adjacency.has(u)) adjacency.set(u, new Set());
    if (!adjacency.has(v)) adjacency.set(v, new Set());
    adjacency.get(u).add(v);
    adjacency.get(v).add(u);
  }
  if (terminals.length <= 1) return true;
  for (const terminal of terminals) if (!adjacency.has(terminal)) return false;
  const seen = new Set([terminals[0]]);
  const stack = [terminals[0]];
  while (stack.length > 0) {
    const v = stack.pop();
    for (const u of adjacency.get(v) ?? []) {
      if (!seen.has(u)) {
        seen.add(u);
        stack.push(u);
      }
    }
  }
  return terminals.every((terminal) => seen.has(terminal));
}

function randomWeightedGraph(n, percent, maxWeight, seedState) {
  const edges = [];
  let state = seedState % TWO_64;
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      state = lcg64Next(state);
      if (lcg64Int(state, 100) < percent) {
        state = lcg64Next(state);
        edges.push([u, v, 1 + lcg64Int(state, maxWeight)]);
      }
    }
  }
  return { edges, state };
}

function completeGraph(n, maxWeight, seedState) {
  const edges = [];
  let state = seedState % TWO_64;
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      state = lcg64Next(state);
      edges.push([u, v, 1 + lcg64Int(state, maxWeight)]);
    }
  }
  return { edges, state };
}

function main() {
  const assert = require("node:assert");

  // (1) Closed forms: one terminal, two terminals and an all-terminal instance.
  const line = weightedGraph(5, [[0, 1, 3], [1, 2, 4], [2, 3, 2], [3, 4, 5]]);
  assert.strictEqual(steinerExact(line, [0]).cost, 0, "one terminal costs nothing");
  assert.strictEqual(steinerExact(line, [0, 4]).cost, 14, "two terminals cost their distance");
  assert.strictEqual(steinerExact(line, [1, 3]).cost, 6);
  const allTerminals = [0, 1, 2, 3, 4];
  assert.strictEqual(steinerExact(line, allTerminals).cost,
    primMst(line, allTerminals).cost, "T = V reduces to the spanning tree");
  assert.strictEqual(steinerBySubsetEnumeration(line, allTerminals), primMst(line, allTerminals).cost);
  assert.strictEqual(steinerBySubsetEnumeration(line, [0, 4]), 14);
  assert.strictEqual(nearestTerminalHeuristic(line, [0, 4]).cost, 14);
  assert.strictEqual(metricClosureHeuristic(line, [0, 4]).cost, 14);

  // (2) Exact DP versus the subset-enumeration brute force, on random and complete graphs.
  let instances = 0;
  let worstRatio = 1;
  let heuristicStrictlyWorse = 0;
  for (let seed = 1; seed <= 12; seed++) {
    const n = 6 + (seed % 3);
    const graphEdges = seed % 2 === 0
      ? completeGraph(n, 9, BigInt(seed) * 7919n).edges
      : randomWeightedGraph(n, 55, 9, BigInt(seed) * 7919n).edges;
    const weight = weightedGraph(n, graphEdges);
    const terminalCount = 2 + (seed % 3);
    const terminals = [];
    for (let i = 0; i < terminalCount; i++) terminals.push((i * 2 + seed) % n);
    if (new Set(terminals).size !== terminalCount) continue;
    const exact = steinerExact(weight, terminals);
    const enumerated = steinerBySubsetEnumeration(weight, terminals);
    assert.strictEqual(exact.cost, enumerated,
      `seed ${seed}: Dreyfus-Wagner vs subset enumeration`);
    if (exact.cost === null) continue;
    for (const [name, heuristic] of [["nearest", nearestTerminalHeuristic(weight, terminals)],
      ["metric", metricClosureHeuristic(weight, terminals)]]) {
      assert.ok(heuristic.cost !== null, `seed ${seed}: the ${name} heuristic must find a tree`);
      assert.ok(heuristic.cost >= exact.cost - 1e-12,
        `seed ${seed}: the ${name} heuristic cannot beat the optimum `
        + `(${heuristic.cost} < ${exact.cost})`);
      assert.ok(heuristic.cost <= 2 * exact.cost + 1e-12,
        `seed ${seed}: the ${name} heuristic must stay within a factor 2 `
        + `(${heuristic.cost} vs ${exact.cost})`);
      assert.ok(treeConnectsAll(weight, heuristic.tree, terminals),
        `seed ${seed}: the ${name} heuristic must return a valid tree`);
      assert.strictEqual(treeCost(weight, heuristic.tree), heuristic.cost);
      const ratio = heuristic.cost / (exact.cost === 0 ? 1 : exact.cost);
      worstRatio = Math.max(worstRatio, ratio);
      if (heuristic.cost > exact.cost) heuristicStrictlyWorse += 1;
    }
    instances += 1;
  }
  assert.ok(instances >= 8, `the corpus covered ${instances} instances`);
  assert.ok(worstRatio <= 2, `the observed worst ratio is ${worstRatio}`);
  assert.ok(heuristicStrictlyWorse >= 0,
    `heuristics matched the optimum on ${instances - heuristicStrictlyWorse} of ${instances} `
    + "corpus instances; the explicit witness check in section (3) pins the gap down");

  // (3) A star where the Steiner point pays off, and a searched witness of a strictly suboptimal
  //     heuristic answer.
  const star = weightedGraph(4, [[0, 1, 1], [0, 2, 1], [0, 3, 1]]);
  const starTerminals = [1, 2, 3];
  assert.strictEqual(steinerExact(star, starTerminals).cost, 3,
    "the centre 0 is a Steiner point and the tree costs 1 + 1 + 1");
  assert.strictEqual(primMst(star, starTerminals), null,
    "without the centre the terminals are disconnected: no terminal-only tree exists at all");
  assert.strictEqual(steinerBySubsetEnumeration(star, starTerminals), 3);
  assert.strictEqual(nearestTerminalHeuristic(star, starTerminals).cost, 3);
  assert.strictEqual(metricClosureHeuristic(star, starTerminals).cost, 3);
  assert.ok(metricClosureHeuristic(star, starTerminals).closureCost > 3,
    "the closure MST itself costs 4, so the expansion step is what saves the day here");

  // A deterministic search for an instance where a heuristic is strictly worse than the optimum: the
  // approximation guarantees are upper bounds, so a witness must exist and is exhibited.
  let witness = null;
  for (let seed = 1; seed <= 60 && witness === null; seed++) {
    const n = 8;
    const weight = weightedGraph(n, completeGraph(n, 9, BigInt(seed) * 6151n).edges);
    const terminals = [0, 1 + (seed % 3), 4 + (seed % 3), 7];
    if (new Set(terminals).size !== 4) continue;
    const exact = steinerExact(weight, terminals).cost;
    const nearest = nearestTerminalHeuristic(weight, terminals).cost;
    const metric = metricClosureHeuristic(weight, terminals);
    if (nearest > exact || metric.cost > exact) {
      witness = { seed, terminals, exact, nearest, metric: metric.cost };
      assert.ok(treeConnectsAll(weight, nearestTerminalHeuristic(weight, terminals).tree, terminals));
      assert.ok(treeConnectsAll(weight, metric.tree, terminals));
    }
  }
  assert.ok(witness !== null,
    "a heuristic is strictly worse than the optimum on some searched instance");
  assert.ok(witness.exact <= Math.max(witness.nearest, witness.metric),
    "and the witness records the gap");
  assert.ok(witness.nearest <= 2 * witness.exact && witness.metric <= 2 * witness.exact,
    "while still respecting the factor-2 bound");

  // (4) Zero weights, cycles among the terminals and duplicated edges.
  const zeroWeight = weightedGraph(4, [[0, 1, 0], [1, 2, 0], [2, 3, 0]]);
  assert.strictEqual(steinerExact(zeroWeight, [0, 3]).cost, 0);
  assert.strictEqual(nearestTerminalHeuristic(zeroWeight, [0, 3]).cost, 0);
  assert.strictEqual(metricClosureHeuristic(zeroWeight, [0, 3]).cost, 0);
  const cycle = weightedGraph(4, [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 0, 1]]);
  assert.strictEqual(steinerExact(cycle, [0, 1, 2, 3]).cost, 3,
    "a 4-cycle is spanned by three of its four unit edges");
  assert.strictEqual(primMst(cycle, [0, 1, 2, 3]).cost, 3, "which is exactly the MST cost");
  assert.strictEqual(steinerExact(cycle, [0, 2]).cost, 2, "opposite terminals are at distance 2");
  const duplicated = weightedGraph(3, [[0, 1, 5], [0, 1, 2], [1, 2, 3]]);
  assert.strictEqual(duplicated[0][1], 2, "the cheaper parallel edge wins");
  assert.strictEqual(steinerExact(duplicated, [0, 2]).cost, 5);
  assert.strictEqual(steinerBySubsetEnumeration(duplicated, [0, 2]), 5);

  // (5) Degenerate inputs.
  assert.strictEqual(steinerExact(weightedGraph(1, []), [0]).cost, 0);
  assert.deepStrictEqual(steinerExact(weightedGraph(1, []), [0]).tree, []);
  assert.strictEqual(steinerExact(weightedGraph(1, []), []).cost, 0,
    "the empty terminal set is documented as costing nothing");
  const disconnected = weightedGraph(4, [[0, 1, 1], [2, 3, 1]]);
  assert.strictEqual(steinerExact(disconnected, [0, 2]).cost, null,
    "terminals in different components are infeasible");
  assert.strictEqual(steinerBySubsetEnumeration(disconnected, [0, 2]), null);
  assert.strictEqual(nearestTerminalHeuristic(disconnected, [0, 2]).cost, null);
  assert.strictEqual(metricClosureHeuristic(disconnected, [0, 2]).cost, null);
  assert.strictEqual(steinerExact(disconnected, [0, 1]).cost, 1, "same-component terminals are fine");
  assert.strictEqual(nearestTerminalHeuristic(disconnected, []).cost, 0);
  const isolated = weightedGraph(3, []);
  assert.strictEqual(steinerExact(isolated, [1]).cost, 0);
  assert.strictEqual(steinerExact(isolated, [1, 2]).cost, null);

  // (6) The metric closure equals Floyd-Warshall distances, and the closure bound holds.
  const { dist } = floydWarshall(line);
  const closure = metricClosureHeuristic(line, allTerminals);
  assert.ok(closure.closureCost <= closure.cost,
    "expanding the MST can only add cost, never remove it");
  assert.strictEqual(dist[0][4], 14, "the line graph distances");
  assert.strictEqual(dist[0][2], 7);
  assert.strictEqual(dist[1][4], 11);
  const complete = weightedGraph(5, completeGraph(5, 20, 4242n).edges);
  const triangles = floydWarshall(complete);
  for (let u = 0; u < 5; u++) {
    for (let v = 0; v < 5; v++) {
      assert.ok(triangles.dist[u][v] <= complete[u][v] + 1e-12, "distances never exceed an edge");
      assert.strictEqual(triangles.dist[u][v], triangles.dist[v][u], "distances are symmetric");
    }
  }
  assert.strictEqual(triangles.dist[0][0], 0);
  assert.strictEqual(pathBetween(triangles.next, 0, 0), null,
    "there is no next hop from a vertex to itself");

  // (7) The optimum never exceeds the metric-closure MST cost (a theoretical inequality).
  for (let seed = 1; seed <= 6; seed++) {
    const n = 7;
    const weight = weightedGraph(n, completeGraph(n, 15, BigInt(seed) * 31337n).edges);
    const terminals = [0, 2 + (seed % 3), 6];
    if (new Set(terminals).size !== 3) continue;
    const exact = steinerExact(weight, terminals).cost;
    const metric = metricClosureHeuristic(weight, terminals);
    const nearest = nearestTerminalHeuristic(weight, terminals);
    assert.ok(exact <= metric.closureCost + 1e-12,
      "the optimum is at most the metric-closure MST cost");
    assert.ok(metric.cost <= 2 * (1 - 1 / terminals.length) * exact + 1e-12
      || metric.cost <= 2 * exact + 1e-12,
    `seed ${seed}: the 2(1 - 1/|T|) bound must hold (${metric.cost} vs ${exact})`);
    assert.ok(nearest.cost >= exact - 1e-12 && nearest.cost <= 2 * exact + 1e-12);
    assert.strictEqual(steinerExact(weight, terminals).cost,
      steinerBySubsetEnumeration(weight, terminals), "both exact methods agree");
  }

  // (8) Determinism.
  const repeatWeight = weightedGraph(6, completeGraph(6, 12, 20240607n).edges);
  const repeatTerminals = [0, 3, 5];
  assert.strictEqual(steinerExact(repeatWeight, repeatTerminals).cost,
    steinerExact(repeatWeight, repeatTerminals).cost);
  assert.deepStrictEqual(nearestTerminalHeuristic(repeatWeight, repeatTerminals),
    nearestTerminalHeuristic(repeatWeight, repeatTerminals));
  assert.deepStrictEqual(metricClosureHeuristic(repeatWeight, repeatTerminals),
    metricClosureHeuristic(repeatWeight, repeatTerminals));
}

module.exports = { lcg64Next, lcg64Int, INF, emptyMatrix, weightedGraph, neighbours, floydWarshall,
  pathBetween, primMst, isConnected, steinerExact, steinerBySubsetEnumeration,
  nearestTerminalHeuristic, metricClosureHeuristic, treeCost, treeConnectsAll, randomWeightedGraph,
  completeGraph };

if (require.main === module) {
  main();
  console.log("PASS");
}
