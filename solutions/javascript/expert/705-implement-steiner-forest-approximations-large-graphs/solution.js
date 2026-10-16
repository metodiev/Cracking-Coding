/** Task 705 - Implement Steiner forest approximations large graphs. (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
 *
 * A Steiner forest connects every demand pair (s_i, t_i) of a weighted graph with the cheapest
 * subgraph - NP-hard, so this file implements the standard approximations and measures them
 * against an exhaustive optimum:
 *   - Steiner tree, MST heuristic: build the metric closure of the terminal set (all-pairs
 *     distances by Dijkstra), take its minimum spanning tree and expand every metric edge into a
 *     shortest path. The cost is at most 2(1 - 1/|L|) times the optimum;
 *   - Steiner tree, shortest-path-to-tree heuristic (Takahashi-Matsuyama): grow one tree by always
 *     attaching the terminal that is cheapest to reach, with the same 2(1 - 1/|L|) bound;
 *   - Steiner forest: process the pairs in order, contracting the components found so far, and
 *     for every still-separated pair add the shortest path between its two components. The result
 *     is feasible by construction; its cost is compared with the exhaustive optimum.
 *
 * What is simplified: a single connected component per instance (a pair that cannot be joined
 * makes the instance infeasible and is reported as such), no terminal-metric preprocessing beyond
 * Dijkstra, and the exhaustive oracle only runs on graphs with at most 12 edges.
 *
 * Invariant verified: every produced subgraph is a set of real graph edges whose weight equals
 * the reported cost, it connects every demand pair, and its cost is at least the brute-force
 * optimum and at most twice it - the classical approximation factor.
 */
"use strict";
const { strictEqual, deepStrictEqual, ok, throws } = require("node:assert");
/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

function edgeKey(u, v) { return u < v ? `${u}-${v}` : `${v}-${u}`; }
/** Adjacency list of {to, weight, key} entries; undirected. */
function buildAdjacency(size, edges) {
  const adjacency = Array.from({ length: size }, () => []);
  for (const edge of edges) {
    adjacency[edge.u].push({ to: edge.v, weight: edge.weight, key: edgeKey(edge.u, edge.v) });
    adjacency[edge.v].push({ to: edge.u, weight: edge.weight, key: edgeKey(edge.u, edge.v) });
  }
  return adjacency;
}
/** Dijkstra from several sources at once; returns distances and the edge used to enter a vertex. */
function dijkstra(size, adjacency, sources) {
  const dist = new Array(size).fill(Infinity);
  const parent = new Array(size).fill(null);
  const done = new Array(size).fill(false);
  for (const source of sources) dist[source] = 0;
  for (;;) {
    let best = -1;
    for (let v = 0; v < size; v++) if (!done[v] && dist[v] < Infinity && (best === -1 || dist[v] < dist[best])) best = v;
    if (best === -1) break;
    done[best] = true;
    for (const step of adjacency[best]) {
      if (dist[step.to] > dist[best] + step.weight) {
        dist[step.to] = dist[best] + step.weight;
        parent[step.to] = { from: best, key: step.key };
      }
    }
  }
  return { dist, parent };
}
/** Walks `parent` pointers from `target` back to a source, collecting the edges on the way. */
function tracePathKeys(parent, target) {
  const keys = new Set();
  for (let at = target; parent[at] !== null; at = parent[at].from) keys.add(parent[at].key);
  return keys;
}

function costOf(edges, keys) {
  let total = 0;
  for (const edge of edges) if (keys.has(edgeKey(edge.u, edge.v))) total += edge.weight;
  return total;
}
/** Minimum spanning tree weight of the complete graph given by a distance matrix. */
function minimumSpanningTree(size, vertices, distance) {
  const inTree = new Set([vertices[0]]);
  let total = 0;
  const mstEdges = [];
  while (inTree.size < vertices.length) {
    let best = null;
    for (const a of inTree) {
      for (const b of vertices) {
        if (inTree.has(b) || distance[a][b] === Infinity) continue;
        if (best === null || distance[a][b] < best.weight) best = { a, b, weight: distance[a][b] };
      }
    }
    if (best === null) return null; // the terminals do not span one component
    inTree.add(best.b);
    total += best.weight;
    mstEdges.push(best);
  }
  return { total, edges: mstEdges };
}
/** Steiner tree by the MST-of-metric-closure heuristic; returns {keys, cost} or null. */
function steinerTreeMst(size, edges, terminals) {
  if (terminals.length <= 1) return { keys: new Set(), cost: 0 };
  const adjacency = buildAdjacency(size, edges);
  const distance = Array.from({ length: size }, () => new Array(size).fill(Infinity));
  const parents = new Array(size).fill(null);
  for (const terminal of terminals) {
    const { dist, parent } = dijkstra(size, adjacency, [terminal]);
    distance[terminal] = dist;
    parents[terminal] = parent;
  }
  const spanning = minimumSpanningTree(size, [...new Set(terminals)], distance);
  if (spanning === null) return null;
  const keys = new Set();
  for (const metricEdge of spanning.edges) {
    for (let at = metricEdge.b; at !== metricEdge.a;) {
      keys.add(parents[metricEdge.a][at].key);
      at = parents[metricEdge.a][at].from;
    }
  }
  return { keys, cost: costOf(edges, keys) };
}
/** Steiner tree by always attaching the terminal closest to the growing tree. */
function steinerTreeGreedy(size, edges, terminals) {
  const unique = [...new Set(terminals)];
  if (unique.length <= 1) return { keys: new Set(), cost: 0 };
  const adjacency = buildAdjacency(size, edges);
  const needed = new Set(unique.slice(1));
  const tree = new Set([unique[0]]);
  const keys = new Set();
  while (needed.size > 0) {
    const { dist, parent } = dijkstra(size, adjacency, [...tree]);
    let chosen = null;
    for (const terminal of needed) {
      if (dist[terminal] < Infinity && (chosen === null || dist[terminal] < dist[chosen])) chosen = terminal;
    }
    if (chosen === null) return null; // some terminal is unreachable
    needed.delete(chosen);
    for (let at = chosen; parent[at] !== null; at = parent[at].from) {
      keys.add(parent[at].key);
      tree.add(at);
    }
    tree.add(chosen);
  }
  return { keys, cost: costOf(edges, keys) };
}
/** Steiner forest: add a shortest component-to-component path for every pair still apart. */
function steinerForest(size, edges, pairs) {
  const adjacency = buildAdjacency(size, edges);
  const chosen = new Set();
  const connectedByChosen = () => {
    const labels = new Array(size).fill(-1);
    let next = 0;
    for (let v = 0; v < size; v++) {
      if (labels[v] !== -1) continue;
      const queue = [v];
      labels[v] = next;
      while (queue.length > 0) {
        const current = queue.pop();
        for (const step of adjacency[current]) {
          if (chosen.has(step.key) && labels[step.to] === -1) {
            labels[step.to] = next;
            queue.push(step.to);
          }
        }
      }
      next++;
    }
    return labels;
  };
  for (const [from, to] of pairs) {
    if (from === to) continue;
    const labels = connectedByChosen();
    if (labels[from] === labels[to]) continue;
    // multi-source Dijkstra from `from`'s component, stopping when `to`'s component is reached
    const sources = labels.map((label, v) => (label === labels[from] ? v : -1)).filter((v) => v !== -1);
    const { dist, parent } = dijkstra(size, adjacency, sources);
    if (dist[to] === Infinity) return null;
    for (let at = to; parent[at] !== null; at = parent[at].from) chosen.add(parent[at].key);
  }
  return { keys: chosen, cost: costOf(edges, chosen) };
}
/** Every demand pair joined by the chosen edges? */
function joinsAllPairs(size, edges, pairs, keys) {
  const adjacency = buildAdjacency(size, [...edges.filter((e) => keys.has(edgeKey(e.u, e.v)))]);
  const labels = new Array(size).fill(-1);
  let next = 0;
  for (let v = 0; v < size; v++) {
    if (labels[v] !== -1) continue;
    const queue = [v];
    labels[v] = next;
    while (queue.length > 0) {
      const current = queue.pop();
      for (const step of adjacency[current]) if (labels[step.to] === -1) { labels[step.to] = next; queue.push(step.to); }
    }
    next++;
  }
  return pairs.every(([a, b]) => labels[a] === labels[b]);
}
/** Exhaustive Steiner tree optimum: for every set of extra vertices, the MST over its closure. */
function bruteSteinerTree(size, edges, terminals) {
  const extra = [];
  for (let v = 0; v < size; v++) if (!terminals.includes(v)) extra.push(v);
  const adjacency = buildAdjacency(size, edges);
  const distance = Array.from({ length: size }, () => dijkstra(size, adjacency, [0]).dist);
  for (let v = 0; v < size; v++) distance[v] = dijkstra(size, adjacency, [v]).dist;
  let best = Infinity;
  for (let mask = 0; mask < (1 << extra.length); mask++) {
    const chosen = [...new Set(terminals)];
    for (let i = 0; i < extra.length; i++) if (mask & (1 << i)) chosen.push(extra[i]);
    if (distance[chosen[0]][chosen[1]] === Infinity && chosen.length > 1) continue;
    const spanning = minimumSpanningTree(size, chosen, distance);
    if (spanning !== null && spanning.total < best) best = spanning.total;
  }
  return best;
}
/** Exhaustive Steiner forest optimum: every subset of edges (graphs here have <= 12 edges). */
function bruteSteinerForest(size, edges, pairs) {
  let best = Infinity;
  for (let mask = 0; mask < (1 << edges.length); mask++) {
    const keys = new Set();
    let cost = 0;
    for (let i = 0; i < edges.length; i++) {
      if (mask & (1 << i)) {
        keys.add(edgeKey(edges[i].u, edges[i].v));
        cost += edges[i].weight;
      }
    }
    if (cost >= best) continue;
    if (joinsAllPairs(size, edges, pairs, keys)) best = cost;
  }
  return best;
}

module.exports = {
  makeRng, edgeKey, buildAdjacency, dijkstra, minimumSpanningTree, steinerTreeMst,
  steinerTreeGreedy, steinerForest, joinsAllPairs, bruteSteinerTree, bruteSteinerForest,
};

if (require.main === module) {
  // -- degenerate instances --------------------------------------------------------------------
  const tinyGraph = { size: 3, edges: [{ u: 0, v: 1, weight: 2 }, { u: 1, v: 2, weight: 3 }] };
  deepStrictEqual(steinerTreeMst(3, tinyGraph.edges, []), { keys: new Set(), cost: 0 });
  deepStrictEqual(steinerTreeMst(3, tinyGraph.edges, [1]), { keys: new Set(), cost: 0 });
  deepStrictEqual(steinerTreeGreedy(3, tinyGraph.edges, [0]), { keys: new Set(), cost: 0 });
  const adjacent = steinerTreeMst(3, tinyGraph.edges, [0, 1]);
  strictEqual(adjacent.cost, 2, "two adjacent terminals pay for their edge");
  deepStrictEqual([...adjacent.keys], ["0-1"]);
  strictEqual(steinerForest(3, tinyGraph.edges, [[2, 2]]).cost, 0, "an empty demand is free");
  deepStrictEqual(steinerForest(3, tinyGraph.edges, [[0, 1]]).keys, new Set(["0-1"]));

  // -- hand-checked star: centre 0 with leaves 1..3, plus a cheap shortcut 1-2 ------------------
  const starEdges = [
    { u: 0, v: 1, weight: 4 }, { u: 0, v: 2, weight: 4 }, { u: 0, v: 3, weight: 4 },
    { u: 1, v: 2, weight: 2 },
  ];
  // The terminals must end up pairwise connected, so 3 still needs a spoke through 0: 2+4+4.
  strictEqual(steinerTreeMst(4, starEdges, [1, 2, 3]).cost, 10, "shortcut plus a spoke via the centre");
  strictEqual(steinerTreeGreedy(4, starEdges, [1, 2, 3]).cost, 10);
  strictEqual(bruteSteinerTree(4, starEdges, [1, 2, 3]), 10, "the exhaustive optimum agrees");
  strictEqual(steinerForest(4, starEdges, [[1, 2], [2, 3]]).cost, 10);
  strictEqual(bruteSteinerForest(4, starEdges, [[1, 2], [2, 3]]), 10);

  // -- infeasible demand ------------------------------------------------------------------------
  const split = [{ u: 0, v: 1, weight: 1 }, { u: 2, v: 3, weight: 1 }];
  strictEqual(steinerForest(4, split, [[0, 2]]), null, "no path between the components");
  strictEqual(bruteSteinerForest(4, split, [[0, 2]]), Infinity);
  strictEqual(steinerTreeMst(4, split, [0, 2]), null);

  // -- randomized tiny graphs against the exhaustive optimum -------------------------------------
  const rng = makeRng(705);
  let treeRatioWorst = 0, forestRatioWorst = 0, feasibleCases = 0;
  for (let trial = 0; trial < 120; trial++) {
    const size = 4 + rng(3);
    const allPairs = [];
    for (let u = 0; u < size; u++) for (let v = u + 1; v < size; v++) allPairs.push([u, v]);
    const edges = [];
    for (const [u, v] of allPairs) if (rng(4) > 0) edges.push({ u, v, weight: 1 + rng(6) });
    if (edges.length === 0 || edges.length > 9) continue;
    const terminals = [...new Set([rng(size), rng(size), rng(size)])];
    const optimum = bruteSteinerTree(size, edges, terminals);
    const mstResult = steinerTreeMst(size, edges, terminals);
    const greedyResult = steinerTreeGreedy(size, edges, terminals);
    if (optimum === Infinity) {
      ok(mstResult === null && greedyResult === null, "unreachable terminals must be reported");
      continue;
    }
    feasibleCases++;
    ok(joinsAllPairs(size, edges, terminals.map((t) => [t, terminals[0]]), mstResult.keys), "MST heuristic connects");
    ok(joinsAllPairs(size, edges, terminals.map((t) => [t, terminals[0]]), greedyResult.keys), "greedy connects");
    ok(mstResult.cost >= optimum && greedyResult.cost >= optimum, "no heuristic may beat the optimum");
    if (optimum === 0) { // every terminal collapsed into one vertex
      strictEqual(mstResult.cost, 0);
      strictEqual(greedyResult.cost, 0);
      continue;
    }
    const mstRatio = mstResult.cost / optimum;
    const greedyRatio = greedyResult.cost / optimum;
    ok(mstRatio <= 2 && greedyRatio <= 2, `2-approximation violated: ${mstRatio}/${greedyRatio}`);
    treeRatioWorst = Math.max(treeRatioWorst, mstRatio, greedyRatio);

    const pairCount = 1 + rng(3);
    const pairs = Array.from({ length: pairCount }, () => [rng(size), rng(size)]);
    const forestOptimum = bruteSteinerForest(size, edges, pairs);
    const forest = steinerForest(size, edges, pairs);
    if (forestOptimum === Infinity) {
      ok(forest === null, "an infeasible forest instance must be reported");
      continue;
    }
    if (forestOptimum === 0) {
      strictEqual(forest.cost, 0, "every pair is a single vertex, so nothing must be added");
      continue;
    }
    ok(forest !== null && joinsAllPairs(size, edges, pairs, forest.keys), "the forest must be feasible");
    strictEqual(forest.cost, costOf(edges, forest.keys), "reported cost must equal the edge weights");
    ok(forest.cost >= forestOptimum, "the forest cannot beat the optimum");
    const forestRatio = forest.cost / forestOptimum;
    ok(forestRatio <= 2, `the forest heuristic left the 2-approximation: ${forestRatio}`);
    forestRatioWorst = Math.max(forestRatioWorst, forestRatio);
  }
  ok(feasibleCases > 60, `the sweep must exercise feasible instances (${feasibleCases})`);

  // -- a bigger structured instance: a 5x5 grid with four corner terminals ----------------------
  const gridEdges = [];
  const gridSize = 5;
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const id = row * gridSize + col;
      if (col + 1 < gridSize) gridEdges.push({ u: id, v: id + 1, weight: 1 });
      if (row + 1 < gridSize) gridEdges.push({ u: id, v: id + gridSize, weight: 1 });
    }
  }
  const corners = [0, 4, 20, 24];
  const gridTree = steinerTreeMst(25, gridEdges, corners);
  // The cheapest Steiner tree is the H shape through the middle column: 2+4+2+2+2 = 12 edges.
  strictEqual(gridTree.cost, 12, "four grid corners need 12 unit edges");
  ok(joinsAllPairs(25, gridEdges, [[0, 24]], gridTree.keys));
  const gridForest = steinerForest(25, gridEdges, [[0, 4], [20, 24], [0, 20]]);
  strictEqual(gridForest.cost, 12, "two rows plus one column, edge-disjoint: 12");
  strictEqual(bruteSteinerForest(4, starEdges, [[1, 2]]), 2, "a single pair just takes the cheapest path");

  // -- determinism ------------------------------------------------------------------------------
  function replay(seed) {
    const r = makeRng(seed);
    const edges = Array.from({ length: 8 }, () => ({ u: r(5), v: r(5), weight: 1 + r(5) })).filter((e) => e.u !== e.v);
    const terminals = [r(5), r(5), r(5)];
    const mst = steinerTreeMst(5, edges, terminals);
    const forest = steinerForest(5, edges, [[r(5), r(5)], [r(5), r(5)]]);
    return `${mst === null ? "null" : mst.cost}|${forest === null ? "null" : [...forest.keys].sort().join(",")}`;
  }
  strictEqual(replay(9), replay(9), "identical seeds must replay identically");

  console.log(`random instances=${feasibleCases} worst MST/greedy ratio=${treeRatioWorst.toFixed(3)}`);
  console.log(`worst forest ratio=${forestRatioWorst.toFixed(3)} grid tree=${gridTree.cost} grid forest=${gridForest.cost}`);
  console.log("PASS");
}
