/** Task 282 - Bridge edges via Tarjan's disc/low values. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

/**
 * One DFS pass: an edge (u, v) is a bridge when the subtree below v cannot reach u or
 * higher (low[v] > disc[u]). Parallel edges are handled by skipping only the exact edge
 * id we arrived on. Returns bridges as canonical sorted (min, max) pairs, cross-checked
 * by deleting each edge and recounting connected components.
 */
function findBridges(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  edges.forEach(([a, b], eid) => {
    if (a === b) return;                         // self loops are never bridges
    adjacency[a].push([b, eid]);
    adjacency[b].push([a, eid]);
  });

  const disc = new Array(n).fill(-1);
  const low = new Array(n).fill(0);
  let timer = 0;
  const bridges = [];

  const dfs = (u, parentEdge) => {
    disc[u] = low[u] = timer++;
    for (const [v, eid] of adjacency[u]) {
      if (eid === parentEdge) continue;          // skip only the arrival edge itself
      if (disc[v] !== -1) {
        low[u] = Math.min(low[u], disc[v]);      // back edge to an ancestor
      } else {
        dfs(v, eid);
        low[u] = Math.min(low[u], low[v]);
        if (low[v] > disc[u]) {                  // v's subtree cannot escape above u
          bridges.push([Math.min(u, v), Math.max(u, v)]);
        }
      }
    }
  };

  for (let v = 0; v < n; v++) if (disc[v] === -1) dfs(v, -1);
  bridges.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  return bridges;
}

function countComponents(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    if (a === b) continue;
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const seen = new Array(n).fill(false);
  let total = 0;
  for (let s = 0; s < n; s++) {
    if (seen[s]) continue;
    total++;
    const stack = [s];
    seen[s] = true;
    while (stack.length) {
      const x = stack.pop();
      for (const y of adjacency[x]) {
        if (!seen[y]) { seen[y] = true; stack.push(y); }
      }
    }
  }
  return total;
}

/** Oracle: deleting a bridge increases the number of connected components. */
function bruteForceBridges(n, edges) {
  const base = countComponents(n, edges);
  const found = new Set();
  for (let i = 0; i < edges.length; i++) {
    const without = edges.slice(0, i).concat(edges.slice(i + 1));
    if (countComponents(n, without) > base) {
      const [a, b] = edges[i];
      found.add(`${Math.min(a, b)}-${Math.max(a, b)}`);
    }
  }
  return [...found]
    .map((k) => k.split("-").map(Number))
    .sort((p, q) => p[0] - q[0] || p[1] - q[1]);
}

/** Simple graphs: each unordered pair {u, v} considered once. */
function randomUndirectedGraphs(n, count, seed = 13579) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    const edges = [];
    for (let u = 0; u < n; u++) {
      for (let v = u + 1; v < n; v++) {
        seed = (seed * 75 + 74) % 65537;
        if (seed % 3 === 0) edges.push([u, v]);
      }
    }
    graphs.push(edges);
  }
  return graphs;
}

/** Multigraphs: (u, v) and (v, u) may both appear as parallel edges. */
function randomMultigraphs(n, count, seed = 13579) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    const edges = [];
    for (let u = 0; u < n; u++) {
      for (let v = 0; v < n; v++) {
        if (u === v) continue;
        seed = (seed * 75 + 74) % 65537;
        if (seed % 3 === 0) edges.push([u, v]);
      }
    }
    graphs.push(edges);
  }
  return graphs;
}

const key = (pairs) => pairs.map((p) => p.join("-")).join("|");

module.exports = { findBridges, countComponents, bruteForceBridges, randomUndirectedGraphs, randomMultigraphs, key };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(key(findBridges(3, [[0, 1], [1, 2]])), "0-1|1-2");                    // path: 2 bridges
  assert.strictEqual(key(findBridges(3, [[0, 1], [1, 2], [2, 0]])), "");                   // triangle
  assert.strictEqual(key(findBridges(4, [[0, 1], [1, 2], [2, 3], [3, 0]])), "");           // square
  assert.strictEqual(key(findBridges(5, [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4]])), "2-4"); // square + tail
  assert.strictEqual(key(findBridges(5, [[0, 1], [1, 2], [3, 4]])), "0-1|1-2|3-4");         // disconnected
  assert.strictEqual(key(findBridges(3, [[0, 1], [0, 1], [1, 2]])), "1-2");                 // parallel edges
  assert.strictEqual(key(findBridges(2, [[0, 0], [0, 1]])), "0-1");                         // self loop ignored
  assert.strictEqual(key(findBridges(3, [[0, 1]])), "0-1");                                 // isolated vertex

  for (const [label, graphs] of [
    ["simple", randomUndirectedGraphs(6, 120)],
    ["multigraph", randomMultigraphs(6, 120)],
  ]) {
    let seenBridge = false, seenNoBridge = false;
    let trial = 0;
    for (const edges of graphs) {
      const bridges = findBridges(6, edges);
      assert.strictEqual(key(bridges), key(bruteForceBridges(6, edges)), `${label} ${trial}`);
      const edgeSet = new Set(edges.filter(([a, b]) => a !== b).map(([a, b]) => `${Math.min(a, b)}-${Math.max(a, b)}`));
      for (const [a, b] of bridges) assert.ok(edgeSet.has(`${a}-${b}`), `${label} edge ${trial}`);
      if (bridges.length) seenBridge = true; else seenNoBridge = true;
      trial++;
    }
    assert.ok(seenBridge && seenNoBridge, `${label} coverage`);
  }

  console.log("PASS");
}
