/** Task 281 - Tarjan's strongly connected components in one DFS pass. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

/**
 * Single-pass Tarjan: one DFS with discovery indices; a vertex roots an SCC when its
 * low-link equals its index. Returns components (sorted vertex lists) ordered by
 * smallest vertex, and is cross-checked against the Kosaraju result (task 280).
 */
function tarjanScc(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) adjacency[a].push(b);

  const index = new Array(n).fill(-1);
  const low = new Array(n).fill(0);
  const onStack = new Array(n).fill(false);
  const stack = [];
  const components = [];
  let counter = 0;

  const strongconnect = (u) => {
    index[u] = low[u] = counter++;
    stack.push(u);
    onStack[u] = true;
    for (const v of adjacency[u]) {
      if (index[v] === -1) {
        strongconnect(v);                 // tree edge: descend
        low[u] = Math.min(low[u], low[v]);
      } else if (onStack[v]) {
        low[u] = Math.min(low[u], index[v]); // back/cross edge to an open SCC
      }
    }
    if (low[u] === index[u]) {            // u closes an SCC: pop until u
      const group = [];
      for (;;) {
        const w = stack.pop();
        onStack[w] = false;
        group.push(w);
        if (w === u) break;
      }
      group.sort((a, b) => a - b);
      components.push(group);
    }
  };

  for (let v = 0; v < n; v++) if (index[v] === -1) strongconnect(v);
  components.sort((a, b) => a[0] - b[0]);
  return components;
}

/** Task 280's two-pass algorithm, used here only as an independent oracle. */
function kosarajuScc(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  const reversed = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    reversed[b].push(a);
  }

  const visited = new Array(n).fill(false);
  const order = [];
  const dfsFinish = (u) => {
    visited[u] = true;
    for (const v of adjacency[u]) if (!visited[v]) dfsFinish(v);
    order.push(u);
  };
  for (let v = 0; v < n; v++) if (!visited[v]) dfsFinish(v);

  const component = new Array(n).fill(-1);
  const components = [];
  for (let i = order.length - 1; i >= 0; i--) {
    const u = order[i];
    if (component[u] !== -1) continue;
    const index = components.length;
    component[u] = index;
    const group = [u];
    const stack = [u];
    while (stack.length) {
      const x = stack.pop();
      for (const y of reversed[x]) {
        if (component[y] === -1) {
          component[y] = index;
          group.push(y);
          stack.push(y);
        }
      }
    }
    group.sort((a, b) => a - b);
    components.push(group);
  }
  return components;
}

function randomGraphs(n, count, seed = 13579) {
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

const canonical = (groups) =>
  groups.map((g) => [...g].sort((a, b) => a - b)).sort((a, b) => a[0] - b[0]);

module.exports = { tarjanScc, kosarajuScc, randomGraphs, canonical };

if (require.main === module) {
  const assert = require("node:assert");

  const classic = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 3], [4, 5]];
  assert.deepStrictEqual(tarjanScc(6, classic), [[0, 1, 2], [3, 4], [5]]);

  assert.deepStrictEqual(tarjanScc(3, [[0, 1], [1, 2], [2, 0]]), [[0, 1, 2]]);
  assert.deepStrictEqual(tarjanScc(3, [[0, 1], [1, 2]]), [[0], [1], [2]]); // DAG
  assert.deepStrictEqual(tarjanScc(3, []), [[0], [1], [2]]);
  assert.deepStrictEqual(tarjanScc(3, [[0, 1], [1, 0], [1, 2]]), [[0, 1], [2]]);
  assert.deepStrictEqual(tarjanScc(2, [[0, 1]]), [[0], [1]]);              // one-way edge
  assert.deepStrictEqual(tarjanScc(1, [[0, 0]]), [[0]]);                   // self loop

  let nontrivialSeen = false;
  let trial = 0;
  for (const edges of randomGraphs(6, 120)) {
    const components = tarjanScc(6, edges);
    assert.deepStrictEqual(components, canonical(kosarajuScc(6, edges)), `kosaraju match ${trial}`);
    const flat = components.flat().sort((a, b) => a - b);
    assert.deepStrictEqual(flat, [0, 1, 2, 3, 4, 5], `partition ${trial}`);
    if (components.some((g) => g.length > 1)) nontrivialSeen = true;
    trial++;
  }
  assert.ok(nontrivialSeen, "LCG graphs should contain real SCCs");

  console.log("PASS");
}
