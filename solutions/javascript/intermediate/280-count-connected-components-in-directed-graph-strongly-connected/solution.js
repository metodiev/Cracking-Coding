/** Task 280 - Count strongly connected components of a directed graph. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function stronglyConnectedComponents(n, edges) {
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

const countComponents = (n, edges) => stronglyConnectedComponents(n, edges).length;

function bruteForceComponents(n, edges) {
  const reach = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j));
  for (const [a, b] of edges) reach[a][b] = true;
  for (let k = 0; k < n; k++)
    for (let i = 0; i < n; i++)
      if (reach[i][k])
        for (let j = 0; j < n; j++) if (reach[k][j]) reach[i][j] = true;

  const groups = new Map();
  for (let v = 0; v < n; v++) {
    const key = [];
    for (let i = 0; i < n; i++) if (reach[v][i] && reach[i][v]) key.push(i);
    const k = key.join(",");
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(v);
  }
  return [...groups.values()].map((g) => g.sort((a, b) => a - b)).sort((a, b) => a.join(",").localeCompare(b.join(",")));
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

module.exports = { stronglyConnectedComponents, countComponents, bruteForceComponents, randomGraphs };

if (require.main === module) {
  const assert = require("node:assert");
  const key = (components) => components.map((g) => g.join(",")).sort().join("|");
  assert.strictEqual(countComponents(3, [[0, 1], [1, 2], [2, 0]]), 1);
  assert.strictEqual(countComponents(4, [[0, 1], [1, 2], [2, 0], [3, 3]]), 2);
  assert.strictEqual(countComponents(3, [[0, 1], [1, 2]]), 3);
  assert.strictEqual(countComponents(3, []), 3);
  assert.strictEqual(countComponents(3, [[0, 1], [1, 0], [1, 2]]), 2);
  assert.strictEqual(countComponents(2, [[0, 1]]), 2);

  const classic = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 3], [4, 5]];
  assert.strictEqual(countComponents(6, classic), 3);
  assert.strictEqual(key(stronglyConnectedComponents(6, classic)), "0,1,2|3,4|5");

  let trial = 0;
  for (const edges of randomGraphs(6, 120)) {
    assert.strictEqual(key(stronglyConnectedComponents(6, edges)), key(bruteForceComponents(6, edges)), `trial ${trial}`);
    assert.ok(countComponents(6, edges) <= 6);
    trial++;
  }
  console.log("PASS");
}
