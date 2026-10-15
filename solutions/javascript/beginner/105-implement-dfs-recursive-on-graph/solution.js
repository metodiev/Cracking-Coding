/** Task 105 - Implement DFS (recursive) on graph. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function dfsOrder(graph, source) {
  if (!(source in graph)) return [];
  const visited = new Set();
  const order = [];
  dfs(graph, source, visited, order);
  return order;
}

function dfsAll(graph) {
  const visited = new Set();
  const order = [];
  for (const v of Object.keys(graph).map(Number).sort((a, b) => a - b)) {
    if (!visited.has(v)) dfs(graph, v, visited, order);
  }
  return order;
}

function dfs(graph, v, visited, order) {
  visited.add(v);
  order.push(v);
  for (const w of graph[v] || []) {
    if (!visited.has(w)) dfs(graph, w, visited, order);
  }
}

module.exports = { dfsOrder, dfsAll };

if (require.main === module) {
  const assert = require("node:assert");
  const cyclic = { 0: [1, 2], 1: [2], 2: [0, 3], 3: [3] };
  assert.deepStrictEqual(dfsOrder(cyclic, 0), [0, 1, 2, 3]);

  const twoCycles = { 0: [1], 1: [0], 2: [3], 3: [2] };
  assert.deepStrictEqual(dfsOrder(twoCycles, 0), [0, 1]);
  assert.deepStrictEqual(dfsAll(twoCycles), [0, 1, 2, 3]);

  assert.deepStrictEqual(dfsOrder({ 5: [] }, 5), [5]);
  assert.deepStrictEqual(dfsOrder({}, 0), []);
  assert.deepStrictEqual(dfsOrder(cyclic, 99), []);
  assert.deepStrictEqual(dfsAll({}), []);
  assert.deepStrictEqual(dfsAll({ 7: [] }), [7]);

  const chain = {};
  for (let i = 0; i <= 50; i++) chain[i] = i < 50 ? [i + 1] : [];
  assert.strictEqual(dfsOrder(chain, 0).length, 51);
  console.log("PASS");
}
