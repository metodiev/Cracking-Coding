/** Task 106 - Topological sort using DFS (DAG). (Tier: Beginner, CTCI Ch 4) */
"use strict";

const WHITE = 0;
const GRAY = 1;
const BLACK = 2;

function topoSort(graph) {
  const color = new Map(Object.keys(graph).map((k) => [Number(k), WHITE]));
  const order = [];

  function visit(v) {
    color.set(v, GRAY);
    for (const w of graph[v] || []) {
      const c = color.has(w) ? color.get(w) : WHITE;
      if (c === GRAY) return false;
      if (c === WHITE && !visit(w)) return false;
    }
    color.set(v, BLACK);
    order.push(v);
    return true;
  }

  for (const v of Object.keys(graph).map(Number)) {
    if (color.get(v) === WHITE && !visit(v)) return null;
  }
  order.reverse();
  return order;
}

function isValidTopo(graph, order) {
  if (order === null || order.length !== Object.keys(graph).length) return false;
  const index = new Map(order.map((v, i) => [v, i]));
  for (const [key, successors] of Object.entries(graph)) {
    const ui = index.get(Number(key));
    if (ui === undefined) return false;
    for (const w of successors) {
      const wi = index.get(w);
      if (wi === undefined || ui >= wi) return false;
    }
  }
  return true;
}

module.exports = { topoSort };

if (require.main === module) {
  const assert = require("node:assert");
  const courses = { 0: [1, 2], 1: [3], 2: [3], 3: [] };
  assert.deepStrictEqual(topoSort(courses), [0, 2, 1, 3]);
  assert.strictEqual(isValidTopo(courses, topoSort(courses)), true);

  assert.strictEqual(topoSort({ 0: [1], 1: [2], 2: [0] }), null);
  assert.strictEqual(topoSort({ 0: [0] }), null);
  assert.deepStrictEqual(topoSort({}), []);
  assert.deepStrictEqual(topoSort({ 9: [] }), [9]);

  const forest = { 0: [1], 1: [], 2: [3], 3: [] };
  const order2 = topoSort(forest);
  assert.strictEqual(isValidTopo(forest, order2), true);
  assert.strictEqual(order2.length, 4);

  const chained = { 0: [], 1: [0], 2: [1], 3: [2] };
  const order3 = topoSort(chained);
  assert.deepStrictEqual(order3, [3, 2, 1, 0]);
  assert.strictEqual(isValidTopo(chained, order3), true);
  console.log("PASS");
}
