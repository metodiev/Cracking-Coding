/** Task 104 - Implement BFS on adjacency list graph. (Tier: Beginner, CTCI Ch 4) */
"use strict";

function bfsOrder(graph, source) {
  if (!(source in graph)) return [];
  const seen = new Set([source]);
  const order = [];
  const queue = [source];
  for (let head = 0; head < queue.length; head++) {
    const v = queue[head];
    order.push(v);
    for (const w of graph[v] || []) {
      if (!seen.has(w)) {
        seen.add(w);
        queue.push(w);
      }
    }
  }
  return order;
}

function bfsDistances(graph, source) {
  const dist = new Map();
  if (!(source in graph)) return dist;
  dist.set(source, 0);
  const queue = [source];
  for (let head = 0; head < queue.length; head++) {
    const v = queue[head];
    for (const w of graph[v] || []) {
      if (!dist.has(w)) {
        dist.set(w, dist.get(v) + 1);
        queue.push(w);
      }
    }
  }
  return dist;
}

module.exports = { bfsOrder, bfsDistances };

if (require.main === module) {
  const assert = require("node:assert");
  const g = { 0: [1, 2], 1: [2], 2: [0, 3], 3: [3] };
  assert.deepStrictEqual(bfsOrder(g, 0), [0, 1, 2, 3]);
  assert.deepStrictEqual(Object.fromEntries(bfsDistances(g, 0)), { 0: 0, 1: 1, 2: 1, 3: 2 });

  const parts = { 0: [1], 1: [0], 2: [3], 3: [2], 4: [] };
  assert.deepStrictEqual(bfsOrder(parts, 0), [0, 1]);
  assert.deepStrictEqual(Object.fromEntries(bfsDistances(parts, 0)), { 0: 0, 1: 1 });
  assert.deepStrictEqual(bfsOrder(parts, 4), [4]);
  assert.deepStrictEqual(Object.fromEntries(bfsDistances(parts, 4)), { 4: 0 });

  assert.deepStrictEqual(bfsOrder(parts, 99), []);
  assert.deepStrictEqual(Object.fromEntries(bfsDistances(parts, 99)), {});
  assert.deepStrictEqual(bfsOrder({}, 0), []);
  assert.deepStrictEqual(Object.fromEntries(bfsDistances({}, 0)), {});

  assert.deepStrictEqual(bfsOrder({ 5: [6], 6: [5, 7], 7: [5] }, 5), [5, 6, 7]);
  console.log("PASS");
}
