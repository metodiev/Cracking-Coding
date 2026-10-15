/** Task 107 - Detect cycle in directed graph (DFS coloring). (Tier: Beginner, CTCI Ch 4) */
"use strict";

const WHITE = 0;
const GRAY = 1;
const BLACK = 2;

function hasCycle(graph) {
  const color = new Map(Object.keys(graph).map((k) => [Number(k), WHITE]));

  function visit(v) {
    color.set(v, GRAY);
    for (const w of graph[v] || []) {
      const c = color.has(w) ? color.get(w) : WHITE;
      if (c === GRAY) return true;
      if (c === WHITE && visit(w)) return true;
    }
    color.set(v, BLACK);
    return false;
  }

  for (const v of Object.keys(graph).map(Number)) {
    if (color.get(v) === WHITE && visit(v)) return true;
  }
  return false;
}

module.exports = { hasCycle };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(hasCycle({ 0: [1, 2], 1: [3], 2: [3], 3: [] }), false);
  assert.strictEqual(hasCycle({}), false);
  assert.strictEqual(hasCycle({ 7: [] }), false);
  assert.strictEqual(hasCycle({ 0: [1], 1: [2], 2: [0] }), true);
  assert.strictEqual(hasCycle({ 0: [0] }), true);
  assert.strictEqual(hasCycle({ 0: [1], 1: [], 2: [3], 3: [2] }), true);
  assert.strictEqual(hasCycle({ 0: [1], 1: [], 2: [3], 3: [] }), false);
  assert.strictEqual(hasCycle({ 0: [1], 1: [2], 2: [1] }), true);
  assert.strictEqual(hasCycle({ 0: [1, 2], 1: [3], 2: [3], 3: [] }), false);
  console.log("PASS");
}
