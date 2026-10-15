/** Task 114 - Bellman-Ford algorithm (with negative edges). (Tier: Beginner, CTCI Ch 4) */
"use strict";

function bellmanFord(n, edges, src) {
  const dist = new Array(n).fill(Infinity);
  dist[src] = 0;
  for (let pass = 0; pass < n - 1; pass++) {
    let changed = false;
    for (const [u, v, w] of edges) {
      if (dist[u] !== Infinity && dist[u] + w < dist[v]) {
        dist[v] = dist[u] + w;
        changed = true;
      }
    }
    if (!changed) break;
  }
  for (const [u, v, w] of edges) {
    if (dist[u] !== Infinity && dist[u] + w < dist[v]) return null;
  }
  return dist;
}

module.exports = { bellmanFord };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(bellmanFord(4, [[0, 1, 4], [0, 2, 5], [1, 2, -3], [2, 3, 2]], 0), [0, 4, 1, 3]);
  assert.deepStrictEqual(bellmanFord(3, [[0, 1, 1], [1, 2, 2], [0, 2, 4]], 0), [0, 1, 3]);
  assert.strictEqual(bellmanFord(3, [[0, 1, 1], [1, 2, -2], [2, 1, -2]], 0), null);
  console.log("PASS");
}
