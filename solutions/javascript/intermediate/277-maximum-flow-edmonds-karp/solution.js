/** Task 277 - Maximum flow (Edmonds-Karp). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function edmondsKarp(n, capacity, source, sink) {
  const residual = capacity.map((row) => row.slice());
  let total = 0;
  for (;;) {
    const parent = new Array(n).fill(-1);
    parent[source] = source;
    const queue = [source];
    for (let head = 0; head < queue.length && parent[sink] === -1; head++) {
      const u = queue[head];
      for (let v = 0; v < n; v++) {
        if (parent[v] === -1 && residual[u][v] > 0) {
          parent[v] = u;
          queue.push(v);
        }
      }
    }
    if (parent[sink] === -1) return total;

    let bottleneck = Infinity;
    for (let v = sink; v !== source; v = parent[v]) bottleneck = Math.min(bottleneck, residual[parent[v]][v]);
    for (let v = sink; v !== source; v = parent[v]) {
      const u = parent[v];
      residual[u][v] -= bottleneck;
      residual[v][u] += bottleneck;
    }
    total += bottleneck;
  }
}

function matrixFromEdges(n, edges) {
  const capacity = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const [u, v, cap] of edges) capacity[u][v] += cap;
  return capacity;
}

module.exports = { edmondsKarp, matrixFromEdges };

if (require.main === module) {
  const assert = require("node:assert");
  const clrs = matrixFromEdges(6, [
    [0, 1, 16], [0, 2, 13], [1, 2, 10], [2, 1, 4], [1, 3, 12],
    [3, 2, 9], [2, 4, 14], [4, 3, 7], [3, 5, 20], [4, 5, 4],
  ]);
  assert.strictEqual(edmondsKarp(6, clrs, 0, 5), 23);

  assert.strictEqual(edmondsKarp(4, matrixFromEdges(4, [[0, 1, 3], [0, 2, 2], [1, 3, 2], [2, 3, 3]]), 0, 3), 4);
  assert.strictEqual(edmondsKarp(3, matrixFromEdges(3, [[0, 1, 3], [1, 2, 5]]), 0, 2), 3);
  assert.strictEqual(edmondsKarp(3, matrixFromEdges(3, [[0, 1, 1]]), 0, 2), 0);
  assert.strictEqual(edmondsKarp(2, matrixFromEdges(2, []), 0, 1), 0);

  const crossing = matrixFromEdges(4, [[0, 1, 1], [0, 2, 1], [1, 2, 1], [1, 3, 1], [2, 3, 1]]);
  assert.strictEqual(edmondsKarp(4, crossing, 0, 3), 2);

  assert.strictEqual(edmondsKarp(3, matrixFromEdges(3, [[0, 1, 2], [0, 1, 3], [1, 2, 4]]), 0, 2), 4);

  const totalOut = clrs[0].reduce((a, b) => a + b, 0);
  assert.ok(edmondsKarp(6, clrs, 0, 5) <= totalOut);
  console.log("PASS");
}
