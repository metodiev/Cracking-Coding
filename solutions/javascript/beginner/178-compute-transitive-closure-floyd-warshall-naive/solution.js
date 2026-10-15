/** Task 178 - Compute transitive closure (Floyd-Warshall naive). (Tier: Beginner, CTCI Ch 4) */
"use strict";

function transitiveClosure(adj) {
  const n = adj.length;
  const reach = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => adj[i][j] !== 0 || i === j));
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      if (reach[i][k]) {
        for (let j = 0; j < n; j++) {
          if (reach[k][j]) reach[i][j] = true;
        }
      }
    }
  }
  return reach;
}

module.exports = { transitiveClosure };

if (require.main === module) {
  const assert = require("node:assert");
  const chain = [[0, 1, 0], [0, 0, 1], [0, 0, 0]];
  const reach = transitiveClosure(chain);
  assert.deepStrictEqual(reach[0], [true, true, true]);
  assert.deepStrictEqual(reach[2], [false, false, true]);

  const cycle = [[0, 1, 0], [1, 0, 0], [0, 0, 0]];
  const reach2 = transitiveClosure(cycle);
  assert.ok(reach2[0][1] && reach2[1][0] && !reach2[0][2]);

  assert.deepStrictEqual(transitiveClosure([]), []);
  assert.deepStrictEqual(transitiveClosure([[0]]), [[true]]);
  console.log("PASS");
}
