/** Task 274 - Check if a graph is bipartite (BFS 2-coloring). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function twoColor(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const color = new Array(n).fill(-1);
  for (let start = 0; start < n; start++) {
    if (color[start] !== -1) continue;
    color[start] = 0;
    const queue = [start];
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      for (const v of adjacency[u]) {
        if (color[v] === -1) {
          color[v] = color[u] ^ 1;
          queue.push(v);
        } else if (color[v] === color[u]) {
          return null;
        }
      }
    }
  }
  return color;
}

const isBipartite = (n, edges) => twoColor(n, edges) !== null;

module.exports = { twoColor, isBipartite };

if (require.main === module) {
  const assert = require("node:assert");
  assert.ok(isBipartite(4, [[0, 1], [1, 2], [2, 3], [3, 0]]));
  assert.ok(!isBipartite(3, [[0, 1], [1, 2], [2, 0]]));
  assert.ok(isBipartite(3, [[0, 1], [1, 2]]));
  assert.ok(!isBipartite(1, [[0, 0]]));
  assert.ok(isBipartite(5, []));
  assert.ok(!isBipartite(7, [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 4]]));
  assert.ok(isBipartite(6, [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]));

  const k33 = [];
  for (let i = 0; i < 3; i++) for (let j = 3; j < 6; j++) k33.push([i, j]);
  assert.ok(isBipartite(6, k33));

  const k4 = [];
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) k4.push([i, j]);
  assert.ok(!isBipartite(4, k4));

  const colors = twoColor(4, [[0, 1], [1, 2], [2, 3], [3, 0]]);
  assert.ok(colors && colors[0] !== colors[1] && colors[1] !== colors[2]);
  assert.strictEqual(twoColor(3, [[0, 1], [1, 2], [2, 0]]), null);
  console.log("PASS");
}
