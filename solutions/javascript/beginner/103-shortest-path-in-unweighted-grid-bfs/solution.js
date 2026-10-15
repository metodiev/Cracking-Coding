/** Task 103 - Shortest path in unweighted grid (BFS). (Tier: Beginner, CTCI Ch 11) */
"use strict";

function shortestPath(grid, start, goal) {
  if (!grid || grid.length === 0 || grid[0].length === 0) return -1;
  const rows = grid.length;
  const cols = grid[0].length;
  const [sr, sc] = start;
  const [gr, gc] = goal;
  if (grid[sr][sc] === 1 || grid[gr][gc] === 1) return -1;
  if (sr === gr && sc === gc) return 0;
  const dist = Array.from({ length: rows }, () => new Array(cols).fill(-1));
  dist[sr][sc] = 0;
  const queue = [[sr, sc]];
  for (let head = 0; head < queue.length; head++) {
    const [r, c] = queue[head];
    for (const [nr, nc] of [[r + 1, c], [r - 1, c], [r, c + 1], [r, c - 1]]) {
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] === 0 && dist[nr][nc] === -1) {
        dist[nr][nc] = dist[r][c] + 1;
        if (nr === gr && nc === gc) return dist[nr][nc];
        queue.push([nr, nc]);
      }
    }
  }
  return -1;
}

module.exports = { shortestPath };

if (require.main === module) {
  const assert = require("node:assert");
  const open3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  assert.strictEqual(shortestPath(open3, [0, 0], [2, 2]), 4);
  assert.strictEqual(shortestPath(open3, [0, 0], [0, 0]), 0);
  assert.strictEqual(shortestPath([[0]], [0, 0], [0, 0]), 0);
  const detour = [[0, 0, 0], [1, 1, 0], [0, 0, 0]];
  assert.strictEqual(shortestPath(detour, [0, 0], [2, 0]), 6);
  const split = [[0, 1], [1, 0]];
  assert.strictEqual(shortestPath(split, [0, 0], [1, 1]), -1);
  const blocked = [[0, 1], [0, 0]];
  assert.strictEqual(shortestPath(blocked, [0, 0], [0, 1]), -1);
  assert.strictEqual(shortestPath(blocked, [0, 1], [0, 0]), -1);
  assert.strictEqual(shortestPath([], [0, 0], [0, 0]), -1);
  const direct = Array.from({ length: 4 }, () => new Array(4).fill(0));
  assert.strictEqual(shortestPath(direct, [0, 0], [3, 3]), 6);
  console.log("PASS");
}
