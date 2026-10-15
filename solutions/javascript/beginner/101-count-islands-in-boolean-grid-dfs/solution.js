/** Task 101 - Count islands in boolean grid (DFS). (Tier: Beginner, CTCI Ch 4) */
"use strict";

function countIslands(grid) {
  if (!grid || grid.length === 0 || grid[0].length === 0) return 0;
  const rows = grid.length;
  const cols = grid[0].length;
  const seen = grid.map((row) => row.slice());
  let islands = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (seen[r][c] === 1) {
        islands += 1;
        flood(seen, r, c);
      }
    }
  }
  return islands;
}

function flood(grid, sr, sc) {
  const rows = grid.length;
  const cols = grid[0].length;
  const stack = [[sr, sc]];
  grid[sr][sc] = 0;
  while (stack.length > 0) {
    const [r, c] = stack.pop();
    for (const [nr, nc] of [[r + 1, c], [r - 1, c], [r, c + 1], [r, c - 1]]) {
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] === 1) {
        grid[nr][nc] = 0;
        stack.push([nr, nc]);
      }
    }
  }
}

module.exports = { countIslands };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countIslands([]), 0);
  assert.strictEqual(countIslands([[]]), 0);
  assert.strictEqual(countIslands([[0, 0], [0, 0]]), 0);
  assert.strictEqual(countIslands([[1, 1], [1, 1]]), 1);
  assert.strictEqual(countIslands([[1]]), 1);
  assert.strictEqual(countIslands([[0]]), 0);
  assert.strictEqual(countIslands([[1, 0], [0, 1]]), 2);
  const mixed = [
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1],
  ];
  assert.strictEqual(countIslands(mixed), 5);
  assert.strictEqual(mixed[0][0], 1);
  console.log("PASS");
}
