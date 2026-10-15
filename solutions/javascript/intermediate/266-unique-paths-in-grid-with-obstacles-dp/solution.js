/** Task 266 - Unique paths in grid with obstacles (DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/**
 * Explicit m x n DP for the number of monotone right/down paths (1 = obstacle).
 * With no obstacles this reduces to the closed form C(m+n-2, m-1)
 * (uniquePathsNoObstacles); obstacles zero out blocked cells, so they can only lower
 * the count. A blocked start or end yields 0. O(m*n) time and space.
 */
function uniquePathsWithObstacles2d(grid) {
  if (grid.length === 0 || grid[0].length === 0) return 0;
  const m = grid.length, n = grid[0].length;
  const dp = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let r = 0; r < m; r++) {
    for (let c = 0; c < n; c++) {
      if (grid[r][c] === 1) dp[r][c] = 0;
      else if (r === 0 && c === 0) dp[r][c] = 1;
      else dp[r][c] = (r > 0 ? dp[r - 1][c] : 0) + (c > 0 ? dp[r][c - 1] : 0);
    }
  }
  return dp[m - 1][n - 1];
}

/** Task 265's rolling 1D DP; identical grids must give identical counts. */
function countWays265(grid) {
  if (grid.length === 0 || grid[0].length === 0) return 0;
  const n = grid[0].length;
  const ways = new Array(n).fill(0);
  ways[0] = 1;
  for (const row of grid) {
    for (let c = 0; c < n; c++) {
      if (row[c] === 1) ways[c] = 0;
      else if (c > 0) ways[c] += ways[c - 1];
    }
  }
  return ways[n - 1];
}

/** Closed form for an obstacle-free m x n grid: C(m+n-2, m-1). */
function uniquePathsNoObstacles(m, n) {
  if (m <= 0 || n <= 0) return 0;
  let result = 1;
  for (let i = 1; i <= m - 1; i++) result = result * (n - 1 + i) / i;
  return Math.round(result);
}

/** Recursive enumeration used only for cross-checks. */
function uniquePathsBrute(grid) {
  const m = grid.length, n = grid[0].length;
  const go = (r, c) => {
    if (grid[r][c] === 1) return 0;
    if (r === m - 1 && c === n - 1) return 1;
    let total = 0;
    if (r + 1 < m) total += go(r + 1, c);
    if (c + 1 < n) total += go(r, c + 1);
    return total;
  };
  return go(0, 0);
}

const freeGrid = (m, n) => Array.from({ length: m }, () => new Array(n).fill(0));

module.exports = { uniquePathsWithObstacles2d, countWays265, uniquePathsNoObstacles, uniquePathsBrute };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(uniquePathsWithObstacles2d([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), 6);   // classic 3x3 = 6
  assert.strictEqual(uniquePathsWithObstacles2d([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), 2);   // obstacle variant
  assert.strictEqual(uniquePathsWithObstacles2d([[0, 1], [0, 0]]), 1);                    // detour down
  assert.strictEqual(uniquePathsWithObstacles2d([[0, 0], [0, 1]]), 0);                    // goal blocked
  assert.strictEqual(uniquePathsWithObstacles2d([[1, 0], [0, 0]]), 0);                    // start blocked
  assert.strictEqual(uniquePathsWithObstacles2d([[0, 0], [1, 0]]), 1);                    // detour right
  assert.strictEqual(uniquePathsWithObstacles2d([[0, 0, 0, 0, 0]]), 1);                   // 1 x n
  assert.strictEqual(uniquePathsWithObstacles2d([[0], [0], [0]]), 1);                     // n x 1
  assert.strictEqual(uniquePathsWithObstacles2d([[0, 1, 0]]), 0);                         // 1 x n blocked
  assert.strictEqual(uniquePathsWithObstacles2d([]), 0);
  assert.strictEqual(uniquePathsWithObstacles2d([[]]), 0);

  const shared = [
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1], [0, 0]],
    [[0, 0], [0, 1]],
    [[1, 0], [0, 0]],
    [[0, 1], [1, 0]],
    [[0, 1, 0], [0, 0, 0]],
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
  ];
  for (const grid of shared)                    // 265's counter agrees on identical grids
    assert.strictEqual(uniquePathsWithObstacles2d(grid), countWays265(grid),
      `agrees with the 265 counter: ${JSON.stringify(grid)}`);

  const sizes = [[1, 1, 1], [1, 5, 1], [2, 2, 2], [3, 3, 6], [3, 7, 28], [4, 4, 20]];
  for (const [m, n, expected] of sizes) {
    assert.strictEqual(uniquePathsWithObstacles2d(freeGrid(m, n)), expected);
    assert.strictEqual(uniquePathsNoObstacles(m, n), expected, `closed form ${m}x${n}`);
  }

  // deterministic LCG so the cross-check is reproducible
  let seed = 266;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  for (let test = 0; test < 300; test++) {      // 2D == 265's rolling DP == recursive enumeration
    const m = randInt(1, 5), n = randInt(1, 5);
    const grid = Array.from({ length: m }, () => Array.from({ length: n }, () => (rand() < 0.3 ? 1 : 0)));
    assert.strictEqual(uniquePathsWithObstacles2d(grid), countWays265(grid), `2D == rolling #${test}`);
    assert.strictEqual(uniquePathsWithObstacles2d(grid), uniquePathsBrute(grid), `2D == brute #${test}`);
  }
  console.log("PASS");
}
