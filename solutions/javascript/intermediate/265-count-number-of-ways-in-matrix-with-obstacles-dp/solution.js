/** Task 265 - Count number of ways in matrix with obstacles (DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/**
 * Rolling 1D DP: ways[c] = ways[c] (from above) + ways[c-1] (from the left), zeroed on
 * obstacles (1 marks an obstacle, 0 a free cell). O(m*n) time, O(n) space.
 * Empty grid / empty first row -> 0; blocked start or end -> 0; 1 x n and n x 1 grids
 * have the single monotone path, so 1 iff no cell is blocked.
 * On identical grids this equals task 266's unique-path DP; on obstacle-free grids both
 * equal C(m+n-2, m-1) - asserted below for several sizes.
 */
function countWays(grid) {
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

/** Recursive enumeration used only for cross-checks. */
function countWaysBrute(grid) {
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

/** Task 266's closed form C(m+n-2, m-1), used for obstacle-free cross-checks. */
function uniquePathsNoObstacles(m, n) {
  if (m <= 0 || n <= 0) return 0;
  let result = 1;
  for (let i = 1; i <= m - 1; i++) result = result * (n - 1 + i) / i;   // C(m+n-2, m-1)
  return Math.round(result);
}

const freeGrid = (m, n) => Array.from({ length: m }, () => new Array(n).fill(0));

module.exports = { countWays, countWaysBrute, uniquePathsNoObstacles };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countWays([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), 6);           // classic 3x3
  assert.strictEqual(countWays([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), 2);           // classic obstacle case
  assert.strictEqual(countWays([[0, 1], [0, 0]]), 1);                            // only down then right
  assert.strictEqual(countWays([[1, 0], [0, 0]]), 0);                            // start blocked
  assert.strictEqual(countWays([[0, 0], [0, 1]]), 0);                            // end blocked
  assert.strictEqual(countWays([[0, 0, 0, 0, 0]]), 1);                           // 1 x n
  assert.strictEqual(countWays([[0], [0], [0]]), 1);                             // n x 1
  assert.strictEqual(countWays([[0, 1, 0]]), 0);                                 // 1 x n blocked
  assert.strictEqual(countWays([[0], [1], [0]]), 0);                             // n x 1 blocked
  assert.strictEqual(countWays([[0, 1, 0], [0, 0, 0]]), 1);
  assert.strictEqual(countWays([[0, 1], [1, 0]]), 0);                            // both moves blocked
  assert.strictEqual(countWays([]), 0);                                          // empty grid
  assert.strictEqual(countWays([[]]), 0);

  const sizes = [[1, 1, 1], [1, 5, 1], [2, 2, 2], [3, 3, 6], [3, 7, 28], [4, 4, 20]];
  for (const [m, n, expected] of sizes) {
    assert.strictEqual(countWays(freeGrid(m, n)), expected);
    assert.strictEqual(uniquePathsNoObstacles(m, n), expected, `closed form ${m}x${n}`);
  }

  // deterministic LCG so the cross-check is reproducible
  let seed = 265;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  for (let test = 0; test < 300; test++) {        // cross-check vs recursive enumeration
    const m = randInt(1, 5), n = randInt(1, 5);
    const grid = Array.from({ length: m }, () => Array.from({ length: n }, () => (rand() < 0.3 ? 1 : 0)));
    assert.strictEqual(countWays(grid), countWaysBrute(grid), `random agreement #${test}`);
  }
  console.log("PASS");
}
