/** Task 267 - Minimum path sum in grid (DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/**
 * Rolling 1D DP: dp[c] = min(dp[c] from above, dp[c-1] from the left) + grid[r][c];
 * the first row and column each have a single predecessor. O(m*n) time, O(n) space.
 * Empty grid / empty first row raises; a single cell returns its value; 1 x n and n x 1
 * grids have a single path. Negatives are allowed.
 */
function minPathSum(grid) {
  if (grid.length === 0 || grid[0].length === 0) throw new Error("grid must be a non-empty rectangle");
  const n = grid[0].length;
  const dp = new Array(n).fill(0);
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < n; c++) {
      if (r === 0 && c === 0) dp[c] = grid[r][c];
      else if (r === 0) dp[c] = dp[c - 1] + grid[r][c];            // only from the left
      else if (c === 0) dp[c] += grid[r][c];                       // only from above
      else dp[c] = Math.min(dp[c], dp[c - 1]) + grid[r][c];
    }
  }
  return dp[n - 1];
}

/** Recursive enumeration of all paths; used only for cross-checks. */
function minPathSumBrute(grid) {
  const m = grid.length, n = grid[0].length;
  const go = (r, c) => {
    if (r === m - 1 && c === n - 1) return grid[r][c];
    let best = null;
    if (r + 1 < m) best = go(r + 1, c);
    if (c + 1 < n) {
      const right = go(r, c + 1);
      best = best === null ? right : Math.min(best, right);
    }
    return grid[r][c] + best;
  };
  return go(0, 0);
}

module.exports = { minPathSum, minPathSumBrute };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(minPathSum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]), 7);   // classic = 7
  assert.strictEqual(minPathSum([[5]]), 5);                               // single cell
  assert.strictEqual(minPathSum([[-3]]), -3);
  assert.strictEqual(minPathSum([[1, 2, 3]]), 6);                         // 1 x n single path
  assert.strictEqual(minPathSum([[1], [2], [3]]), 6);                     // n x 1 single path
  assert.strictEqual(minPathSum([[1, 2, 3], [4, 100, 6]]), 12);           // route above the huge cell
  assert.strictEqual(minPathSum([[1, 100], [1, 1]]), 3);                  // down first, then right
  assert.strictEqual(minPathSum([[1, -3], [1, -5]]), -7);                 // negatives allowed
  assert.throws(() => minPathSum([]), /non-empty/);
  assert.throws(() => minPathSum([[]]), /non-empty/);

  // deterministic LCG so the cross-check is reproducible
  let seed = 267;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  for (let test = 0; test < 300; test++) {        // cross-check vs recursive enumeration
    const m = randInt(1, 5), n = randInt(1, 5);
    const grid = Array.from({ length: m }, () => Array.from({ length: n }, () => randInt(-5, 9)));
    assert.strictEqual(minPathSum(grid), minPathSumBrute(grid), `random agreement #${test}`);
  }
  console.log("PASS");
}
