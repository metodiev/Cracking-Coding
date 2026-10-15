/** Task 220 - Implement bitmask DP for TSP small N. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function tsp(dist) {
  const n = dist.length;
  if (n <= 1) return 0;
  const INF = Infinity;
  const full = (1 << n) - 1;
  const dp = Array.from({ length: 1 << n }, () => new Array(n).fill(INF));
  dp[1][0] = 0;
  for (let mask = 0; mask <= full; mask++) {
    for (let last = 0; last < n; last++) {
      if (dp[mask][last] === INF) continue;
      for (let nxt = 0; nxt < n; nxt++) {
        if (mask & (1 << nxt)) continue;
        const newMask = mask | (1 << nxt);
        const candidate = dp[mask][last] + dist[last][nxt];
        if (candidate < dp[newMask][nxt]) dp[newMask][nxt] = candidate;
      }
    }
  }
  let best = INF;
  for (let last = 0; last < n; last++) {
    if (dp[full][last] !== INF) best = Math.min(best, dp[full][last] + dist[last][0]);
  }
  return best;
}

module.exports = { tsp };

if (require.main === module) {
  const assert = require("node:assert");
  const square = [
    [0, 1, 10, 10],
    [1, 0, 1, 10],
    [10, 1, 0, 1],
    [1, 10, 1, 0],
  ];
  assert.strictEqual(tsp(square), 4);
  assert.strictEqual(tsp([[0, 1, 3], [1, 0, 2], [3, 2, 0]]), 6);
  assert.strictEqual(tsp([[0]]), 0);
  assert.strictEqual(tsp([[0, 7], [7, 0]]), 14);

  // brute-force cross-check on a random 7-city instance
  let seed = 6;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const n = 7;
  const d = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) d[i][j] = d[j][i] = 1 + Math.floor(rng() * 19);
  }
  let brute = Infinity;
  const perm = [1, 2, 3, 4, 5, 6];
  const permute = (arr, k = 0) => {
    if (k === arr.length) {
      let cost = d[0][arr[0]];
      for (let i = 0; i < arr.length - 1; i++) cost += d[arr[i]][arr[i + 1]];
      cost += d[arr[arr.length - 1]][0];
      brute = Math.min(brute, cost);
      return;
    }
    for (let i = k; i < arr.length; i++) {
      [arr[k], arr[i]] = [arr[i], arr[k]];
      permute(arr, k + 1);
      [arr[k], arr[i]] = [arr[i], arr[k]];
    }
  };
  permute(perm);
  assert.strictEqual(tsp(d), brute);
  console.log("PASS");
}
