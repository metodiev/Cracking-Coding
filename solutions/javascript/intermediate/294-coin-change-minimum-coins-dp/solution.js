/** Task 294 - Coin change: minimum coins (DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function minCoins(coins, amount) {
  const infinity = Infinity;
  const dp = new Array(amount + 1).fill(infinity);
  dp[0] = 0;
  for (let s = 1; s <= amount; s++) {
    for (const coin of coins) {
      if (coin <= s && dp[s - coin] + 1 < dp[s]) dp[s] = dp[s - coin] + 1;
    }
  }
  return dp[amount] === infinity ? -1 : dp[amount];
}

function bfsMinCoins(coins, amount) {
  const dist = new Array(amount + 1).fill(-1);
  dist[0] = 0;
  const queue = [0];
  for (let head = 0; head < queue.length; head++) {
    const u = queue[head];
    for (const coin of coins) {
      const v = u + coin;
      if (v <= amount && dist[v] === -1) {
        dist[v] = dist[u] + 1;
        queue.push(v);
      }
    }
  }
  return dist[amount];
}

function lcgCoins(n, seed) {
  const values = [];
  while (values.length < n) {
    seed = (seed * 75 + 74) % 65537;
    const value = 1 + (seed % 7);
    if (!values.includes(value)) values.push(value);
  }
  values.sort((a, b) => a - b);
  return [values, seed];
}

module.exports = { minCoins, bfsMinCoins, lcgCoins };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(minCoins([1, 2, 5], 11), 3);
  assert.strictEqual(minCoins([2], 3), -1);
  assert.strictEqual(minCoins([1, 2, 5], 0), 0);
  assert.strictEqual(minCoins([1, 3, 4], 6), 2);
  assert.strictEqual(minCoins([1, 3, 4], 9), 3);
  assert.strictEqual(minCoins([5, 7], 11), -1);
  assert.strictEqual(minCoins([], 0), 0);

  let seed = 13579;
  for (let trial = 0; trial < 150; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const count = 1 + (seed % 4);
    let coins;
    [coins, seed] = lcgCoins(count, seed);
    seed = (seed * 75 + 74) % 65537;
    const amount = seed % 80;
    assert.strictEqual(minCoins(coins, amount), bfsMinCoins(coins, amount), `trial ${trial}`);
  }
  console.log("PASS");
}
