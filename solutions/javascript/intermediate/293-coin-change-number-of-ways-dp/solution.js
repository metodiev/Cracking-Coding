/** Task 293 - Coin change: number of ways (DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function countWays(coins, amount) {
  const ways = new Array(amount + 1).fill(0);
  ways[0] = 1;
  for (const coin of coins) {
    for (let s = coin; s <= amount; s++) ways[s] += ways[s - coin];
  }
  return ways[amount];
}

function bruteWays(coins, amount) {
  const count = (index, remaining) => {
    if (remaining === 0) return 1;
    if (index === coins.length) return 0;
    let total = 0;
    for (let take = 0; take * coins[index] <= remaining; take++) {
      total += count(index + 1, remaining - take * coins[index]);
    }
    return total;
  };
  return count(0, amount);
}

function lcgCoins(n, seed) {
  const values = [];
  while (values.length < n) {
    seed = (seed * 75 + 74) % 65537;
    const value = 1 + (seed % 8);
    if (!values.includes(value)) values.push(value);
  }
  values.sort((a, b) => a - b);
  return [values, seed];
}

module.exports = { countWays, bruteWays, lcgCoins };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(countWays([1, 2, 5], 5), 4);
  assert.strictEqual(countWays([2], 3), 0);
  assert.strictEqual(countWays([1, 2, 5], 0), 1);
  assert.strictEqual(countWays([1], 10), 1);
  assert.strictEqual(countWays([], 0), 1);
  assert.strictEqual(countWays([2, 3, 5], 9), 3);
  assert.strictEqual(countWays([1, 2], 3), 2);

  let seed = 24680;
  for (let trial = 0; trial < 150; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const count = 1 + (seed % 5);
    let coins;
    [coins, seed] = lcgCoins(count, seed);
    seed = (seed * 75 + 74) % 65537;
    const amount = seed % 60;
    assert.strictEqual(countWays(coins, amount), bruteWays(coins, amount), `trial ${trial}`);
  }
  console.log("PASS");
}
