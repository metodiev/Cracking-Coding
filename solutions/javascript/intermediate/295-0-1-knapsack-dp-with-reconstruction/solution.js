/** Task 295 - 0/1 knapsack (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function knapsack(weights, values, capacity) {
  const n = weights.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(capacity + 1).fill(0));
  for (let i = 1; i <= n; i++) {
    const weight = weights[i - 1];
    const value = values[i - 1];
    for (let c = 0; c <= capacity; c++) {
      let best = dp[i - 1][c];
      if (c >= weight && dp[i - 1][c - weight] + value > best) best = dp[i - 1][c - weight] + value;
      dp[i][c] = best;
    }
  }
  const chosen = [];
  let c = capacity;
  for (let i = n; i >= 1; i--) {
    if (dp[i][c] !== dp[i - 1][c]) {
      chosen.push(i - 1);
      c -= weights[i - 1];
    }
  }
  chosen.reverse();
  return [dp[n][capacity], chosen];
}

function bruteKnapsack(weights, values, capacity) {
  const n = weights.length;
  let best = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    let weight = 0;
    let value = 0;
    for (let i = 0; i < n; i++) {
      if ((mask >> i) & 1) {
        weight += weights[i];
        value += values[i];
      }
    }
    if (weight <= capacity && value > best) best = value;
  }
  return best;
}

function lcgInstance(n, seed) {
  const weights = [];
  const values = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    weights.push(1 + (seed % 12));
    seed = (seed * 75 + 74) % 65537;
    values.push(1 + (seed % 12));
  }
  return [weights, values, seed];
}

module.exports = { knapsack, bruteKnapsack, lcgInstance };

if (require.main === module) {
  const assert = require("node:assert");

  const weights = [1, 3, 4, 5];
  const values = [1, 4, 5, 7];
  let [value, chosen] = knapsack(weights, values, 7);
  assert.strictEqual(value, 9);
  assert.deepStrictEqual(chosen.map((i) => weights[i]).sort((a, b) => a - b), [3, 4]);
  assert.strictEqual(chosen.reduce((acc, i) => acc + values[i], 0), 9);

  assert.deepStrictEqual(knapsack(weights, values, 0), [0, []]);
  assert.deepStrictEqual(knapsack([2, 9], [3, 100], 5), [3, [0]]);
  assert.deepStrictEqual(knapsack([4], [9], 4), [9, [0]]);

  [value, chosen] = knapsack(weights, values, 100);
  assert.strictEqual(value, 17);
  assert.deepStrictEqual(chosen, [0, 1, 2, 3]);

  let seed = 555;
  for (let trial = 0; trial < 120; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 1 + (seed % 14);
    let w;
    let v;
    [w, v, seed] = lcgInstance(n, seed);
    seed = (seed * 75 + 74) % 65537;
    const capacity = seed % 40;
    const [answer, picked] = knapsack(w, v, capacity);
    assert.strictEqual(answer, bruteKnapsack(w, v, capacity), `trial ${trial}`);
    const weight = picked.reduce((acc, i) => acc + w[i], 0);
    const total = picked.reduce((acc, i) => acc + v[i], 0);
    assert.ok(weight <= capacity && total === answer, `reconstruction ${trial}`);
  }
  console.log("PASS");
}
