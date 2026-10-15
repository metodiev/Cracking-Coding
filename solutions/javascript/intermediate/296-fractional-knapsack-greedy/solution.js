/** Task 296 - Fractional knapsack (greedy). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function fractionalKnapsack(weights, values, capacity) {
  const order = Array.from({ length: weights.length }, (_, i) => i);
  order.sort((a, b) => {
    const left = values[a] / weights[a];
    const right = values[b] / weights[b];
    if (left !== right) return right - left;
    return a - b;
  });
  let remaining = capacity;
  let total = 0;
  const picks = [];
  for (const index of order) {
    if (remaining <= 0) break;
    const fraction = Math.min(1, remaining / weights[index]);
    total += fraction * values[index];
    remaining -= fraction * weights[index];
    picks.push([index, fraction]);
  }
  return { total, picks };
}

function bruteFractional(weights, values, capacity) {
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
    if (weight > capacity) continue;
    best = Math.max(best, value);
    for (let j = 0; j < n; j++) {
      if (!((mask >> j) & 1)) {
        // The fraction is capped at 1: the remaining item exists only once (an uncapped
        // term would model an unbounded fluid supply instead).
        const fraction = Math.min(1, (capacity - weight) / weights[j]);
        best = Math.max(best, value + fraction * values[j]);
      }
    }
  }
  return best;
}

function lcgInstance(n, seed) {
  const weights = [];
  const values = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    weights.push(1 + (seed % 9));
    seed = (seed * 75 + 74) % 65537;
    values.push(1 + (seed % 9));
  }
  return [weights, values, seed];
}

module.exports = { fractionalKnapsack, bruteFractional, lcgInstance };

if (require.main === module) {
  const assert = require("node:assert");

  const weights = [10, 20, 30];
  const values = [60, 100, 120];
  let { total, picks } = fractionalKnapsack(weights, values, 50);
  assert.ok(Math.abs(total - 240) < 1e-9, `classic example ${total}`);
  assert.deepStrictEqual(picks.map(([index]) => index), [0, 1, 2]);
  assert.ok(Math.abs(picks[0][1] - 1) < 1e-9 && Math.abs(picks[1][1] - 1) < 1e-9);
  assert.ok(Math.abs(picks[2][1] - 2 / 3) < 1e-9);

  ({ total, picks } = fractionalKnapsack(weights, values, 60));
  assert.ok(Math.abs(total - 280) < 1e-9);
  assert.ok(picks.every(([, fraction]) => Math.abs(fraction - 1) < 1e-9));

  ({ total, picks } = fractionalKnapsack(weights, values, 0));
  assert.strictEqual(total, 0);
  assert.deepStrictEqual(picks, []);

  ({ total, picks } = fractionalKnapsack([5], [3], 2));
  assert.ok(Math.abs(total - 1.2) < 1e-9 && Math.abs(picks[0][1] - 0.4) < 1e-9);

  let seed = 424242;
  for (let trial = 0; trial < 120; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 1 + (seed % 8);
    let w;
    let v;
    [w, v, seed] = lcgInstance(n, seed);
    seed = (seed * 75 + 74) % 65537;
    const capacity = seed % 50;
    const answer = fractionalKnapsack(w, v, capacity);
    const expected = bruteFractional(w, v, capacity);
    assert.ok(Math.abs(answer.total - expected) < 1e-9, `trial ${trial}`);
    const weight = answer.picks.reduce((acc, [index, fraction]) => acc + fraction * w[index], 0);
    const value = answer.picks.reduce((acc, [index, fraction]) => acc + fraction * v[index], 0);
    assert.ok(weight <= capacity + 1e-9 && Math.abs(value - answer.total) < 1e-9, `picks ${trial}`);
    assert.ok(answer.picks.every(([, fraction]) => fraction > 0 && fraction <= 1 + 1e-12), `fractions ${trial}`);
  }
  console.log("PASS");
}
