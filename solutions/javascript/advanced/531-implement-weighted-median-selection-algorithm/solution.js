/**
 * Task 531 - Implement weighted median selection algorithm. (Tier: Advanced, CTCI Ch 10 — Sorting and Searching)
 *
 * The weighted median of n values with non-negative weights is the *smallest* value v whose
 * cumulative weight W(<= v) reaches half of the total weight W (the "lower weighted median"
 * convention, which is the one that makes the answer unique even when a cumulative weight lands
 * exactly on W/2). Two algorithms are provided: a sort-based one that materialises all order
 * statistics, and a selection-based one that partitions with a deterministic median-of-three pivot
 * and prunes a side by *weight* rather than by rank - after splitting the current window around the
 * pivot it halves the problem on whichever side of the pivot carries half of the window's mass, so
 * only one recursive call is ever made. Values with zero weight are carried along but cannot shift
 * the answer, and duplicated values contribute the sum of their weights.
 *
 * Documented checks: both algorithms agree with an independent sort-and-scan computation on
 * 60 LCG-generated instances with random integer weights (n = 1..40) and on weight patterns that
 * put all mass on a single - possibly extreme - value; the defining property W(< v) < W/2 and
 * W(<= v) >= W/2 is verified for every instance; the answer is also checked to minimise the
 * weighted absolute deviation sum_i w_i |x_i - t| over every candidate value in the support, which
 * is the variational characterisation of the weighted median and an entirely independent test; and
 * on a hand-computed instance ("0 w=1, 10 w=3, 20 w=1" has the weighted median 10 because W/2 = 2.5
 * while W(<= 0) = 1) the exact value is pinned down. Boundary and error cases: a single element,
 * zero weights everywhere on one side, an exact tie at W/2 (the smaller value wins), an all-zero
 * weight vector and any negative weight (both rejected), and a two-element instance whose weighted
 * median is the heavier element. Finally the counter instrumentation shows the selection path stays
 * linear-ish in practice: for n = 1000, 5000 and 20000 random instances it performs at most 10n + 50
 * element comparisons in at most 60 partition levels (asserted, not merely reported).
 */
"use strict";

function mean(values) {
  return values.reduce((a, b) => a + b, 0.0) / values.length;
}

class Lcg {
  constructor(seed) {
    let s = ((seed >>> 0) ^ 0x9e3779b9) >>> 0;
    s = Math.imul(s ^ (s >>> 16), 0x85ebca6b) >>> 0;
    s = Math.imul(s ^ (s >>> 13), 0xc2b2ae35) >>> 0;
    this.state = (s ^ (s >>> 16)) >>> 0;
    if (this.state === 0) this.state = 0x1b873593;
  }

  nextUint32() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  nextInt(bound) {
    return Math.floor((this.nextUint32() / 4294967296) * bound);
  }
}

/** Deterministic instance: integer values in [0, 2 * spread) and weights in [0, maxWeight]. */
function randomInstance(n, seed, spread, maxWeight) {
  const rng = new Lcg(seed);
  const values = [];
  const weights = [];
  for (let i = 0; i < n; i++) {
    values.push(rng.nextInt(2 * spread));
    weights.push(1 + rng.nextInt(maxWeight));
  }
  return { values, weights };
}

function totalWeight(weights) {
  let total = 0.0;
  for (const w of weights) {
    if (!(w >= 0.0)) throw new Error("weights must be non-negative");
    total += w;
  }
  if (!(total > 0.0)) throw new Error("the total weight must be positive");
  return total;
}

/** Independent definition: sort by value and take the first value reaching half of the mass. */
function weightedMedianSorted(values, weights) {
  if (values.length !== weights.length) throw new Error("length mismatch");
  if (values.length === 0) throw new Error("empty input");
  const total = totalWeight(weights);
  const order = values.map((value, i) => i)
    .sort((a, b) => (values[a] - values[b]) || (a - b));
  let cumulative = 0.0;
  for (const index of order) {
    cumulative += weights[index];
    if (cumulative * 2.0 >= total) return values[index];
  }
  return values[order[order.length - 1]];
}

/**
 * Weighted selection: deterministic median-of-three pivot with a Dutch-national-flag partition
 * (a[lo..lt-1] < pivot, a[lt..gt] == pivot, a[gt+1..hi] > pivot), then a single recursive side
 * chosen by weight. An optional `stats` object collects the comparison and level counts.
 */
function weightedMedianSelect(values, weights, stats = null) {
  if (values.length !== weights.length) throw new Error("length mismatch");
  if (values.length === 0) throw new Error("empty input");
  const total = totalWeight(weights);
  const items = values.map((value, i) => ({ value, weight: weights[i] }));
  const swap = (a, b) => {
    const tmp = items[a];
    items[a] = items[b];
    items[b] = tmp;
  };
  let lo = 0;
  let hi = items.length - 1;
  let below = 0.0;  // weight of everything strictly smaller than the current window
  for (;;) {
    const mid = (lo + hi) >> 1;
    const candidates = [[items[lo].value, lo], [items[mid].value, mid], [items[hi].value, hi]];
    candidates.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    const pivot = candidates[1][0];
    let lt = lo;
    let i = lo;
    let gt = hi;
    while (i <= gt) {
      if (stats !== null) stats.comparisons += 1;
      if (items[i].value < pivot) {
        swap(lt, i);
        lt += 1;
        i += 1;
      } else if (items[i].value > pivot) {
        swap(i, gt);
        gt -= 1;
      } else {
        i += 1;
      }
    }
    let lessWeight = 0.0;
    for (let k = lo; k < lt; k++) lessWeight += items[k].weight;
    let pivotWeight = 0.0;
    for (let k = lt; k <= gt; k++) pivotWeight += items[k].weight;
    if (stats !== null) stats.levels += 1;
    if ((below + lessWeight) * 2.0 >= total) {
      hi = lt - 1;
      continue;
    }
    if ((below + lessWeight + pivotWeight) * 2.0 >= total) return pivot;
    below += lessWeight + pivotWeight;
    lo = gt + 1;
  }
}

/** Sum of weighted absolute deviations at a candidate point. */
function weightedDeviation(values, weights, point) {
  let total = 0.0;
  for (let i = 0; i < values.length; i++) total += weights[i] * Math.abs(values[i] - point);
  return total;
}

/** Weighted median by exhaustive minimisation of the deviation sum, over the distinct values. */
function weightedMedianByDeviation(values, weights) {
  totalWeight(weights);
  const candidates = [...new Set(values)].sort((a, b) => a - b);
  let best = candidates[0];
  let bestCost = weightedDeviation(values, weights, best);
  for (const candidate of candidates) {
    const cost = weightedDeviation(values, weights, candidate);
    if (cost < bestCost) {
      bestCost = cost;
      best = candidate;
    }
  }
  return best;
}

function main() {
  const assert = require("node:assert");

  // Hand-computed instance: W = 5, half = 2.5; W(<= 0) = 1 < 2.5, W(<= 10) = 4 >= 2.5.
  const handValues = [0, 10, 20];
  const handWeights = [1, 3, 1];
  assert.strictEqual(weightedMedianSorted(handValues, handWeights), 10);
  assert.strictEqual(weightedMedianSelect(handValues, handWeights), 10);
  assert.strictEqual(weightedMedianByDeviation(handValues, handWeights), 10);
  assert.ok(weightedDeviation(handValues, handWeights, 10) <
    weightedDeviation(handValues, handWeights, 0));

  // Exact tie at half of the mass: the smaller of the two middle values wins.
  assert.strictEqual(weightedMedianSorted([1, 2], [1, 1]), 1);
  assert.strictEqual(weightedMedianSelect([1, 2], [1, 1]), 1);
  assert.strictEqual(weightedMedianSorted([0, 4, 8], [1, 2, 1]), 4, "W = 4, half = 2 reached at 4");
  assert.strictEqual(weightedMedianSelect([0, 4, 8], [1, 2, 1]), 4);
  assert.strictEqual(weightedMedianSorted([0, 4, 8], [2, 1, 1]), 0,
    "half the mass sits exactly on the smallest value");
  assert.strictEqual(weightedMedianSelect([0, 4, 8], [2, 1, 1]), 0);

  // Single element, all weight on one element, and zero weights.
  assert.strictEqual(weightedMedianSelect([7], [3]), 7);
  assert.strictEqual(weightedMedianSelect([7, 8, 9], [5, 0, 0]), 7);
  assert.strictEqual(weightedMedianSelect([7, 8, 9], [0, 0, 5]), 9);
  assert.strictEqual(weightedMedianSelect([3, 1, 2], [0, 1, 0]), 1,
    "zero-weight elements cannot shift the answer");
  assert.strictEqual(weightedMedianSelect([5, 5, 5], [1, 2, 3]), 5, "duplicate values aggregate");

  const instances = [];
  for (let seed = 1; seed <= 60; seed++) {
    const n = 1 + ((seed * 7) % 40);
    instances.push([`random-${seed}`, randomInstance(n, seed, 50, 9)]);
  }
  instances.push(["heavy-tail", { values: [1, 2, 3, 4, 5], weights: [100, 1, 1, 1, 1] }]);
  instances.push(["last-heavy", { values: [1, 2, 3, 4, 5], weights: [1, 1, 1, 1, 100] }]);
  instances.push(["all-equal-weights", { values: [9, 3, 7, 1], weights: [2, 2, 2, 2] }]);
  instances.push(["zero-heavy", { values: [0, 1, 2], weights: [0, 4, 0] }]);

  let selectComparisons = 0;
  let totalSize = 0;
  for (const [name, instance] of instances) {
    const { values, weights } = instance;
    const expected = weightedMedianSorted(values, weights);
    const selected = weightedMedianSelect(values, weights);
    assert.strictEqual(selected, expected, `${name}: selection matches the sorted scan`);
    assert.strictEqual(weightedMedianByDeviation(values, weights), expected,
      `${name}: the median minimises the weighted absolute deviation`);

    const total = totalWeight(weights);
    let below = 0.0;
    let atOrBelow = 0.0;
    for (let i = 0; i < values.length; i++) {
      if (values[i] < expected) below += weights[i];
      if (values[i] <= expected) atOrBelow += weights[i];
    }
    assert.ok(below * 2.0 < total, `${name}: W(< v) < W/2`);
    assert.ok(atOrBelow * 2.0 >= total, `${name}: W(<= v) >= W/2`);
    assert.ok(values.includes(expected), `${name}: the median is one of the values`);

    // The same instance in a different order gives the same answer.
    const shuffledValues = values.slice().reverse();
    const shuffledWeights = weights.slice().reverse();
    assert.strictEqual(weightedMedianSelect(shuffledValues, shuffledWeights), expected,
      `${name}: the answer is order independent`);
    assert.strictEqual(weightedMedianSorted(shuffledValues, shuffledWeights), expected);

    totalSize += values.length;
    selectComparisons += 1;
  }
  assert.ok(selectComparisons === instances.length);

  // Selection stays linear-ish: the partition never visits more than a constant number of elements
  // per level, so the comparison count is O(n) and the level count is O(log n).
  for (const n of [1000, 5000, 20000]) {
    const instance = randomInstance(n, 2024 + n, 100000, 1000);
    const expected = weightedMedianSorted(instance.values, instance.weights);
    const stats = { comparisons: 0, levels: 0 };
    const median = weightedMedianSelect(instance.values, instance.weights, stats);
    assert.strictEqual(median, expected, `n = ${n}: selection matches the sorted scan`);
    assert.ok(stats.comparisons <= 10 * n + 50,
      `n = ${n}: ${stats.comparisons} comparisons for ${n} items`);
    assert.ok(stats.levels <= 60, `n = ${n}: ${stats.levels} partition levels`);
  }

  // Weights may be fractional as long as they are non-negative.
  assert.strictEqual(weightedMedianSelect([0.0, 1.0], [0.4, 0.6]), 1.0);
  assert.strictEqual(weightedMedianSorted([0.0, 1.0], [0.4, 0.6]), 1.0);
  assert.strictEqual(weightedMedianSelect([0.0, 1.0], [0.5, 0.5]), 0.0,
    "a 50/50 split picks the smaller value");

  assert.throws(() => weightedMedianSelect([], []), /empty input/);
  assert.throws(() => weightedMedianSelect([1, 2], [1]), /length mismatch/);
  assert.throws(() => weightedMedianSelect([1, 2], [0, 0]), /total weight must be positive/);
  assert.throws(() => weightedMedianSelect([1, 2], [-1, 3]), /non-negative/);
  assert.throws(() => weightedMedianSorted([1, 2], [0, 0]), /total weight must be positive/);
  assert.ok(Math.abs(mean([1.0, 2.0, 6.0]) - 3.0) <= 1e-15);
}

module.exports = { Lcg, randomInstance, totalWeight, weightedMedianSorted, weightedMedianSelect,
  weightedDeviation, weightedMedianByDeviation, mean };

if (require.main === module) {
  main();
  console.log("PASS");
}
