/**
 * Task 533 - Implement linear-time median of medians selection. (Tier: Advanced, CTCI Ch 10 — Sorting and Searching)
 *
 * Deterministic order-statistic selection with the median-of-medians pivot rule: the array is split
 * into groups of at most five, each group's median is computed by a tiny insertion sort, the medians
 * are collected and the *pivot* is the median of that collection (found by the same routine applied
 * recursively), and the array is then partitioned in place with the Dutch-national-flag scheme into
 * "smaller", "equal" and "greater" blocks. Because at least 3/10 of the elements are greater than
 * (and at least 3/10 smaller than) the pivot, each recursive call discards at least 30% of the
 * window, which yields the classic O(n) worst-case bound with a linear number of comparisons; the
 * three-way partition additionally makes duplicate-heavy and already sorted inputs cheap. Every
 * element comparison is counted in an instrumented counter so the linear behaviour is asserted, not
 * assumed.
 *
 * Documented checks: for LCG arrays of many sizes, for floats, negatives, all-equal, two-valued,
 * already sorted, reverse sorted and organ-pipe inputs, the selected value equals the sorted
 * array's k-th element for *every* k (not just the median), including k = 0 and k = n - 1; the
 * answer is verified independently of sorting by the rank condition count(< v) <= k < count(<= v);
 * the selection is invariant under LCG shuffling of the input; the instrumentation shows at most
 * 30n + 100 comparisons - checked for random, sorted, reverse-sorted and duplicate-heavy inputs, so
 * the degenerate cases that ruin a naive two-way quickselect are covered - and the comparison count
 * grows sub-quadratically (comparisons(4n) <= 4.6 * comparisons(n)); the algorithm does not mutate
 * the caller's array; and invalid k is rejected.
 */
"use strict";

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

function lcgIntegers(n, seed, limit) {
  const rng = new Lcg(seed);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = rng.nextInt(limit);
  return out;
}

function shuffle(values, seed) {
  const rng = new Lcg(seed);
  const out = values.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

/** In-place insertion sort for groups of at most five; counts comparisons. */
function sortSmall(values, lo, hi, stats) {
  for (let i = lo + 1; i < hi; i++) {
    const value = values[i];
    let j = i - 1;
    while (j >= lo) {
      stats.comparisons += 1;
      if (values[j] <= value) break;
      values[j + 1] = values[j];
      j -= 1;
    }
    values[j + 1] = value;
  }
}

const GROUP = 5;

/**
 * Median-of-medians pivot for the window [lo, hi). Groups of at most five are sorted *in place*
 * (which preserves the multiset of the window) and their medians are collected into a separate
 * array, on which the same routine recurses - writing the medians back into the window would
 * destroy elements the later partition still needs.
 */
function medianOfMedians(values, lo, hi, stats) {
  const count = hi - lo;
  if (count <= GROUP) {
    sortSmall(values, lo, hi, stats);
    return values[lo + ((count - 1) >> 1)];
  }
  const medians = [];
  for (let start = lo; start < hi; start += GROUP) {
    const end = Math.min(start + GROUP, hi);
    sortSmall(values, start, end, stats);
    medians.push(values[start + ((end - start - 1) >> 1)]);
  }
  return medianOfMedians(medians, 0, medians.length, stats);
}

/** In-place three-way partition around `pivot`; returns [lt, gt] with [lt, gt] equal to the pivot. */
function partition(values, lo, hi, pivot, stats) {
  let lt = lo;
  let i = lo;
  let gt = hi - 1;
  while (i <= gt) {
    stats.comparisons += 1;
    if (values[i] < pivot) {
      const tmp = values[lt];
      values[lt] = values[i];
      values[i] = tmp;
      lt += 1;
      i += 1;
    } else if (values[i] > pivot) {
      const tmp = values[i];
      values[i] = values[gt];
      values[gt] = tmp;
      gt -= 1;
    } else {
      i += 1;
    }
  }
  return [lt, gt];
}

/** k-th smallest (0-based) by deterministic median-of-medians selection. */
function selectKth(values, k, stats = null) {
  if (!Number.isInteger(k) || k < 0 || k >= values.length) {
    throw new Error("k must be an integer index into the array");
  }
  const counters = stats === null ? { comparisons: 0, pivotLevels: 0 } : stats;
  const work = values.slice();
  const result = selectInPlace(work, 0, work.length, k, counters);
  if (stats === null) {
    return result;
  }
  return result;
}

function selectInPlace(work, lo, hi, k, stats) {
  for (;;) {
    if (hi - lo <= GROUP) {
      sortSmall(work, lo, hi, stats);
      return work[k];
    }
    const pivot = medianOfMedians(work, lo, hi, stats);
    stats.pivotLevels += 1;
    const [lt, gt] = partition(work, lo, hi, pivot, stats);
    if (k < lt) {
      hi = lt;
    } else if (k > gt) {
      lo = gt + 1;
    } else {
      return work[k];
    }
  }
}

/** Convention-free helper: the median for odd n, the lower median for even n. */
function median(values, stats = null) {
  return selectKth(values, (values.length - 1) >> 1, stats);
}

/** Independent rank verification that never sorts: count(< v) <= k < count(<= v). */
function verifyRank(values, k, value) {
  let below = 0;
  let atOrBelow = 0;
  for (const candidate of values) {
    if (candidate < value) below += 1;
    if (candidate <= value) atOrBelow += 1;
  }
  return below <= k && k < atOrBelow;
}

function main() {
  const assert = require("node:assert");

  assert.strictEqual(selectKth([1], 0), 1);
  assert.strictEqual(selectKth([3, 1], 0), 1);
  assert.strictEqual(selectKth([3, 1], 1), 3);
  assert.strictEqual(median([5, 3, 1]), 3);
  assert.strictEqual(median([5, 3, 1, 2]), 2, "even length: the lower median");
  assert.strictEqual(selectKth([0.5, -1.5, 2.25, 0.0], 0), -1.5);
  assert.strictEqual(selectKth([0.5, -1.5, 2.25, 0.0], 3), 2.25);

  const inputs = [];
  for (const n of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 31, 100, 257, 1000]) {
    inputs.push([`random-${n}`, lcgIntegers(n, 1 + n, Math.max(2, n))]);
  }
  inputs.push(["wide-range", lcgIntegers(500, 77, 1000000)]);
  inputs.push(["few-values", lcgIntegers(400, 91, 2)]);
  inputs.push(["all-equal", new Array(333).fill(7)]);
  inputs.push(["sorted", Array.from({ length: 300 }, (_, i) => i)]);
  inputs.push(["reverse", Array.from({ length: 300 }, (_, i) => 300 - i)]);
  inputs.push(["organ-pipe", Array.from({ length: 240 }, (_, i) => Math.min(i, 239 - i))]);
  inputs.push(["floats", Array.from({ length: 200 }, (_, i) => ((i * 37) % 101) / 3.0 - 5.0)]);
  inputs.push(["negative", lcgIntegers(150, 55, 41).map((v) => v - 20)]);

  let worstComparisonRatio = 0.0;
  for (const [name, values] of inputs) {
    const sorted = values.slice().sort((a, b) => a - b);
    let worstPerSelection = 0;
    for (let k = 0; k < values.length; k++) {
      const stats = { comparisons: 0, pivotLevels: 0 };
      const value = selectKth(values, k, stats);
      assert.strictEqual(value, sorted[k], `${name}: k = ${k}`);
      assert.ok(verifyRank(values, k, value), `${name}: rank condition at k = ${k}`);
      assert.ok(stats.comparisons <= 30 * values.length + 100, `${name}: k = ${k} took ` +
        `${stats.comparisons} comparisons for n = ${values.length}`);
      worstPerSelection = Math.max(worstPerSelection, stats.comparisons);
    }
    worstComparisonRatio = Math.max(worstComparisonRatio, worstPerSelection / values.length);
    const shuffled = shuffle(values, 99);
    assert.strictEqual(median(shuffled), sorted[(values.length - 1) >> 1],
      `${name}: shuffled input gives the same median`);
  }
  assert.ok(worstComparisonRatio <= 30,
    `worst observed comparisons per element: ${worstComparisonRatio}`);

  // The algorithm must not touch the caller's array.
  const original = lcgIntegers(200, 5, 50);
  const before = original.slice();
  selectKth(original, 42);
  assert.deepStrictEqual(original, before, "the input array is left untouched");

  // Linear-ish growth: four times the data costs well under sixteen times the work.
  const small = lcgIntegers(2000, 33, 5000);
  const large = lcgIntegers(8000, 34, 5000);
  const smallStats = { comparisons: 0, pivotLevels: 0 };
  const largeStats = { comparisons: 0, pivotLevels: 0 };
  selectKth(small, 1000, smallStats);
  selectKth(large, 4000, largeStats);
  assert.ok(largeStats.comparisons <= 4.6 * smallStats.comparisons,
    `comparisons grow sub-quadratically: ${smallStats.comparisons} -> ${largeStats.comparisons}`);
  assert.ok(largeStats.comparisons <= 30 * large.length + 100);

  // Sorted-ascending input is the classic quickselect worst case; the three-way partition with the
  // median-of-medians pivot keeps it linear.
  const ascending = Array.from({ length: 4000 }, (_, i) => i);
  const ascendingStats = { comparisons: 0, pivotLevels: 0 };
  assert.strictEqual(selectKth(ascending, 3999, ascendingStats), 3999);
  assert.ok(ascendingStats.comparisons <= 30 * ascending.length + 100,
    "ascending input stays linear");
  const duplicates = lcgIntegers(4000, 12, 3);
  const duplicateStats = { comparisons: 0, pivotLevels: 0 };
  selectKth(duplicates, 2000, duplicateStats);
  assert.ok(duplicateStats.comparisons <= 30 * duplicates.length + 100,
    "three-valued input stays linear thanks to the three-way partition");

  // Extremes: the minimum and maximum statistics.
  const extremes = lcgIntegers(777, 3, 10000);
  assert.strictEqual(selectKth(extremes, 0), Math.min(...extremes));
  assert.strictEqual(selectKth(extremes, extremes.length - 1), Math.max(...extremes));

  assert.throws(() => selectKth([1, 2, 3], -1), /k must be an integer/);
  assert.throws(() => selectKth([1, 2, 3], 3), /k must be an integer/);
  assert.throws(() => selectKth([1, 2, 3], 1.5), /k must be an integer/);
}

module.exports = { Lcg, lcgIntegers, shuffle, sortSmall, GROUP, medianOfMedians, partition,
  selectKth, median, verifyRank };

if (require.main === module) {
  main();
  console.log("PASS");
}
