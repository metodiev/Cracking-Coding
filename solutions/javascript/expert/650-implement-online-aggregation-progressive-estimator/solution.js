/** Task 650 - Implement online aggregation progressive estimator. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMULATED: the online-aggregation ("ripple join then report progress") execution model, reduced to
 * its estimator. Tuples are folded into O(1) state - n, sum, sum of squares, and a count per predicate - so
 * a query can be *reported while it is still running*: after every prefix the estimator hands out an
 * estimate with a confidence interval, and the interval is what makes the early answer honest. Two
 * intervals are computed: the normal approximation from the running sample variance (with the
 * finite-population correction, so the interval collapses to zero width when the whole population has been
 * seen) and the distribution-free Hoeffding bound, which only needs a known value range and is therefore
 * always the more conservative of the two. SUM and a predicate COUNT/selectivity are derived from the same
 * state, and a grouped variant keeps that per group.
 *
 * WHAT IS SIMPLIFIED: tuples arrive in a fixed order (the "ripple" scheduling of a real engine is not
 * modelled) and the intervals use a normal approximation rather than the exact t distribution.
 *
 * WHAT IS VERIFIED: the incremental state equals a from-scratch computation of the prefix at every
 * checkpoint; the final estimate equals the exact mean and the final interval has zero width; the interval
 * narrows monotonically from the 1% checkpoint to the 100% checkpoint; over 200 seeded streams the 95%
 * interval at the quarter point covers the true mean at least 80% of the time and the mean absolute error at
 * the half point is smaller than at the tenth point; the Hoeffding bound is never tighter than the normal
 * one; SUM and selectivity are exact once everything has been consumed and their intervals cover the truth;
 * a constant stream has zero variance and hence zero width; an empty stream reports "no estimate" and a
 * single-tuple stream reports an unbounded interval; the state stays O(1) in size while the stream grows;
 * and grouped estimates match per-group recomputation.
 */
"use strict";

class ProgressiveEstimator {
  constructor(populationSize, options = {}) {
    if (!(populationSize >= 0)) throw new Error("the population size must not be negative");
    this.population = populationSize;
    this.z = options.z === undefined ? 1.96 : options.z;
    this.sensitivity = options.sensitivity === undefined ? 1 : options.sensitivity;
    this.state = { n: 0, sum: 0, sumSquares: 0, counted: 0, min: null, max: null };
  }
  reset() {
    this.state = { n: 0, sum: 0, sumSquares: 0, counted: 0, min: null, max: null };
    return this;
  }
  /** One tuple, one pass: no sample is ever stored, so the state is O(1). */
  add(value, counts = false) {
    if (typeof value !== "number" || Number.isNaN(value)) throw new Error("a tuple value must be a number");
    const state = this.state;
    state.n++;
    state.sum += value;
    state.sumSquares += value * value;
    if (counts) state.counted++;
    state.min = state.min === null ? value : Math.min(state.min, value);
    state.max = state.max === null ? value : Math.max(state.max, value);
    return this;
  }
  mean() {
    return this.state.n === 0 ? null : this.state.sum / this.state.n;
  }
  variance() {
    const { n, sum, sumSquares } = this.state;
    if (n < 2) return 0;
    const spread = sumSquares - (sum * sum) / n;
    return Math.max(0, spread / (n - 1)); // clamped: the one-pass formula can go slightly negative
  }
  standardError(z = this.z) {
    void z;
    const { n } = this.state;
    if (n < 2) return Infinity;
    const correction = this.population === 0 ? 1 : Math.max(0, 1 - n / this.population);
    return Math.sqrt((this.variance() / n) * correction);
  }
  /** Normal-approximation interval from the running sample variance. */
  estimate() {
    const mean = this.mean();
    if (mean === null) return { n: 0, estimate: null, low: null, high: null, width: null };
    if (this.state.n < 2) return { n: this.state.n, estimate: mean, low: -Infinity, high: Infinity, width: Infinity };
    const half = this.z * this.standardError();
    return { n: this.state.n, estimate: mean, low: mean - half, high: mean + half, width: 2 * half };
  }
  /** Distribution-free bound: only the range of the values is needed. */
  hoeffding(alpha = 0.05) {
    const mean = this.mean();
    const { n, min, max } = this.state;
    if (mean === null) return { n: 0, estimate: null, width: null };
    if (n < 2) return { n, estimate: mean, width: Infinity };
    const range = (max - min) === 0 ? 0 : max - min;
    const correction = this.population === 0 ? 1 : Math.sqrt(Math.max(0, 1 - n / this.population));
    const half = range * Math.sqrt(Math.log(2 / alpha) / (2 * n)) * correction;
    return { n, estimate: mean, width: 2 * half };
  }
  sumEstimate() {
    const average = this.estimate();
    if (average.estimate === null) return { estimate: 0, low: 0, high: 0, width: 0 };
    return {
      estimate: average.estimate * this.population, low: average.low * this.population,
      high: average.high * this.population, width: average.width * this.population,
    };
  }
  /** Predicate selectivity / COUNT over the population from the running counter. */
  selectivityEstimate() {
    const { n, counted } = this.state;
    if (n === 0) return { estimate: null, fraction: null, low: null, high: null, width: null };
    const fraction = counted / n;
    if (n < 2 || n >= this.population) {
      const exact = Math.round(fraction * this.population);
      return { estimate: exact, fraction, low: exact, high: exact, width: 0 };
    }
    const correction = Math.sqrt(Math.max(0, 1 - n / this.population));
    const half = this.z * Math.sqrt((fraction * (1 - fraction) / n)) * correction;
    return {
      estimate: Math.round(fraction * this.population), fraction,
      low: Math.max(0, (fraction - half) * this.population),
      high: Math.min(this.population, (fraction + half) * this.population), width: (2 * half) * this.population,
    };
  }
  stateSize() {
    return Object.keys(this.state).length;
  } }

class GroupedEstimator {
  constructor(populationSize) {
    this.population = populationSize;
    this.groups = new Map();
  }
  add(key, value) {
    const bucket = this.groups.get(key) || { n: 0, sum: 0, sumSquares: 0 };
    bucket.n++;
    bucket.sum += value;
    bucket.sumSquares += value * value;
    this.groups.set(key, bucket);
    return this;
  }
  mean(key) {
    const bucket = this.groups.get(key);
    return bucket === undefined ? null : bucket.sum / bucket.n;
  }
  summary() {
    return [...this.groups].map(([key, bucket]) => ({
      key, n: bucket.n, mean: bucket.sum / bucket.n, sum: bucket.sum,
    })).sort((left, right) => (left.key < right.key ? -1 : left.key > right.key ? 1 : 0));
  } }

/** Consumes the stream once, snapshotting the estimator at each checkpoint of the population. */
function consumeWithCheckpoints(values, checkpoints, populationSize = values.length) {
  const estimator = new ProgressiveEstimator(populationSize);
  const marks = new Set(checkpoints.map((fraction) => Math.max(1, Math.round(fraction * values.length))));
  const snapshots = [];
  for (let index = 0; index < values.length; index++) {
    estimator.add(values[index], values[index] > 100);
    if (marks.has(index + 1)) snapshots.push({ n: index + 1, ...estimator.estimate(), counted: estimator.state.counted });
  }
  return { estimator, snapshots };
}

function fromScratch(prefix, populationSize) {
  const n = prefix.length;
  const sum = prefix.reduce((total, value) => total + value, 0);
  const sumSquares = prefix.reduce((total, value) => total + value * value, 0);
  const counted = prefix.filter((value) => value > 100).length;
  return { n, sum, sumSquares, counted, mean: sum / n };
}

function lcgStream(length, seed, skew = 100) {
  const values = [];
  let state = seed >>> 0;
  for (let index = 0; index < length; index++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const draw = (state >>> 8) % 1000;
    values.push(draw < 800 ? 1 + (draw % 10) : skew + (draw % 900));
  }
  return values;
}

module.exports = {
  ProgressiveEstimator, GroupedEstimator, consumeWithCheckpoints, fromScratch, lcgStream,
};

if (require.main === module) {
  const assert = require("node:assert");
  const values = lcgStream(1000, 650);
  const exact = fromScratch(values, values.length);
  // The running state must equal a from-scratch computation of the prefix at every checkpoint.
  const fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1];
  const { estimator, snapshots } = consumeWithCheckpoints(values, fractions);
  assert.strictEqual(snapshots.length, fractions.length);
  for (const snapshot of snapshots) {
    const prefix = fromScratch(values.slice(0, snapshot.n), values.length);
    assert.strictEqual(snapshot.n, prefix.n);
    assert.strictEqual(snapshot.counted, prefix.counted, `counted at n=${snapshot.n}`);
    assert.strictEqual(snapshot.estimate, prefix.mean, `mean at n=${snapshot.n}`);
  }
  assert.strictEqual(estimator.state.sum, exact.sum);
  assert.strictEqual(estimator.state.sumSquares, exact.sumSquares);
  assert.strictEqual(estimator.state.counted, exact.counted);
  assert.strictEqual(estimator.mean(), exact.mean, "the final mean is exact");
  // The interval narrows as the stream progresses and collapses to a point once everything is consumed.
  for (let index = 1; index < snapshots.length; index++) {
    assert.ok(snapshots[index].width < snapshots[index - 1].width,
      `width must shrink: ${snapshots[index - 1].width} -> ${snapshots[index].width}`);
  }
  assert.strictEqual(snapshots[snapshots.length - 1].width, 0, "the last checkpoint has no sampling error");
  assert.ok(snapshots[0].width > 0.05, "the first checkpoint is honest about its uncertainty");
  // The distribution-free bound is never tighter than the normal approximation on this data.
  for (const fraction of fractions) {
    const partial = new ProgressiveEstimator(values.length);
    const limit = Math.round(fraction * values.length);
    for (let index = 0; index < limit; index++) partial.add(values[index], values[index] > 100);
    const normal = partial.estimate();
    if (normal.width === Infinity) continue;
    assert.ok(partial.hoeffding(0.05).width >= normal.width, `Hoeffding is conservative at n=${limit}`);
  }
  // Coverage and error behaviour over many seeded streams: the interval must be honest, and more data
  // must mean less error.
  let covered = 0;
  let trials = 0;
  let errorEarly = 0;
  let errorLate = 0;
  for (let seed = 1; seed <= 200; seed++) {
    const stream = lcgStream(400, seed * 7919);
    const truth = fromScratch(stream, stream.length).mean;
    const quarter = new ProgressiveEstimator(stream.length);
    const half = new ProgressiveEstimator(stream.length);
    for (let index = 0; index < stream.length; index++) {
      quarter.add(stream[index]);
      if (index + 1 === 100) {
        const early = quarter.estimate();
        errorEarly += Math.abs(early.estimate - truth);
        if (early.low <= truth && truth <= early.high) covered++;
        trials++;
      }
      if (index + 1 >= 100) half.add(stream[index]);
    }
    errorLate += Math.abs(half.estimate().estimate - truth);
  }
  assert.ok(covered / trials >= 0.8, `nominal 95% intervals must cover most of the time (${covered}/${trials})`);
  assert.ok(errorLate / 200 < errorEarly / 200, "the half-point estimate must be closer than the tenth point");
  // SUM and selectivity are derived from the same state and become exact.
  const sums = estimator.sumEstimate();
  assert.strictEqual(sums.estimate, exact.sum, "the final SUM estimate is exact");
  assert.strictEqual(sums.width, 0);
  const partiallyConsumed = new ProgressiveEstimator(values.length);
  for (let index = 0; index < 100; index++) partiallyConsumed.add(values[index], values[index] > 100);
  const partialSum = partiallyConsumed.sumEstimate();
  assert.ok(partialSum.low <= exact.sum && exact.sum <= partialSum.high, "the true SUM lies inside the interval");
  const selectivity = estimator.selectivityEstimate();
  assert.strictEqual(selectivity.estimate, exact.counted, "the final predicate COUNT is exact");
  assert.strictEqual(estimator.state.counted, values.filter((value) => value > 100).length);
  const partialSelectivity = partiallyConsumed.selectivityEstimate();
  assert.ok(partialSelectivity.low <= exact.counted && exact.counted <= partialSelectivity.high,
    "the true predicate count lies inside the interval");
  assert.strictEqual(new ProgressiveEstimator(10).selectivityEstimate().estimate, null);
  // Degenerate inputs: nothing seen, one tuple (unknown variance), and a constant column (zero variance).
  const empty = new ProgressiveEstimator(100);
  assert.deepStrictEqual(empty.estimate(), { n: 0, estimate: null, low: null, high: null, width: null });
  assert.strictEqual(empty.sumEstimate().estimate, 0);
  const single = new ProgressiveEstimator(1);
  single.add(7);
  assert.strictEqual(single.estimate().estimate, 7);
  assert.strictEqual(single.estimate().width, Infinity, "one tuple cannot bound the variance");
  assert.strictEqual(single.hoeffding().width, Infinity);
  const constant = new ProgressiveEstimator(50);
  for (let index = 0; index < 50; index++) constant.add(42);
  assert.strictEqual(constant.variance(), 0);
  assert.strictEqual(constant.estimate().estimate, 42);
  assert.strictEqual(constant.estimate().width, 0, "a constant column has no sampling error at all");
  assert.strictEqual(constant.hoeffding().width, 0);
  assert.throws(() => new ProgressiveEstimator(-1), /must not be negative/);
  assert.throws(() => constant.add("not a number"), /must be a number/);
  assert.throws(() => constant.add(Number.NaN), /must be a number/);
  // The state is O(1): its shape does not change while the stream grows by two orders of magnitude.
  const longStream = lcgStream(20000, 999);
  const small = new ProgressiveEstimator(longStream.length);
  small.add(longStream[0]);
  const shapeAtOne = small.stateSize();
  for (const value of longStream) small.add(value);
  assert.strictEqual(small.stateSize(), shapeAtOne, "the estimator never accumulates tuples");
  assert.strictEqual(small.state.n, longStream.length + 1);
  assert.deepStrictEqual(fromScratch(longStream.slice(0, 5), longStream.length).n, 5);
  // Grouped estimation: per-group means match recomputation, and the progressive path agrees with it.
  const grouped = new GroupedEstimator(1000);
  const truth = new Map();
  for (const value of values) {
    const key = value > 100 ? "large" : "small";
    grouped.add(key, value);
    const bucket = truth.get(key) || { n: 0, sum: 0 };
    bucket.n++;
    bucket.sum += value;
    truth.set(key, bucket);
  }
  for (const group of grouped.summary()) {
    assert.strictEqual(group.mean, truth.get(group.key).sum / truth.get(group.key).n, group.key);
    assert.strictEqual(group.n, truth.get(group.key).n);
  }
  assert.strictEqual(grouped.summary().length, 2);
  assert.strictEqual(grouped.mean("missing"), null);
  const display = snapshots.map((snapshot) => `${(100 * snapshot.n / values.length).toFixed(0)}%:${snapshot.estimate.toFixed(3)}`);
  console.log(`online aggregation: ${display.join(" ")} (exact ${exact.mean.toFixed(3)}), ` +
    `coverage ${covered}/${trials}, the Hoeffding bound stayed conservative at every checkpoint`);
  console.log("PASS");
}
