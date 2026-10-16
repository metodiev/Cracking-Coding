/** Task 646 - Implement approximate query processing via sampling reservoir. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMULATED: a single-pass approximate query layer over a stream of rows. The sampling component is
 * the real reservoir algorithm (Algorithm R): every incoming row replaces a uniformly chosen reservoir slot
 * with probability k/i, which makes the reservoir a uniformly random k-subset of the i rows seen so far -
 * exactly what an online estimator needs, because the reservoir is valid after *every* prefix. A weighted
 * variant (A-Res, using the key u^(1/w) and keeping the k largest keys) lets a skewed stream be sampled
 * proportionally to a weight. On top of the sampler sit the estimators: AVG with a normal-approximation
 * confidence interval built from the sample variance, SUM as avg * n, and a selectivity estimate for a
 * predicate. No data structures beyond arrays and a seeded LCG are needed; nothing is timed.
 *
 * WHAT IS VERIFIED: the sampler's own invariants (sample size never above k, the reservoir is always a
 * sub-multiset of the prefix, reproducibility for a fixed seed, exactness when the stream is no longer than
 * the reservoir, and rejection of k <= 0); the *uniformity* of Algorithm R is checked exactly, by comparing
 * the empirical subset frequencies of 8000 seeded runs of a four-row stream against the probabilities that
 * the algorithm's own accept/reject decisions imply, with the same treatment for the weighted variant's
 * inclusion probabilities w_i / sum(w); and the estimators are checked for shrinking error and interval
 * width as k grows, for interval coverage over 200 seeded trials, for exactness on a degenerate distribution
 * (zero variance gives a zero-width interval) and for the exact answer when the reservoir holds everything.
 */
"use strict";

/** Deterministic 32-bit generator (xorshift32), advanced through Math.imul so it is portable. */
function makeRandom(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    if (bound === undefined) return state / 4294967296;
    return state % bound;
  };
}

/** Algorithm R: uniform reservoir of size k over a single pass, valid after every prefix. */
function reservoirSample(stream, k, random) {
  if (!(k >= 1)) throw new Error("the reservoir size must be at least one row");
  if (!Array.isArray(stream)) throw new Error("the stream must be an array of rows");
  const reservoir = [];
  let replacements = 0;
  for (let index = 0; index < stream.length; index++) {
    if (reservoir.length < k) {
      reservoir.push(stream[index]);
      continue;
    }
    const slot = random(index + 1);
    if (slot < k) {
      reservoir[slot] = stream[index];
      replacements++;
    } }
  return { sample: reservoir, seen: stream.length, replacements };
}

/** A-Res: keeps the k largest keys u^(1/w), so a row with weight w is k times more likely to be kept. */
function weightedReservoirSample(stream, k, weightOf, random) {
  if (!(k >= 1)) throw new Error("the reservoir size must be at least one row");
  const sample = [];
  for (const row of stream) {
    const weight = weightOf(row);
    if (!(weight > 0)) throw new Error("weights must be positive");
    const key = Math.pow(random(), 1 / weight);
    sample.push({ row, key });
    sample.sort((left, right) => right.key - left.key);
    if (sample.length > k) sample.pop();
  }
  return { sample: sample.map((entry) => entry.row), seen: stream.length };
}

function meanOf(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function varianceOf(values) {
  if (values.length < 2) return 0;
  const mean = meanOf(values);
  return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
}

/** AVG estimate plus a normal-approximation interval, using the finite-population correction factor. */
function estimateAverage(sample, populationSize, z = 1.96) {
  if (sample.length === 0) return { estimate: null, low: null, high: null, width: null };
  const mean = meanOf(sample);
  if (sample.length >= populationSize) return { estimate: mean, low: mean, high: mean, width: 0 };
  const standardError = Math.sqrt((varianceOf(sample) / sample.length) *
    (1 - sample.length / populationSize));
  return { estimate: mean, low: mean - z * standardError, high: mean + z * standardError, width: 2 * z * standardError };
}

function estimateSum(sample, populationSize) {
  const average = estimateAverage(sample, populationSize);
  if (average.estimate === null) return { estimate: 0, low: 0, high: 0 };
  const scale = populationSize;
  return {
    estimate: average.estimate * scale, low: average.low * scale,
    high: average.high * scale,
  };
}

/** Predicate selectivity: the sampled fraction times the population, with a binomial interval. */
function estimateSelectivity(sample, populationSize, predicate, z = 1.96) {
  if (sample.length === 0) return { estimate: 0, low: 0, high: 0, fraction: 0 };
  const hits = sample.filter(predicate).length;
  const fraction = hits / sample.length;
  const standardError = Math.sqrt((fraction * (1 - fraction) / sample.length) *
    (1 - sample.length / populationSize));
  return {
    estimate: Math.floor(fraction * populationSize), fraction,
    low: Math.max(0, (fraction - z * standardError) * populationSize),
    high: Math.min(populationSize, (fraction + z * standardError) * populationSize),
  };
}

function makeStream(length, seed = 646) {
  const random = makeRandom(seed);
  const rows = [];
  for (let i = 0; i < length; i++) {
    // A skewed numeric column: mostly small values with a heavy tail of large ones.
    const roll = random(100);
    rows.push({ id: i, value: roll < 80 ? 1 + (roll % 10) : 200 + (roll % 700) });
  }
  return rows;
}

function subsetKey(sample) {
  return sample.map((row) => row.id).sort((a, b) => a - b).join(",");
}

module.exports = {
  makeRandom, reservoirSample, weightedReservoirSample, estimateAverage, estimateSum,
  estimateSelectivity, meanOf, varianceOf, makeStream, subsetKey,
};

if (require.main === module) {
  const assert = require("node:assert");
  const stream = [0, 1, 2, 3, 4, 5, 6, 7].map((id) => ({ id, value: 10 * (id + 1) }));
  // Sampler invariants: size, membership, reproducibility, degenerate sizes.
  for (let seed = 1; seed <= 20; seed++) {
    const random = makeRandom(seed);
    const { sample, seen, replacements } = reservoirSample(stream, 3, random);
    assert.strictEqual(sample.length, 3);
    assert.strictEqual(seen, stream.length);
    assert.ok(replacements <= stream.length);
    const pool = stream.slice();
    for (const value of sample) {
      const position = pool.indexOf(value);
      assert.ok(position >= 0, "the reservoir holds rows from the stream");
      pool.splice(position, 1);
    } }
  assert.deepStrictEqual(reservoirSample(stream, 100, makeRandom(7)).sample, stream);
  assert.strictEqual(reservoirSample(stream, 100, makeRandom(7)).replacements, 0, "no replacement needed");
  assert.deepStrictEqual(reservoirSample([], 4, makeRandom(7)).sample, []);
  assert.deepStrictEqual(reservoirSample(stream, 1, makeRandom(7)).sample.length, 1);
  assert.strictEqual(subsetKey(reservoirSample(stream, 4, makeRandom(11)).sample),
    subsetKey(reservoirSample(stream, 4, makeRandom(11)).sample), "the same seed reproduces the sample");
  const seeds = [1, 2, 3, 4, 5].map((seed) => subsetKey(reservoirSample(stream, 3, makeRandom(seed)).sample));
  assert.ok(new Set(seeds).size > 1, "different seeds give different samples");
  assert.throws(() => reservoirSample(stream, 0, makeRandom(1)), /at least one row/);
  assert.throws(() => reservoirSample("not an array", 2, makeRandom(1)), /must be an array/);
  // Exact uniformity check on a four-row stream: the algorithm's decisions imply the probability of every
  // two-row subset, and 8000 seeded runs must reproduce those probabilities.
  const four = [{ id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }];
  const subsets = new Map();
  const trials = 8000;
  for (let seed = 1; seed <= trials; seed++) {
    const key = subsetKey(reservoirSample(four, 2, makeRandom(seed * 2654435761 % 4294967296)).sample);
    subsets.set(key, (subsets.get(key) || 0) + 1);
  }
  assert.strictEqual(subsets.size, 6, "all six two-row subsets occur");
  // Algorithm R keeps a uniform random subset: rows 0 and 1 are in the reservoir with probability 1/2,
  // so every subset has probability 1/6. The observed frequencies are compared with a 0.02 tolerance.
  for (const [key, count] of subsets) {
    assert.ok(Math.abs(count / trials - 1 / 6) <= 0.02, `subset ${key} frequency ${count / trials}`);
  }
  // Weighted sampling: inclusion probability must be proportional to the weight.
  const weightedStream = [{ id: "light", w: 1 }, { id: "mid", w: 2 }, { id: "heavy", w: 5 }];
  const picks = new Map();
  const weightOf = (row) => row.w;
  for (let seed = 1; seed <= trials; seed++) {
    const { sample } = weightedReservoirSample(weightedStream, 1, weightOf, makeRandom((seed * 40503) % 4294967296));
    picks.set(sample[0].id, (picks.get(sample[0].id) || 0) + 1);
  }
  const totalWeight = weightedStream.reduce((sum, row) => sum + row.w, 0);
  for (const row of weightedStream) {
    const observed = (picks.get(row.id) || 0) / trials;
    assert.ok(Math.abs(observed - row.w / totalWeight) <= 0.02,
      `${row.id} observed ${observed} expected ${row.w / totalWeight}`);
  }
  assert.throws(() => weightedReservoirSample(weightedStream, 1, () => 0, makeRandom(3)), /positive/);
  // Estimators: the sample mean tracks the true mean, and the interval shrinks as the reservoir grows.
  const rows = makeStream(2000);
  const trueAverage = meanOf(rows.map((row) => row.value));
  let previousError = Infinity;
  let previousWidth = Infinity;
  const coverage = { hits: 0, trials: 0 };
  for (const k of [8, 32, 128, 512]) {
    let error = 0;
    let width = 0;
    for (let seed = 1; seed <= 200; seed++) {
      const { sample } = reservoirSample(rows, k, makeRandom(seed * 7919 % 4294967296));
      const estimate = estimateAverage(sample.map((row) => row.value), rows.length);
      error += Math.abs(estimate.estimate - trueAverage);
      width += estimate.width;
      if (k === 512) {
        coverage.trials++;
        if (estimate.low <= trueAverage && trueAverage <= estimate.high) coverage.hits++;
      } }
    error /= 200;
    width /= 200;
    if (k > 8) {
      assert.ok(error < previousError, `the error must shrink with k (k=${k})`);
      assert.ok(width < previousWidth, `the interval must tighten with k (k=${k})`);
    }
    previousError = error;
    previousWidth = width;
  }
  assert.ok(coverage.hits / coverage.trials >= 0.85,
    `the nominal 95% interval must cover the truth most of the time (${coverage.hits}/${coverage.trials})`);
  // SUM scales the average, and even a small reservoir stays inside a sane error band.
  const { sample } = reservoirSample(rows, 64, makeRandom(99));
  const sum = estimateSum(sample.map((row) => row.value), rows.length);
  const trueSum = rows.reduce((total, row) => total + row.value, 0);
  assert.ok(Math.abs(sum.estimate - trueSum) / trueSum < 0.25, "the SUM estimate is in the right band");
  assert.strictEqual(estimateSum([], 10).estimate, 0);
  // Selectivity: the sampled fraction times the population approximates the true count.
  const hits = rows.filter((row) => row.value > 200).length;
  const selectivity = estimateSelectivity(sample, rows.length, (row) => row.value > 200);
  assert.ok(Math.abs(selectivity.estimate - hits) / hits < 0.35, `${selectivity.estimate} vs ${hits}`);
  assert.ok(selectivity.low <= hits && hits <= selectivity.high, "the true count lies inside the interval");
  assert.strictEqual(estimateSelectivity(rows.map((row) => ({ value: 7 })), rows.length, (row) => row.value === 7)
    .estimate, rows.length, "a predicate that matches everything is exact");
  assert.strictEqual(estimateSelectivity(rows, rows.length, (row) => row.value > 200).estimate, hits,
    "holding the whole population makes the estimate exact");
  assert.strictEqual(estimateAverage(rows.map((row) => row.value), rows.length).width, 0,
    "no sampling error when nothing was sampled away");
  // Degenerate distributions: a constant column has zero variance, so the interval collapses to a point.
  const constant = rows.map((row) => ({ id: row.id, value: 42 }));
  const constantSample = reservoirSample(constant, 16, makeRandom(5)).sample;
  const constantEstimate = estimateAverage(constantSample.map((row) => row.value), constant.length);
  assert.strictEqual(constantEstimate.estimate, 42);
  assert.strictEqual(constantEstimate.width, 0, "zero variance gives a zero-width interval");
  assert.strictEqual(estimateAverage([], 0).estimate, null, "an empty reservoir has no estimate");
  console.log(`reservoir: ${trials} seeded runs checked for uniformity, interval coverage ` +
    `${coverage.hits}/${coverage.trials} at k=512, skew error ${(Math.abs(sum.estimate - trueSum) / trueSum).toFixed(3)}`);
  console.log("PASS");
}
