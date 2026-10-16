/**
 * Task 522 - Implement bootstrap sampling estimator for mean CI. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Non-parametric bootstrap confidence interval for the sample mean. B resamples are drawn with
 * replacement (each of the n observations picked independently per draw), the mean of every
 * resample is recorded, and three intervals are derived from the same distribution: the percentile
 * interval (quantiles alpha/2 and 1 - alpha/2 of the resampled means, computed with the type-7
 * linear interpolation h = (n - 1)p), the basic/"reverse percentile" interval
 * (2*mean - q_{1-a/2}, 2*mean - q_{a/2}) and the normal-approximation interval
 * mean +/- z * SE_boot with the bootstrap standard error. All randomness comes from a seeded 32-bit
 * LCG whose state is scrambled on construction, so every reported interval is reproducible.
 *
 * Documented checks: the bootstrap standard error is compared with the closed form s/sqrt(n)
 * (within 5% for n = 50), and for the two-point sample {1, 2} the exact resample-mean distribution
 * (1.0 with multiplicity 1, 1.5 with multiplicity 2, 2.0 with multiplicity 1) is enumerated
 * exhaustively and its probabilities are matched by the LCG draw to within 0.01. The percentile and
 * basic intervals are verified to be mirror images about the sample mean (basic_lo =
 * 2m - pct_hi, basic_hi = 2m - pct_lo) and the quantile helper is checked against hand-computed
 * type-7 values and monotonicity in p. A constant sample must collapse to a zero-width interval at
 * the constant, the interval must narrow as n grows (width(n = 320) < width(n = 20), a ratio
 * near sqrt(16)), and a
 * 300-replication coverage experiment on uniform(0,1) data (n = 20, B = 400, nominal 95%) must land
 * inside the documented band [0.87, 0.99] around the true mean 0.5.
 */
"use strict";

/** Seeded 32-bit LCG with a scrambled state so that neighbouring seeds are independent. */
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

  nextFloat() {
    return this.nextUint32() / 4294967296;
  }

  nextInt(bound) {
    return Math.floor(this.nextFloat() * bound);
  }
}

function mean(values) {
  if (values.length === 0) throw new Error("empty sample");
  return values.reduce((a, b) => a + b, 0.0) / values.length;
}

/** Unbiased sample standard deviation (n - 1 denominator); 0 for a single observation. */
function stdev(values) {
  const n = values.length;
  if (n < 2) return 0.0;
  const m = mean(values);
  const ss = values.reduce((acc, v) => acc + (v - m) * (v - m), 0.0);
  return Math.sqrt(ss / (n - 1));
}

/** One bootstrap resample: n draws with replacement. */
function resample(values, rng) {
  const n = values.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = values[rng.nextInt(n)];
  return out;
}

function bootstrapMeans(values, B, rng) {
  if (!Number.isInteger(B) || B <= 0) throw new Error("B must be a positive integer");
  const out = new Array(B);
  for (let b = 0; b < B; b++) out[b] = mean(resample(values, rng));
  return out;
}

/** Type-7 quantile: h = (n - 1) * p, linear interpolation between the neighbouring order stats. */
function quantileSorted(sorted, p) {
  if (sorted.length === 0) throw new Error("empty sample");
  if (p < 0.0 || p > 1.0) throw new Error("p must lie in [0, 1]");
  const h = (sorted.length - 1) * p;
  const lo = Math.floor(h);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

function quantile(values, p) {
  return quantileSorted(values.slice().sort((a, b) => a - b), p);
}

const Z_95 = 1.959963984540054;

/** Percentile, basic and normal intervals from one bootstrap run (single RNG pass). */
function meanCI(values, B, rng, alpha = 0.05) {
  const means = bootstrapMeans(values, B, rng).sort((a, b) => a - b);
  const sampleMean = mean(values);
  const pctLo = quantileSorted(means, alpha / 2.0);
  const pctHi = quantileSorted(means, 1.0 - alpha / 2.0);
  const se = stdev(means);
  const z = Z_95;
  return {
    n: values.length, B, sampleMean, se,
    sdOverSqrtN: stdev(values) / Math.sqrt(values.length),
    percentile: [pctLo, pctHi],
    basic: [2.0 * sampleMean - pctHi, 2.0 * sampleMean - pctLo],
    normal: [sampleMean - z * se, sampleMean + z * se],
    medianMean: quantileSorted(means, 0.5),
  };
}

/** Exact distribution of the resample mean: all n^n resamples with their multiplicities. */
function exactResampleMeanDistribution(values) {
  const n = values.length;
  const table = new Map();
  const counts = new Array(n).fill(0);
  const walk = (depth, sum) => {
    if (depth === n) {
      const m = sum / n;
      const key = m.toFixed(12);
      table.set(key, (table.get(key) || 0) + 1);
      return;
    }
    for (let i = 0; i < n; i++) walk(depth + 1, sum + values[i]);
  };
  walk(0, 0.0);
  let total = 0;
  for (const count of table.values()) total += count;
  if (total !== n ** n) throw new Error("enumeration lost resamples");
  return table;
}

function main() {
  const assert = require("node:assert");

  assert.ok(Math.abs(mean([1.0, 2.0, 3.0, 4.0]) - 2.5) <= 1e-15);
  assert.ok(Math.abs(stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) -
    Math.sqrt(32.0 / 7.0)) <= 1e-12, "documented n - 1 example");
  assert.strictEqual(stdev([3.0]), 0.0);

  assert.ok(Math.abs(quantileSorted([1.0, 2.0, 3.0, 4.0], 0.25) - 1.75) <= 1e-15,
    "type-7: h = 0.75 between 1 and 2");
  assert.strictEqual(quantileSorted([0.0, 1.0, 2.0, 3.0, 4.0], 0.5), 2.0);
  assert.strictEqual(quantile([5.0, 1.0, 3.0], 0.0), 1.0);
  assert.strictEqual(quantile([5.0, 1.0, 3.0], 1.0), 5.0);
  let previous = -Infinity;
  for (let i = 0; i <= 20; i++) {
    const q = quantile([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], i / 20.0);
    assert.ok(q >= previous - 1e-15, "the quantile function is non-decreasing in p");
    previous = q;
  }
  assert.throws(() => quantile([1.0], 1.5), /p must lie/);
  assert.throws(() => mean([]), /empty sample/);
  assert.throws(() => bootstrapMeans([1.0], 0, new Lcg(1)), /positive integer/);

  // Two-point sample: the exact resample-mean law is {1.0: 1, 1.5: 2, 2.0: 1} out of 4.
  const exact = exactResampleMeanDistribution([1.0, 2.0]);
  assert.deepStrictEqual([...exact.entries()].sort(), [["1.000000000000", 1],
    ["1.500000000000", 2], ["2.000000000000", 1]]);

  const B = 20000;
  const draws = bootstrapMeans([1.0, 2.0], B, new Lcg(2024));
  const counts = { 1.0: 0, 1.5: 0, 2.0: 0 };
  for (const m of draws) {
    assert.ok(counts[m] !== undefined, `unexpected resample mean ${m}`);
    counts[m] += 1;
  }
  assert.ok(Math.abs(counts[1.0] / B - 0.25) <= 0.01, "P(mean = 1.0) = 1/4");
  assert.ok(Math.abs(counts[2.0] / B - 0.25) <= 0.01, "P(mean = 2.0) = 1/4");
  assert.ok(Math.abs(counts[1.5] / B - 0.5) <= 0.01, "P(mean = 1.5) = 1/2");
  assert.ok(Math.abs(mean(draws) - 1.5) <= 4.0 * Math.sqrt(0.125 / B),
    "the bootstrap mean is centred on the sample mean (within 4 Monte-Carlo standard errors)");

  // Constant sample: every resample equals the constant, so all three intervals are zero-width.
  const constant = meanCI([7.5, 7.5, 7.5, 7.5], 200, new Lcg(11));
  assert.strictEqual(constant.se, 0.0);
  assert.deepStrictEqual(constant.percentile, [7.5, 7.5]);
  assert.deepStrictEqual(constant.basic, [7.5, 7.5]);
  assert.deepStrictEqual(constant.normal, [7.5, 7.5]);

  // Bootstrap SE versus the closed form s/sqrt(n).
  const sample = [];
  let seed = 5;
  for (let i = 0; i < 50; i++) {
    seed = (seed * 75 + 74) % 65537;
    sample.push(seed / 65537.0);
  }
  const ci = meanCI(sample, 4000, new Lcg(31337));
  assert.ok(Math.abs(ci.se - ci.sdOverSqrtN) <= 0.05 * ci.sdOverSqrtN,
    "bootstrap SE approximates s/sqrt(n)");
  assert.ok(Math.abs(ci.sampleMean - mean(sample)) <= 1e-15);

  // Percentile and basic intervals are mirror images about the sample mean.
  assert.ok(Math.abs(ci.basic[0] - (2.0 * ci.sampleMean - ci.percentile[1])) <= 1e-12);
  assert.ok(Math.abs(ci.basic[1] - (2.0 * ci.sampleMean - ci.percentile[0])) <= 1e-12);
  assert.ok(ci.percentile[0] <= ci.medianMean + 1e-15 && ci.percentile[1] >= ci.medianMean - 1e-15,
    "the resample median lies inside the percentile interval");
  assert.ok(ci.normal[0] < ci.sampleMean && ci.sampleMean < ci.normal[1]);

  // Determinism, and sensitivity to the seed only in the details of the draw.
  const repeat = meanCI(sample, 4000, new Lcg(31337));
  assert.deepStrictEqual(repeat.percentile, ci.percentile);
  assert.deepStrictEqual(repeat.basic, ci.basic);
  const other = meanCI(sample, 4000, new Lcg(31338));
  assert.notDeepStrictEqual(other.percentile, ci.percentile, "a different seed shifts the draw");
  assert.ok(Math.abs(other.percentile[0] - ci.percentile[0]) <= 0.15 * ci.se + 1e-9,
    "but the interval is stable up to Monte-Carlo noise (SE/sqrt(B))");

  // Width shrinks like 1/sqrt(n).
  const big = [];
  seed = 5;
  for (let i = 0; i < 320; i++) {
    seed = (seed * 75 + 74) % 65537;
    big.push(seed / 65537.0);
  }
  const small = big.slice(0, 20);
  const smallCI = meanCI(small, 2000, new Lcg(42));
  const bigCI = meanCI(big, 2000, new Lcg(42));
  const width = (interval) => interval[1] - interval[0];
  assert.ok(width(bigCI.percentile) < width(smallCI.percentile),
    "n = 320 gives a tighter percentile interval than n = 20");
  const ratio = width(smallCI.percentile) / width(bigCI.percentile);
  assert.ok(ratio > 2.0 && ratio < 7.0, `width ratio ${ratio} tracks sqrt(320/20) = 4`);

  // Coverage: 300 replications of n = 20 uniform(0,1) samples, nominal 95% percentile interval.
  const replications = 300;
  let covered = 0;
  let coveredNormal = 0;
  for (let r = 0; r < replications; r++) {
    const master = new Lcg(1000 + r);
    const values = [];
    for (let i = 0; i < 20; i++) values.push(master.nextFloat());
    const rep = meanCI(values, 400, new Lcg(90001 + r));
    if (rep.percentile[0] <= 0.5 && 0.5 <= rep.percentile[1]) covered += 1;
    if (rep.normal[0] <= 0.5 && 0.5 <= rep.normal[1]) coveredNormal += 1;
  }
  const coverage = covered / replications;
  const coverageNormal = coveredNormal / replications;
  assert.ok(coverage >= 0.87 && coverage <= 0.99, `percentile coverage ${coverage} near 0.95`);
  assert.ok(coverageNormal >= 0.87 && coverageNormal <= 0.99,
    `normal-approximation coverage ${coverageNormal} near 0.95`);
  assert.ok(Math.abs(coverage - coverageNormal) <= 0.06,
    "the two interval families agree within Monte-Carlo noise");
}

module.exports = { Lcg, mean, stdev, resample, bootstrapMeans, quantileSorted, quantile, Z_95,
  meanCI, exactResampleMeanDistribution };

if (require.main === module) {
  main();
  console.log("PASS");
}
