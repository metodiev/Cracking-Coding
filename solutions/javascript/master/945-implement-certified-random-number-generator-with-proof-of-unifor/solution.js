/** Task 945 - Implement certified random number generator with proof of uniformity (small demo).
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * Two generators, two kinds of evidence. First a 16-bit full-period LCG whose state space is small
 * enough to enumerate: a linear congruential generator is a bijection of Z/2^k when c is odd and
 * a = 1 mod 4, so walking one whole period visits every state exactly once. That gives a *proof* of
 * uniformity for bucketing and for rejection sampling - below the rejection threshold the accepted
 * values are exactly equinumerous, which is checked by counting, not by statistics. Second a 64-bit
 * splitmix64 stream whose uniformity is certified statistically: chi-square over 100 bins with a
 * p-value, a Kolmogorov-Smirnov distance against the closed-form CDF, a runs test around the median
 * and a lag-1 serial correlation.
 *
 * Simplified: the certifiable generator is deliberately tiny (2^16 states) so the certificate is
 * exhaustive; the 64-bit stream is only certified statistically, and the chi-square p-value uses the
 * Wilson-Hilferty approximation rather than a full incomplete-gamma evaluation. Verified: exact
 * full-period completeness, exact bucket and rejection-sampler uniformity, the bias of the naive
 * modulo reduction, the four statistical certificates, seed-independence of the certificate, and
 * single-value/bucket boundaries.
 */
"use strict";
const assert = require("node:assert");

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}
const unit = (rng) => Number(rng.next() >> 11n) / 2 ** 53;

/** Full-period LCG on 2^16: c odd and a = 1 (mod 4) make it a permutation of the whole state space. */
const LCG_A = 20077n;
const LCG_C = 12345n;
const LCG_M = 1n << 16n;
function lcgCycle(seed = 0n) {
  const sequence = [];
  let x = BigInt.asUintN(16, seed);
  for (let i = 0; i < Number(LCG_M); i += 1) {
    x = (LCG_A * x + LCG_C) % LCG_M;
    sequence.push(Number(x));
  }
  return sequence;
}
/** Rejection sampler: drop the incomplete top slice so every bucket gets the same number of hits. */
function uniformBelow(sequence, n) {
  const limit = BigInt(Math.floor(Number(LCG_M) / n)) * BigInt(n);
  return sequence.filter((value) => BigInt(value) < limit).map((value) => value % n);
}
const counts = (values, n) => {
  const tally = new Array(n).fill(0);
  for (const value of values) tally[value] += 1;
  return tally;
};
const spread = (tally) => Math.max(...tally) - Math.min(...tally);

/** Abramowitz & Stegun 7.1.26 - the normal CDF used by the chi-square approximation. */
function erf(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const y = 1 - poly * Math.exp(-x * x);
  return x < 0 ? -y : y;
}
/** Wilson-Hilferty: (chi^2 / k)^(1/3) is approximately normal with the given mean and variance. */
function chiSquarePValue(statistic, bins) {
  const cube = Math.cbrt(statistic / bins);
  const mean = 1 - 2 / (9 * bins);
  const sd = Math.sqrt(2 / (9 * bins));
  return 1 - 0.5 * (1 + erf((cube - mean) / (sd * Math.SQRT2)));
}
function kolmogorovSmirnov(values) {
  const sorted = [...values].sort((a, b) => a - b);
  let worst = 0;
  sorted.forEach((value, i) => {
    const upper = (i + 1) / sorted.length;
    const lower = i / sorted.length;
    worst = Math.max(worst, Math.abs(upper - value), Math.abs(value - lower));
  });
  return worst;
}
function runsAroundMedian(values) {
  const median = [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
  let runs = 1;
  let previous = values[0] >= median;
  for (const value of values.slice(1)) {
    const above = value >= median;
    if (above !== previous) runs += 1;
    previous = above;
  }
  return runs;
}
const serialCorrelation = (values) => {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  let cov = 0;
  let variance = 0;
  for (let i = 0; i < values.length; i += 1) {
    cov += (values[i] - mean) * (values[(i + 1) % values.length] - mean);
    variance += (values[i] - mean) ** 2;
  }
  return cov / variance;
};

function main() {
  const cycle = lcgCycle();
  const period = Number(LCG_M);

  // 1. Proof of uniformity: the full period visits every state exactly once.
  assert(cycle.length === period && new Set(cycle).size === period, `the period visits ${new Set(cycle).size} distinct states`);
  assert(Math.min(...cycle) === 0 && Math.max(...cycle) === period - 1, "the cycle covers the whole 16-bit range");

  // 2. Exact bucket uniformity wherever the bucket count divides the range.
  for (const n of [2, 4, 8, 256]) {
    const tally = counts(cycle.map((value) => value % n), n);
    assert(spread(tally) === 0, `n=${n} buckets are equinumerous, spread ${spread(tally)}`);
    assert(tally[0] === period / n, `n=${n} buckets hold ${period / n} values each`);
  }

  // 3. The naive modulo reduction is biased off-divisor; rejection sampling repairs it exactly.
  const naiveThree = counts(cycle.map((value) => value % 3), 3);
  assert(spread(naiveThree) === 1, `x % 3 is biased by one, spread ${spread(naiveThree)}`);
  for (const n of [3, 5, 6, 10]) {
    const tidied = counts(uniformBelow(cycle, n), n);
    assert(spread(tidied) === 0, `rejection sampling makes n=${n} exact, spread ${spread(tidied)}`);
    assert(tidied.reduce((a, b) => a + b, 0) === Math.floor(period / n) * n, `n=${n} keeps the accepted slice`);
  }

  // 4. The certificate is seed-independent: any rotation of a full period is the same multiset.
  for (const seed of [1n, 12345n, 65535n]) {
    const rotated = lcgCycle(seed);
    assert(spread(counts(rotated.map((value) => value % 8), 8)) === 0 && new Set(rotated).size === period, `seed ${seed} certifies too`);
  }

  // 5. Statistical certificate for the 64-bit stream: chi-square with a p-value.
  const rng = splitmix64(945);
  const draws = Array.from({ length: 100000 }, () => unit(rng));
  const bins = 100;
  const tally = counts(draws.map((d) => Math.min(bins - 1, Math.floor(d * bins))), bins);
  const expected = draws.length / bins;
  const chiSquare = tally.reduce((a, count) => a + (count - expected) ** 2 / expected, 0);
  const pValue = chiSquarePValue(chiSquare, bins - 1);
  assert(Math.abs(chiSquare - (bins - 1)) < 4 * Math.sqrt(2 * (bins - 1)), `chi-square ${chiSquare} near its mean`);
  assert(pValue > 0.01, `chi-square p-value ${pValue} must not reject uniformity`);

  // 6. Kolmogorov-Smirnov against the uniform CDF.
  const ks = kolmogorovSmirnov(draws);
  assert(ks < 1.36 / Math.sqrt(draws.length), `KS distance ${ks} below the 5% critical value`);

  // 7. Runs around the median and the lag-1 correlation.
  const runs = runsAroundMedian(draws);
  const runSd = Math.sqrt(draws.length - 2) / 2;
  assert(Math.abs(runs - (draws.length / 2 + 1)) < 3 * runSd, `runs ${runs} within three sigma`);
  const rho = serialCorrelation(draws);
  assert(Math.abs(rho) < 3 / Math.sqrt(draws.length), `serial correlation ${rho} is negligible`);

  // 8. Boundary: one bucket receives everything, and a single sample certifies nothing but computes.
  assert(Math.max(...counts(cycle.map(() => 0), 1)) === period, "a single bucket collects the whole cycle");
  assert(uniformBelow(cycle, period).length === period && spread(counts(uniformBelow(cycle, period), period)) === 0, "one bin per state is exactly uniform");
  const single = [unit(splitmix64(7))];
  assert(single.length === 1 && kolmogorovSmirnov(single) >= 0.5 && kolmogorovSmirnov(single) <= 1, "a single sample yields a finite certificate");
  assert(Number.isFinite(chiSquarePValue(0, 1)) && chiSquarePValue(0, 1) > 0.9, "a small statistic cannot reject uniformity");
  assert(chiSquarePValue(0, 99) > chiSquarePValue(200, 99) && chiSquarePValue(200, 99) < 0.01, "a large statistic rejects uniformity");

  // 9. Determinism: the generator reproduces its stream and the cycle for a fixed seed.
  const stream = (seed) => { const local = splitmix64(seed); return Array.from({ length: 2000 }, () => unit(local)); };
  assert(JSON.stringify(stream(945)) === JSON.stringify(stream(945)), "the same seed reproduces the stream");
  assert(JSON.stringify(stream(945)) !== JSON.stringify(stream(946)), "another seed gives another stream");
  assert(JSON.stringify(lcgCycle(0n)) === JSON.stringify(cycle), "the LCG cycle is deterministic");

  // 10. Offsets into the stream differ, so the generator is not stuck on one value.
  assert(new Set(draws.slice(0, 5000)).size > 4900, "consecutive draws are distinct");
  assert(Math.abs(draws.reduce((a, b) => a + b, 0) / draws.length - 0.5) < 0.01, "the stream mean is one half");

  console.log(`certified-rng: period=${period} chi2=${chiSquare.toFixed(1)} p=${pValue.toFixed(3)} ks=${ks.toFixed(4)} runs=${runs} rho=${rho.toFixed(4)}`);
  console.log("PASS");
}
module.exports = { splitmix64, lcgCycle, uniformBelow, counts, spread, chiSquarePValue, kolmogorovSmirnov,
  runsAroundMedian, serialCorrelation, LCG_A, LCG_C, LCG_M };
if (require.main === module) main();
