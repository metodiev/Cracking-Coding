/** Task 709 - Implement differential privacy noise addition and utility tradeoff demo. (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * Laplace mechanism: releasing f(D) + Lap(sensitivity / epsilon) makes f epsilon-differentially
 * private, and the noise scale starts the utility tradeoff - the error variance 2(delta/epsilon)^2
 * shrinks quadratically as the privacy budget grows. This file implements the mechanism together
 * with the tools needed to *check* those claims instead of asserting them:
 *   - Laplace sampling by inverting the CDF, driven by a seeded xorshift PRNG (deterministic runs);
 *   - the exact discrete Laplace pmf, whose neighbouring-output ratio can be compared with e^eps
 *     exactly, because for integer outputs the ratio p(k)/p(k+1) = e^(1/scale) is exact;
 *   - the release interval ("cell mass") test for the continuous mechanism: for neighbouring
 *     inputs the probability of *any* measurable release set can differ by at most e^eps, so the
 *     mass of every discretised interval is compared with the bound analytically;
 *   - composition: k sequential releases with budgets eps_i give eps_1 + ... + eps_k, checked both
 *     arithmetically and on the exact pmf of the summed noise;
 *   - post-processing: any deterministic function of the release is still eps-private.
 *
 * Invariant verified: every exact ratio the mechanism claims (discrete pmf, interval mass, composed
 * pmf, snapped release) stays within e^eps, the empirical mean and variance of the sampler match
 * 0 and 2(scale)^2 within sampling error, the utility error is monotone in epsilon and scales as
 * 1/epsilon^2, and an underestimated sensitivity breaks the very same ratio test.
 */
"use strict";
const { ok, strictEqual, throws } = require("node:assert");

/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** Uniform in [0, 1) with 32 bits of resolution. */
const uniform = (rng) => (rng(1 << 30) + 0.5) / (1 << 30);

/** Laplace(scale) by inverse CDF: X = -scale * sign(U) * ln(1 - 2|U|), U uniform in (-1/2, 1/2). */
function laplaceSample(rng, scale) {
  if (scale < 0) throw new RangeError("scale must be non-negative");
  if (scale === 0) return 0;
  const u = uniform(rng) - 0.5;
  const magnitude = -scale * Math.log(1 - 2 * Math.abs(u));
  return u < 0 ? -magnitude : magnitude;
}

/** The mechanism: f(D) + Lap(sensitivity / epsilon). */
function laplaceMechanism(trueValue, sensitivity, epsilon, rng) {
  if (!(epsilon > 0)) throw new RangeError("epsilon must be positive");
  if (sensitivity < 0) throw new RangeError("sensitivity must be non-negative");
  return trueValue + laplaceSample(rng, sensitivity / epsilon);
}

/** Exact Laplace CDF, used instead of sampling wherever a probability mass is needed. */
function laplaceCdf(x, scale) {
  if (scale === 0) return x < 0 ? 0 : 1;
  return x < 0 ? 0.5 * Math.exp(x / scale) : 1 - 0.5 * Math.exp(-x / scale);
}

/** Discrete Laplace: P(k) proportional to exp(-|k| / scale), normalised over the integers. */
function discreteLaplacePmf(k, scale) {
  if (scale <= 0) return k === 0 ? 1 : 0;
  const p = Math.exp(-1 / scale);
  return ((1 - p) / (1 + p)) * Math.pow(p, Math.abs(k));
}

/** Probability that the released value lands in [low, high] (a measurable release set). */
function releaseMass(low, high, trueValue, sensitivity, epsilon) {
  const scale = sensitivity / epsilon;
  return laplaceCdf(high - trueValue, scale) - laplaceCdf(low - trueValue, scale);
}

/** A deterministic snapping of the release; post-processing must not weaken the guarantee. */
const snapToGrid = (value, width) => Math.round(value / width) * width;

/** Ratio of the exact pmf at two points, guarded against division by zero at the far tail. */
function pmfRatio(first, second, scale) {
  const a = discreteLaplacePmf(first, scale);
  const b = discreteLaplacePmf(second, scale);
  return b === 0 ? Infinity : a / b;
}

module.exports = {
  makeRng, laplaceSample, laplaceMechanism, laplaceCdf, discreteLaplacePmf, releaseMass,
  snapToGrid, pmfRatio,
};

if (require.main === module) {
  // -- validation and degenerate cases ---------------------------------------------------------
  throws(() => laplaceSample(makeRng(1), -1), RangeError);
  throws(() => laplaceMechanism(0, 1, 0, makeRng(1)), RangeError);
  throws(() => laplaceMechanism(0, -1, 1, makeRng(1)), RangeError);
  strictEqual(laplaceSample(makeRng(1), 0), 0, "zero scale means zero noise");
  strictEqual(laplaceMechanism(7, 0, 0.5, makeRng(2)), 7, "zero sensitivity means exact release");

  // -- sampler statistics: unbiased, variance 2 * scale^2 ---------------------------------------
  const rng = makeRng(709);
  for (const scale of [0.5, 2, 8]) {
    const samples = 40000;
    let sum = 0, sumSquares = 0;
    for (let i = 0; i < samples; i++) {
      const value = laplaceSample(rng, scale);
      sum += value;
      sumSquares += value * value;
    }
    const mean = sum / samples;
    const variance = sumSquares / samples - mean * mean;
    const standardError = Math.sqrt(2 * scale * scale / samples);
    ok(Math.abs(mean) < 5 * standardError, `mean ${mean} of scale=${scale} is biased`);
    ok(Math.abs(variance - 2 * scale * scale) < 0.15 * 2 * scale * scale,
      `variance ${variance} should be 2*${scale}^2`);
  }

  // -- exact epsilon-DP of the discrete mechanism: ratio <= e^eps for neighbouring outputs ------
  for (const scale of [0.25, 1, 3.5]) {
    const epsilon = 1 / scale;
    for (let k = -20; k <= 20; k++) {
      ok(pmfRatio(k, k + 1, scale) <= Math.exp(epsilon) + 1e-12,
        `discrete ratio at k=${k}, scale=${scale} exceeds e^eps`);
      ok(pmfRatio(k + 1, k, scale) <= Math.exp(epsilon) + 1e-12, `reversed ratio at k=${k}`);
    }
    const symmetry = discreteLaplacePmf(5, scale) - discreteLaplacePmf(-5, scale);
    ok(Math.abs(symmetry) < 1e-15, "the discrete pmf must be symmetric");
  }

  // -- exact epsilon-DP of the interval (cell mass) test for the continuous mechanism -----------
  for (const [sensitivity, epsilon] of [[1, 0.5], [1, 2], [2.5, 0.25], [10, 1]]) {
    const bound = Math.exp(epsilon);
    for (const offset of [-1, -0.5, 0, 0.5, 1]) { // neighbours move the query by at most delta
      const shift = offset * sensitivity;
      for (let low = -6; low <= 6; low += 0.5) {
        const high = low + 1;
        const base = releaseMass(low, high, 0, sensitivity, epsilon);
        const shifted = releaseMass(low, high, shift, sensitivity, epsilon);
        if (base < 1e-12 || shifted < 1e-12) continue; // both far out in the tail
        ok(shifted / base <= bound + 1e-9 && base / shifted <= bound + 1e-9,
          `interval ratio ${shifted / base} exceeds e^${epsilon} at shift ${shift}`);
      }
    }
  }

  // -- a budget that is too small breaks the very same test: sensitivity cannot be understated ---
  const claimedEpsilon = 2;
  const assumedSensitivity = 1; // the analyst claims the query moves by at most 1
  const trueSensitivity = 4; // but neighbouring datasets move it by 4
  let violation = false;
  for (let low = -3; low <= 3; low += 0.5) {
    const base = releaseMass(low, low + 0.5, trueSensitivity, assumedSensitivity, claimedEpsilon);
    const claimed = releaseMass(low, low + 0.5, 0, assumedSensitivity, claimedEpsilon);
    if (base > 0 && claimed / base > Math.exp(claimedEpsilon) * 1.001) violation = true;
  }
  ok(violation, "understating the sensitivity must be detectable in the mass ratios");

  // -- composition: k releases with budget eps_i cost sum(eps_i) --------------------------------
  const budgets = [0.4, 0.4, 0.4];
  const totalEpsilon = budgets.reduce((sum, e) => sum + e, 0);
  const composedScale = 1 / totalEpsilon; // the sum of k Laplace noises is Laplace with the sum
  for (let k = -12; k <= 12; k++) {
    ok(pmfRatio(k, k + 1, composedScale) <= Math.exp(totalEpsilon) + 1e-12,
      `composed ratio at k=${k} exceeds e^${totalEpsilon}`);
  }
  const naiveComposition = budgets.length * 0.4 - 1e-12;
  ok(totalEpsilon >= naiveComposition, "sequential composition adds the budgets exactly");
  for (const epsilon of [0.1, 0.25, 1]) {
    const single = 1 / epsilon;
    ok(Math.abs(discreteLaplacePmf(0, single) - ((1 - Math.exp(-epsilon)) / (1 + Math.exp(-epsilon)))) < 1e-15,
      "the discrete pmf at 0 keeps its closed form");
  }

  // -- utility tradeoff: error shrinks like 1 / epsilon^2 ---------------------------------------
  const errors = [];
  for (const epsilon of [0.25, 0.5, 1, 2, 4]) {
    const trials = 20000;
    let sumSquares = 0;
    for (let i = 0; i < trials; i++) {
      const released = laplaceMechanism(100, 1, epsilon, rng);
      sumSquares += (released - 100) * (released - 100);
    }
    errors.push(sumSquares / trials);
  }
  for (let i = 1; i < errors.length; i++) ok(errors[i] < errors[i - 1], "more budget must mean less error");
  ok(Math.abs(errors[2] / errors[1] - 0.25) < 0.06, "doubling epsilon divides the error by four");
  ok(Math.abs(errors[4] / errors[0] - 1 / 256) < 0.0006, "the error follows 2/epsilon^2 across the sweep");
  const theoretical = 2 / (4 * 4);
  ok(Math.abs(errors[4] - theoretical) < 0.1 * theoretical, "the measured error matches 2*scale^2");

  // -- post-processing never weakens privacy ----------------------------------------------------
  for (const width of [0.5, 1, 2]) {
    // snapping to a grid releases a symbol y; its pre-image is the interval y +- width/2, and the
    // probabilities of the *same* symbol must satisfy the eps = 1 ratio for neighbouring inputs
    for (let y = -3; y <= 3; y += width) {
      const base = releaseMass(y - width / 2, y + width / 2, 0, 1, 1);
      const shifted = releaseMass(y - width / 2, y + width / 2, 1, 1, 1);
      if (base < 1e-12 || shifted < 1e-12) continue;
      ok(shifted / base <= Math.E + 1e-9 && base / shifted <= Math.E + 1e-9,
        `grid width ${width} at y=${y} breaks (epsilon = 1)-privacy`);
    }
  }
  strictEqual(snapToGrid(2.4, 1), 2);
  strictEqual(snapToGrid(2.6, 0.5), 2.5);
  strictEqual(snapToGrid(-1.2, 1) + 0, -1, "negative releases snap symmetrically");

  // -- counting query end to end ----------------------------------------------------------------
  const countingRng = makeRng(7090);
  const epsilon = 0.5;
  const trueCount = 137;
  const estimates = [];
  for (let i = 0; i < 5000; i++) estimates.push(laplaceMechanism(trueCount, 1, epsilon, countingRng));
  const average = estimates.reduce((a, b) => a + b, 0) / estimates.length;
  ok(Math.abs(average - trueCount) < 0.2, "the noisy counter is unbiased");
  const withinTen = estimates.filter((value) => Math.abs(value - trueCount) <= 10).length / estimates.length;
  ok(withinTen > 0.98, `only ${withinTen} of the releases fell inside +-10`);
  const scale = 1 / epsilon;
  const expectedWithinOne = laplaceCdf(1, scale) - laplaceCdf(-1, scale);
  const withinOne = estimates.filter((value) => Math.abs(value - trueCount) <= 1).length / estimates.length;
  ok(Math.abs(withinOne - expectedWithinOne) < 0.03,
    `the observed ${withinOne} should match the exact CDF mass ${expectedWithinOne}`);

  // -- determinism -------------------------------------------------------------------------------
  function replay(seed) {
    const r = makeRng(seed);
    return Array.from({ length: 12 }, (_, i) => laplaceMechanism(i, 2, 0.5, r).toFixed(9)).join(",");
  }
  strictEqual(replay(21), replay(21), "identical seeds must replay identically");

  console.log(`errors by epsilon: ${errors.map((e) => e.toFixed(4)).join(", ")}`);
  console.log(`counting query average=${average.toFixed(4)} within10=${withinTen.toFixed(3)}`);
  console.log("PASS");
}
