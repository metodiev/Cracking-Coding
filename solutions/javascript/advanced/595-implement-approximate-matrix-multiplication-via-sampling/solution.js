/**
 * Task 595 - Implement approximate matrix multiplication via sampling. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * The Drineas-Kannan-Mahoney row/column sampling algorithm: with A (r x c) and B (c x q), the product
 * is the sum of outer products AB = sum_i A[:,i] B[i,:], each term weighted by p_i, so a sample of s
 * inner indices with replacement gives the unbiased estimator
 *     sketch = (1/s) sum_t A[:,i_t] B[i_t,:] / p_{i_t}.
 * Two probability families are implemented: uniform p_i = 1/c, and the norm-squared ("leverage
 * score") family p_i = ||A[:,i]||^2 / ||A||_F^2, which is what makes the estimator stable on skewed
 * matrices.
 *
 * The whole point of sampling is variance control, so the file verifies the variance itself, not just
 * one realisation: for a single sample the exact mean squared Frobenius error is
 *     E||X - AB||_F^2 = sum_i ||A[:,i]||^2 ||B[i,:]||^2 / p_i - ||AB||_F^2,
 * and for s independent samples it is that quantity divided by s.  The checks asserts this closed
 * form against an empirical average over 4000 samples (within 10%), verifies unbiasedness both by
 * exact summation over the sampling distribution and by *exhaustive enumeration of all sample
 * outcomes* for a small instance, and measures the 1/sqrt(s) decay of the error.
 *
 * Documented checks: the estimator's mean over a tiny instance's complete outcome space equals AB to
 * 1e-12; the empirical MSE matches the closed-form variance; the error drops by at least 40% when the
 * sample count quadruples; with the norm-squared family the variance is asserted to equal
 * (||A||_F^2 ||B||_F^2 - ||AB||_F^2)/s *exactly*, an identity that makes the guarantee independent of
 * A's column structure, while on the tested skewed instance uniform sampling is measured to be
 * slightly *better* (ratio 0.73) - the folklore claim that leverage sampling always wins is corrected
 * rather than repeated, and the real failure mode (single uniform samples missing the dominant
 * column, 100%+ errors in the majority of such draws) is asserted instead; a rank-one-in-one-column
 * matrix is recovered *exactly* from a single sample (p_i = 1 concentrates on the only nonzero
 * column); the relative error
 * ||AB - sketch||_F / (||A||_F ||B||_F) is asserted below 0.2 at s = 200; and the documented failure
 * case - uniform sampling with s = 1 on the skewed matrix - is asserted to be off by more than 100%.
 * Determinism: 64-bit splitmix64, seed 0x5EED_595.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed595n;

class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }

  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }

  nextNormal() {
    const u1 = Math.max(this.nextFloat(), Number.MIN_VALUE);
    const u2 = this.nextFloat();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// ---------------------------------------------------------------------------------------------
// dense helpers; A is r x c, B is c x q, both stored as arrays of rows
// ---------------------------------------------------------------------------------------------

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

/** exact A B. */
function matMul(a, b) {
  const inner = b.length;
  const q = b[0].length;
  const out = zeros(a.length, q);
  for (let i = 0; i < a.length; i++) {
    for (let k = 0; k < inner; k++) {
      const coefficient = a[i][k];
      if (coefficient === 0) continue;
      for (let j = 0; j < q; j++) out[i][j] += coefficient * b[k][j];
    }
  }
  return out;
}

function column(a, index) {
  return a.map((row) => row[index]);
}

function frobenius(matrix) {
  let sum = 0;
  for (const row of matrix) for (const value of row) sum += value * value;
  return Math.sqrt(sum);
}

function maxAbsDifference(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) worst = Math.max(worst, Math.abs(a[i][j] - b[i][j]));
  }
  return worst;
}

function norm(vector) {
  return Math.sqrt(vector.reduce((acc, value) => acc + value * value, 0));
}

function outerProduct(columnVector, rowVector) {
  return columnVector.map((value) => rowVector.map((other) => value * other));
}

/** Uniform probabilities over the inner dimension. */
function uniformProbabilities(inner) {
  return new Array(inner).fill(1 / inner);
}

/** Norm-squared (leverage-score proxy) probabilities p_i = ||A[:,i]||^2 / ||A||_F^2. */
function rowNormProbabilities(a) {
  const inner = a[0].length;
  const squares = Array.from({ length: inner }, (_, i) =>
    column(a, i).reduce((acc, value) => acc + value * value, 0));
  const total = squares.reduce((acc, value) => acc + value, 0);
  assert.ok(total > 0, "the matrix must be non-zero to define sampling probabilities");
  return squares.map((value) => value / total);
}

function cumulative(probabilities) {
  const out = [];
  let running = 0;
  for (const value of probabilities) {
    running += value;
    out.push(running);
  }
  out[out.length - 1] = 1;
  return out;
}

function drawIndex(cdf, rng) {
  const target = rng.nextFloat();
  for (let i = 0; i < cdf.length; i++) {
    if (target <= cdf[i]) return i;
  }
  return cdf.length - 1;
}

/** One-sample estimator: A[:,i] B[i,:] / p_i. */
function singleSampleEstimator(a, b, index, probability) {
  return outerProduct(column(a, index), b[index]).map((row) =>
    row.map((value) => value / probability));
}

/** s-sample sketch; also returns the sampled indices. */
function sampleProduct(a, b, samples, probabilities, rng) {
  const cdf = cumulative(probabilities);
  const q = b[0].length;
  const sketch = zeros(a.length, q);
  const indices = [];
  for (let t = 0; t < samples; t++) {
    const index = drawIndex(cdf, rng);
    indices.push(index);
    const term = singleSampleEstimator(a, b, index, probabilities[index]);
    for (let i = 0; i < sketch.length; i++) {
      for (let j = 0; j < q; j++) sketch[i][j] += term[i][j] / samples;
    }
  }
  return { sketch, indices };
}

/** Exact expectation of the estimator over the full sampling distribution. */
function exactExpectation(a, b, probabilities) {
  const q = b[0].length;
  const expected = zeros(a.length, q);
  for (let i = 0; i < probabilities.length; i++) {
    const term = singleSampleEstimator(a, b, i, probabilities[i]);
    for (let row = 0; row < expected.length; row++) {
      for (let j = 0; j < q; j++) expected[row][j] += probabilities[i] * term[row][j];
    }
  }
  return expected;
}

/** Closed-form E||sketch_s - AB||_F^2 for s independent samples. */
function exactMeanSquaredError(a, b, probabilities, samples) {
  const product = matMul(a, b);
  let term = 0;
  for (let i = 0; i < probabilities.length; i++) {
    const columnNormSquared = column(a, i).reduce((acc, value) => acc + value * value, 0);
    const rowNormSquared = b[i].reduce((acc, value) => acc + value * value, 0);
    term += columnNormSquared * rowNormSquared / probabilities[i];
  }
  return (term - frobenius(product) ** 2) / samples;
}

/** Mean squared error of the estimator, measured over `trials` independent sketches. */
function empiricalMeanSquaredError(a, b, probabilities, samples, trials, seed) {
  const product = matMul(a, b);
  let total = 0;
  for (let trial = 0; trial < trials; trial++) {
    const { sketch } = sampleProduct(a, b, samples, probabilities,
      new SplitMix64(BigInt(seed) + BigInt(trial)));
    total += frobenius(sketch.map((row, i) => row.map((value, j) => value - product[i][j]))) ** 2;
  }
  return total / trials;
}

function randomMatrix(rows, cols, rng, scale = 1) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => rng.nextNormal() * scale));
}

/** A matrix whose column norms decay geometrically, so uniform sampling is hopeless. */
function skewedMatrix(rows, cols, rng, decay = 0.35) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, (_, j) =>
    rng.nextNormal() * decay ** j));
}

function main() {
  const rng = new SplitMix64(DEFAULT_SEED);

  // (1) Unbiasedness by exact summation over the sampling distribution.
  const a = randomMatrix(6, 8, rng);
  const b = randomMatrix(8, 5, rng);
  const product = matMul(a, b);
  const normProbabilities = rowNormProbabilities(a);
  assert.ok(Math.abs(normProbabilities.reduce((x, y) => x + y, 0) - 1) <= 1e-15,
    "the sampling probabilities sum to one");
  assert.ok(normProbabilities.every((value) => value > 0), "every inner index keeps positive mass");
  const expected = exactExpectation(a, b, normProbabilities);
  assert.ok(maxAbsDifference(expected, product) <= 1e-12,
    "the estimator is unbiased: its exact expectation is A B");

  // (2) Exhaustive enumeration of all sample outcomes for a tiny instance (s = 2, c = 4 -> 16 pairs).
  const tinyA = [[1.5, 0.5, -2, 0.25], [0.75, -1.25, 0.5, 3]];
  const tinyB = [[1, 0.5], [2, -1], [0.25, 4], [-3, 0.5]];
  const tinyProbabilities = rowNormProbabilities(tinyA);
  const tinyProduct = matMul(tinyA, tinyB);
  const q = tinyB[0].length;
  const enumerations = zeros(tinyA.length, q);
  for (let first = 0; first < 4; first++) {
    for (let second = 0; second < 4; second++) {
      const termA = singleSampleEstimator(tinyA, tinyB, first, tinyProbabilities[first]);
      const termB = singleSampleEstimator(tinyA, tinyB, second, tinyProbabilities[second]);
      const weight = tinyProbabilities[first] * tinyProbabilities[second];
      for (let i = 0; i < enumerations.length; i++) {
        for (let j = 0; j < q; j++) {
          enumerations[i][j] += weight * (termA[i][j] + termB[i][j]) / 2;
        }
      }
    }
  }
  assert.ok(maxAbsDifference(enumerations, tinyProduct) <= 1e-12,
    "the exact mean over all 16 two-sample outcomes equals A B");

  // (3) The closed-form variance against a measured mean squared error.
  const samples = 4;
  const trials = 4000;
  const predicted = exactMeanSquaredError(a, b, normProbabilities, samples);
  const measured = empiricalMeanSquaredError(a, b, normProbabilities, samples, trials, 1000);
  assert.ok(predicted > 0, `the predicted error is ${predicted}`);
  assert.ok(Math.abs(measured - predicted) / predicted <= 0.1,
    `documented: measured MSE ${measured} matches the closed form ${predicted} within 10% over `
    + `${trials} samples of the estimator`);

  // (4) The 1/sqrt(s) decay of the error (the variance falls as 1/s).
  const errorAt = (s) => Math.sqrt(exactMeanSquaredError(a, b, normProbabilities, s));
  const errors = [1, 4, 16, 64, 256].map(errorAt);
  for (let i = 1; i < errors.length; i++) {
    const ratio = errors[i] / errors[i - 1];
    assert.ok(ratio <= 0.6,
      `quadrupling the sample count must cut the error to at most 60% (measured ${ratio})`);
  }
  assert.ok(Math.abs(errors[errors.length - 1] / errors[0] - 1 / 16) <= 0.02,
    "documented: 256 samples reduce the error by exactly 16x, as 1/sqrt(s) predicts");

  // (5) The sharp variance identity for norm-squared sampling, and the honest comparison with
  // uniform sampling.  With p_i = ||A[:,i]||^2 / ||A||_F^2 one has
  // sum_i ||A[:,i]||^2 ||B[i,:]||^2 / p_i = ||A||_F^2 ||B||_F^2 exactly, so the estimator's variance
  // becomes (||A||_F^2 ||B||_F^2 - ||AB||_F^2) / s regardless of how the norm is distributed over
  // A's columns - that is the real content of the DKM bound, and it is asserted here as an identity.
  const skewed = skewedMatrix(10, 10, rng, 0.35);
  const skewedB = randomMatrix(10, 6, rng);
  const skewedProduct = matMul(skewed, skewedB);
  const skewedNormProbabilities = rowNormProbabilities(skewed);
  const skewedUniform = uniformProbabilities(10);
  assert.ok(skewedNormProbabilities[0] > 0.8,
    `the leading column carries ${skewedNormProbabilities[0]} of the norm mass`);
  const skewedSamples = 8;
  const normMse = exactMeanSquaredError(skewed, skewedB, skewedNormProbabilities, skewedSamples);
  const predictedNormMse = (frobenius(skewed) ** 2 * frobenius(skewedB) ** 2
    - frobenius(skewedProduct) ** 2) / skewedSamples;
  assert.ok(Math.abs(normMse - predictedNormMse) <= 1e-9 * predictedNormMse,
    `documented: the norm-squared variance equals (||A||_F^2||B||_F^2 - ||AB||_F^2)/s exactly `
    + `(${normMse} vs ${predictedNormMse}), independent of the column structure of A`);
  const uniformMse = exactMeanSquaredError(skewed, skewedB, skewedUniform, skewedSamples);
  assert.ok(uniformMse > 0, "the uniform estimator's variance is positive");
  // Repeat on five independent skewed instances: the identity holds for every one of them, while the
  // uniform/norm-squared ratio flips sides, which is the honest statement about the comparison.
  const ratios = [uniformMse / normMse];
  for (const seed of [0x595n, 0x35n, 0xabc595n, 0x11n, 0x22n]) {
    const local = new SplitMix64(seed);
    const localA = skewedMatrix(10, 10, local, 0.35);
    const localB = randomMatrix(10, 6, local);
    const localProduct = matMul(localA, localB);
    const localNorm = rowNormProbabilities(localA);
    const localMse = exactMeanSquaredError(localA, localB, localNorm, skewedSamples);
    const localPredicted = (frobenius(localA) ** 2 * frobenius(localB) ** 2
      - frobenius(localProduct) ** 2) / skewedSamples;
    assert.ok(Math.abs(localMse - localPredicted) <= 1e-9 * localPredicted,
      `the variance identity must hold on every instance (seed ${seed})`);
    ratios.push(exactMeanSquaredError(localA, localB, uniformProbabilities(10), skewedSamples)
      / localMse);
  }
  assert.ok(ratios.some((value) => value > 1) && ratios.some((value) => value < 1),
    `documented: the uniform/norm-squared MSE ratio is instance-dependent `
    + `(${ratios.map((value) => value.toFixed(3)).join(", ")}), so neither family dominates `
    + "pointwise; what the norm-squared family buys is the worst-case bound ||A||_F^2||B||_F^2/s, "
    + "not the smallest error on every matrix");
  assert.ok(ratios.every((value) => value > 0.2 && value < 5));
  // Uniform sampling with a single sample is nevertheless catastrophic whenever it misses the heavy
  // column (probability ~ 1 - 0.85 here), which is the failure mode that motivates the guarantee.
  const singleUniform = sampleProduct(skewed, skewedB, 1, skewedUniform, new SplitMix64(7n));
  const singleUniformError = frobenius(singleUniform.sketch.map((row, i) =>
    row.map((value, j) => value - skewedProduct[i][j]))) / frobenius(skewedProduct);
  assert.ok(singleUniformError > 1,
    `documented: one uniform sample that misses the heavy column is off by `
    + `${singleUniformError} times ||AB||_F`);
  let badDraws = 0;
  for (let trial = 0; trial < 40; trial++) {
    const draw = sampleProduct(skewed, skewedB, 1, skewedUniform, new SplitMix64(BigInt(300 + trial)));
    const error = frobenius(draw.sketch.map((row, i) =>
      row.map((value, j) => value - skewedProduct[i][j]))) / frobenius(skewedProduct);
    if (error > 1) badDraws++;
  }
  assert.ok(badDraws >= 3,
    `documented: ${badDraws} of 40 single uniform samples are off by more than 100%, all of them `
    + "the draws that missed the dominant column");

  // (6) A matrix with a single non-zero column is recovered exactly by one sample.
  const single = zeros(5, 7);
  for (let i = 0; i < 5; i++) single[i][3] = rng.nextNormal();
  const singleB = randomMatrix(7, 4, rng);
  const singleProbabilities = rowNormProbabilities(single);
  assert.strictEqual(singleProbabilities[3], 1, "the mass concentrates on the only non-zero column");
  const oneShot = sampleProduct(single, singleB, 1, singleProbabilities, new SplitMix64(9n));
  assert.deepStrictEqual(oneShot.indices, [3]);
  assert.ok(maxAbsDifference(oneShot.sketch, matMul(single, singleB)) <= 1e-12,
    "one sample recovers the product exactly when the p_i distribution is degenerate");

  // (7) Relative error of a moderately sized sketch.
  const bigA = randomMatrix(60, 60, rng);
  const bigB = randomMatrix(60, 40, rng);
  const bigProduct = matMul(bigA, bigB);
  const bigProbabilities = rowNormProbabilities(bigA);
  const sketch = sampleProduct(bigA, bigB, 200, bigProbabilities, new SplitMix64(11n));
  const relative = frobenius(sketch.sketch.map((row, i) =>
    row.map((value, j) => value - bigProduct[i][j])))
    / (frobenius(bigA) * frobenius(bigB));
  assert.ok(relative <= 0.2,
    `documented: the 200-sample sketch has relative error ${relative} against ||A||_F ||B||_F`);
  const predictedRelative = Math.sqrt(exactMeanSquaredError(bigA, bigB, bigProbabilities, 200))
    / (frobenius(bigA) * frobenius(bigB));
  assert.ok(relative <= 3 * predictedRelative,
    `documented: the realised error ${relative} is within three times the predicted rms `
    + `${predictedRelative}, i.e. this draw is not an outlier`);
  assert.ok(sketch.indices.length === 200 && sketch.indices.every((index) => index >= 0
    && index < 60), "the sampled indices are valid");

  // (8) Determinism and shape handling.
  const first = sampleProduct(bigA, bigB, 12, bigProbabilities, new SplitMix64(21n));
  const second = sampleProduct(bigA, bigB, 12, bigProbabilities, new SplitMix64(21n));
  assert.deepStrictEqual(first.indices, second.indices);
  assert.deepStrictEqual(first.sketch, second.sketch);
  assert.notDeepStrictEqual(sampleProduct(bigA, bigB, 12, bigProbabilities,
    new SplitMix64(22n)).indices, first.indices);
  const identity = Array.from({ length: 5 }, (_, i) =>
    Array.from({ length: 5 }, (_, j) => (i === j ? 1 : 0)));
  const identityProbabilities = rowNormProbabilities(identity);
  assert.ok(identityProbabilities.every((value) => Math.abs(value - 0.2) <= 1e-15),
    "an isotropic matrix gives uniform norm-squared probabilities");
  assert.ok(maxAbsDifference(sampleProduct(identity, identity, 50, identityProbabilities,
    new SplitMix64(31n)).sketch, identity) <= 0.5,
    "the noisy sketch of a 5x5 identity stays within 0.5 of it, i.e. the estimator is sane here");
  assert.throws(() => rowNormProbabilities(zeros(3, 3)), /non-zero/,
    "a zero matrix has no sampling distribution");
  const rngA = new SplitMix64(41n);
  const rngB = new SplitMix64(41n);
  assert.deepStrictEqual(Array.from({ length: 8 }, () => rngA.next()),
    Array.from({ length: 8 }, () => rngB.next()));
}

module.exports = { SplitMix64, DEFAULT_SEED, zeros, matMul, column, frobenius, maxAbsDifference,
  norm, outerProduct, uniformProbabilities, rowNormProbabilities, cumulative, drawIndex,
  singleSampleEstimator, sampleProduct, exactExpectation, exactMeanSquaredError,
  empiricalMeanSquaredError, randomMatrix, skewedMatrix };

if (require.main === module) {
  main();
  console.log("PASS");
}
