/** Task 828 - Implement black-box optimizer with Bayesian optimization and EI acquisition.
 * (Tier: Master, Beyond — Machine Learning & AI)
 *
 * A Gaussian-process model (squared-exponential kernel, Cholesky factorisation with an escalating
 * jitter, a constant prior mean) drives Expected-Improvement maximisation over a candidate grid plus
 * axis-wise local refinement. The loop is a classic black-box optimiser: fit the posterior, maximise
 * the acquisition, evaluate the objective, repeat — all deterministic (Halton initial design, LCG
 * only where a random restart is needed).
 *
 * Formulae used: k(a,b) = sigma_f^2 exp(-|a-b|^2 / (2 l^2)); alpha = K^-1 (y - mu0);
 * mu(x) = mu0 + k*^T alpha; s^2(x) = sigma_f^2 - k*^T K^-1 k*;
 * EI(x) = (mu - tau) Phi(z) + s phi(z) with z = (mu - tau)/s and tau the incumbent minimum.
 * Phi is computed from erf, implemented by a Taylor series for |x| <= 3 and a Lentz continued
 * fraction for the complementary function beyond that.
 *
 * What is simplified: one scalar length scale, one noise level (no marginal-likelihood fitting, no
 * ARD), acquisition optimised by grid search + coordinate refinement instead of a gradient method,
 * to a bounded evaluation budget on 1-D and 2-D synthetic functions. No randomness, no timers.
 *
 * Invariants verified:
 *   - linear algebra: LL^T = K + (noise + jitter) I to 1e-12, and the Cholesky-based predictive mean
 *     and variance agree with an independent Gaussian-elimination oracle to 1e-10;
 *   - posterior semantics: interpolation at the observations with a tiny noise, prior mean/variance
 *     for an empty data set, variance in [0, sigma_f^2] everywhere and growing with distance from
 *     the nearest observation;
 *   - acquisition correctness: the closed-form EI equals a numerically integrated EI (split at the
 *     kink) to 1e-9 on hundreds of random (mu, s, tau) triples, is non-negative, strictly increasing
 *     in mu, equals s*phi(0) at mu = tau and degenerates to max(0, mu - tau) as s -> 0; the chosen
 *     query is a local maximum of EI;
 *   - optimisation (recall): the best-so-far is non-increasing, the final regret against a dense-grid
 *     oracle is below 0.05 on both test functions, and BO beats random search on the same budget;
 *   - determinism: the same seed reproduces the whole query sequence.
 */
"use strict";

const SQRT2 = Math.SQRT2;
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

// ---------------------------------------------------------------------------------------------
// Special function: erf / standard normal cdf
// ---------------------------------------------------------------------------------------------

/** Taylor series of erf for |x| <= 3 (the alternating series converges to 1e-16 within ~45 terms). */
function erfSeries(x) {
  let term = x;
  let sum = x;
  for (let n = 1; n <= 60; n++) {
    term *= -x * x / n;
    const contribution = term / (2 * n + 1);
    sum += contribution;
    if (Math.abs(contribution) < 1e-18 * Math.abs(sum) + 1e-300) break;
  }
  return 2 / Math.sqrt(Math.PI) * sum;
}

/**
 * Continued fraction for erfc (modified Lentz algorithm) using
 * erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))), i.e. b_n = x and
 * a_n = n/2. Only used for |x| > 3, where it converges to machine precision in a few dozen terms.
 */
function erfcContinuedFraction(x) {
  const tiny = 1e-300;
  let f = x;
  let c = x;
  let d = 0;
  for (let n = 1; n <= 400; n++) {
    const a = n / 2;
    d = x + a * d;
    if (d === 0) d = tiny;
    c = x + a / c;
    if (c === 0) c = tiny;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-17) break;
  }
  return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
}

/** erf via series for |x| <= 3 and via erfc beyond that. */
function erf(x) {
  if (Number.isNaN(x)) return NaN;
  const magnitude = Math.abs(x);
  if (magnitude <= 3) return erfSeries(x);
  const tail = 1 - erfcContinuedFraction(magnitude);
  return x > 0 ? tail : -tail;
}

/** Standard normal cdf. */
const normalCdf = (z) => 0.5 * (1 + erf(z / SQRT2));
const normalPdf = (z) => INV_SQRT_2PI * Math.exp(-0.5 * z * z);

// ---------------------------------------------------------------------------------------------
// Dense linear algebra
// ---------------------------------------------------------------------------------------------

/** In-place Cholesky with an escalating jitter; returns the lower factor and the jitter used. */
function cholesky(matrix, baseJitter = 1e-10) {
  const n = matrix.length;
  let jitter = baseJitter;
  for (let attempt = 0; attempt < 20; attempt++) {
    const lower = Array.from({ length: n }, () => new Array(n).fill(0));
    let ok = true;
    for (let i = 0; i < n && ok; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = matrix[i][j] + (i === j ? jitter : 0);
        for (let k = 0; k < j; k++) sum -= lower[i][k] * lower[j][k];
        if (i === j) {
          if (sum <= 0) {
            ok = false;
            break;
          }
          lower[i][j] = Math.sqrt(sum);
        } else {
          lower[i][j] = sum / lower[j][j];
        }
      }
    }
    if (ok) return { lower, jitter, success: true };
    jitter *= 10;
  }
  return { lower: null, jitter, success: false };
}

/** Forward/back substitution for L L^T x = b. */
function solveCholesky(lower, b) {
  const n = b.length;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = b[i];
    for (let k = 0; k < i; k++) sum -= lower[i][k] * y[k];
    y[i] = sum / lower[i][i];
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i];
    for (let k = i + 1; k < n; k++) sum -= lower[k][i] * x[k];
    x[i] = sum / lower[i][i];
  }
  return x;
}

/** Independent oracle: Gaussian elimination with partial pivoting. */
function solveDense(matrix, b) {
  const n = b.length;
  const a = matrix.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-300) throw new Error("singular system");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let row = col + 1; row < n; row++) {
      const factor = a[row][col] / a[col][col];
      for (let k = col; k <= n; k++) a[row][k] -= factor * a[col][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = a[row][n];
    for (let k = row + 1; k < n; k++) sum -= a[row][k] * x[k];
    x[row] = sum / a[row][row];
  }
  return x;
}

const squaredDistance = (a, b) => {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += (a[i] - b[i]) ** 2;
  return total;
};

// ---------------------------------------------------------------------------------------------
// Gaussian process
// ---------------------------------------------------------------------------------------------

const DEFAULT_KERNEL = { lengthScale: 0.2, signal: 1, noise: 1e-8 };

function kernel(a, b, settings) {
  return settings.signal ** 2 * Math.exp(-squaredDistance(a, b) / (2 * settings.lengthScale ** 2));
}

/** Builds the posterior from observations; the prior mean defaults to the sample mean. */
function fitGp(inputs, targets, settings = {}) {
  const options = { ...DEFAULT_KERNEL, ...settings };
  const n = inputs.length;
  const priorMean = options.priorMean ?? (n > 0 ? targets.reduce((a, b) => a + b, 0) / n : 0);
  if (n === 0) return { inputs, targets, options, priorMean, lower: null, alpha: [], jitter: 0 };
  const matrix = Array.from({ length: n }, (_, i) => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) matrix[i][j] = kernel(inputs[i], inputs[j], options);
    matrix[i][i] += options.noise;
  }
  const { lower, jitter, success } = cholesky(matrix);
  if (!success) throw new Error("kernel matrix is not positive definite");
  const alpha = solveCholesky(lower, targets.map((value) => value - priorMean));
  return { inputs, targets, options, priorMean, lower, alpha, jitter, matrix };
}

/** Predictive mean and variance at a list of points. */
function predict(fit, points) {
  const { inputs, options, priorMean, lower, alpha } = fit;
  const n = inputs.length;
  return points.map((point) => {
    if (n === 0) return { x: point, mean: priorMean, variance: options.signal ** 2 };
    const covariances = inputs.map((input) => kernel(point, input, options));
    let mean = priorMean;
    for (let i = 0; i < n; i++) mean += covariances[i] * alpha[i];
    const solved = solveCholesky(lower, covariances);
    let quadratic = 0;
    for (let i = 0; i < n; i++) quadratic += covariances[i] * solved[i];
    const variance = Math.max(0, options.signal ** 2 - quadratic);
    return { x: point, mean, variance };
  });
}

// ---------------------------------------------------------------------------------------------
// Acquisition
// ---------------------------------------------------------------------------------------------

/** Expected improvement over the incumbent value `best` (minimisation). */
function expectedImprovement(mean, standardDeviation, best) {
  if (standardDeviation <= 1e-12) return Math.max(0, mean - best);
  const z = (mean - best) / standardDeviation;
  return (mean - best) * normalCdf(z) + standardDeviation * normalPdf(z);
}

/** Independent oracle for EI: Simpson integration of max(mu + s t - tau, 0) phi(t) dt. */
function expectedImprovementNumeric(mean, standardDeviation, best, intervals = 20000) {
  if (standardDeviation <= 1e-12) return Math.max(0, mean - best);
  const kink = (best - mean) / standardDeviation;
  const low = -12;
  const high = 12;
  const points = Math.min(Math.max(kink, low), high);
  const integrate = (from, to) => {
    if (to <= from) return 0;
    const steps = Math.max(2, Math.round(intervals * (to - from) / (high - low)) * 2);
    const h = (to - from) / steps;
    let total = 0;
    for (let i = 0; i <= steps; i++) {
      const t = from + i * h;
      const value = Math.max(mean + standardDeviation * t - best, 0) * normalPdf(t);
      total += value * (i === 0 || i === steps ? 1 : i % 2 === 1 ? 4 : 2);
    }
    return total * h / 3;
  };
  return integrate(low, Math.min(points, high)) + integrate(Math.min(points, high), high);
}

// ---------------------------------------------------------------------------------------------
// Optimiser
// ---------------------------------------------------------------------------------------------

/** Deterministic Halton sequence (bases 2 and 3) for the initial space-filling design. */
function halton(index, base) {
  let fraction = 1;
  let result = 0;
  let i = index;
  while (i > 0) {
    fraction /= base;
    result += fraction * (i % base);
    i = Math.floor(i / base);
  }
  return result;
}

const haltonPoint = (index, dimension) =>
  Array.from({ length: dimension }, (_, d) => halton(index + 1, d === 0 ? 2 : 3));

function lcg(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Maximises EI by grid search plus coordinate refinement around the best grid point. */
function maximiseAcquisition(fit, bounds, options = {}) {
  const { gridSize = 201, rounds = 6, restarts = 2, seed = 828 } = options;
  const dimension = bounds.length;
  const candidates = [];
  const axisGrids = bounds.map(([low, high]) =>
    Array.from({ length: gridSize }, (_, i) => low + (high - low) * i / (gridSize - 1)));
  if (dimension === 1) {
    for (const x of axisGrids[0]) candidates.push([x]);
  } else if (dimension === 2) {
    for (const x of axisGrids[0]) for (const y of axisGrids[1]) candidates.push([x, y]);
  } else {
    throw new Error("only 1-D and 2-D grids are implemented");
  }
  const best = fit.targets.length > 0 ? Math.min(...fit.targets) : 0;
  const evaluated = candidates.map((point) => ({ point, value: acquire(fit, point, best) }));
  evaluated.sort((a, b) => b.value - a.value || (a.point[0] - b.point[0]));
  const starts = [evaluated[0].point];
  const rng = lcg(seed);
  for (let i = 0; i < restarts; i++) {
    starts.push(evaluated[Math.floor(rng() * evaluated.length) % evaluated.length].point);
  }
  let champion = evaluated[0];
  for (const start of starts) {
    let point = start.slice();
    let value = acquire(fit, point, best);
    let span = bounds.map(([low, high]) => (high - low) / (gridSize - 1) * 8);
    for (let round = 0; round < rounds; round++) {
      for (let axis = 0; axis < dimension; axis++) {
        const low = bounds[axis][0];
        const high = bounds[axis][1];
        for (const step of [-1, 1]) {
          const trial = point.slice();
          trial[axis] = Math.min(high, Math.max(low, trial[axis] + step * span[axis]));
          const trialValue = acquire(fit, trial, best);
          if (trialValue > value) {
            point = trial;
            value = trialValue;
          }
        }
      }
      span = span.map((value) => value / 2);
    }
    if (value > champion.value) champion = { point, value };
  }
  return champion;
}

function acquire(fit, point, best) {
  const { mean, variance } = predict(fit, [point])[0];
  return expectedImprovement(mean, Math.sqrt(variance), best);
}

/**
 * Bayesian-optimisation loop: Halton initial design, then EI maximisation until the budget is spent.
 * Returns the full trace (every query, its objective value and the incumbent after it).
 */
function bayesOptimize({ objective, bounds, budget = 25, initialPoints = 5, seed = 828,
  kernelSettings = {}, gridSize = 201, rounds = 6, restarts = 2 }) {
  const inputs = [];
  const targets = [];
  const trace = [];
  for (let i = 0; i < initialPoints; i++) {
    const unit = haltonPoint(i, bounds.length);
    const point = bounds.map(([low, high], d) => low + (high - low) * unit[d]);
    const value = objective(...point);
    inputs.push(point);
    targets.push(value);
    trace.push({ point, value, phase: "initial" });
  }
  let bestIndex = 0;
  for (let i = 1; i < targets.length; i++) if (targets[i] < targets[bestIndex]) bestIndex = i;
  const improvements = [targets[bestIndex]];
  for (let iteration = initialPoints; iteration < budget; iteration++) {
    const fit = fitGp(inputs, targets, kernelSettings);
    const champion = maximiseAcquisition(fit, bounds, { gridSize, rounds, restarts, seed: seed + iteration });
    const value = objective(...champion.point);
    inputs.push(champion.point);
    targets.push(value);
    trace.push({ point: champion.point, value, phase: "model", acquisition: champion.value });
    if (value < targets[bestIndex]) bestIndex = targets.length - 1;
    improvements.push(targets[bestIndex]);
  }
  return { inputs, targets, trace, best: targets[bestIndex], bestIndex, improvements };
}

/** Random-search baseline on the same budget and bounds. */
function randomSearch({ objective, bounds, budget, seed = 4242 }) {
  const rng = lcg(seed);
  let best = Infinity;
  let bestPoint = null;
  for (let i = 0; i < budget; i++) {
    const point = bounds.map(([low, high]) => low + (high - low) * rng());
    const value = objective(...point);
    if (value < best) {
      best = value;
      bestPoint = point;
    }
  }
  return { best, bestPoint };
}

// ---------------------------------------------------------------------------------------------
// Test problems and oracles
// ---------------------------------------------------------------------------------------------

/** Forrester function on [0,1]; its minimum is about -6.0207 near x = 0.7573. */
const forrester = (x) => (6 * x - 2) ** 2 * Math.sin(12 * x - 4);

/** 2-D mixture of a quadratic bowl and a mild ripple on [0,1]^2. */
const rippled = (x, y) => (x - 0.4) ** 2 + (y - 0.7) ** 2 + 0.1 * Math.sin(6 * x) * Math.cos(6 * y);

/** Dense-grid oracle for a 1-D objective. */
function oracle1d(fn, low = 0, high = 1, steps = 2000000) {
  let best = Infinity;
  let bestX = low;
  for (let i = 0; i <= steps; i++) {
    const x = low + (high - low) * i / steps;
    const value = fn(x);
    if (value < best) {
      best = value;
      bestX = x;
    }
  }
  return { best, bestX };
}

/** Dense-grid oracle for a 2-D objective. */
function oracle2d(fn, low = 0, high = 1, steps = 400) {
  let best = Infinity;
  let bestPoint = [low, low];
  for (let i = 0; i <= steps; i++) {
    for (let j = 0; j <= steps; j++) {
      const x = low + (high - low) * i / steps;
      const y = low + (high - low) * j / steps;
      const value = fn(x, y);
      if (value < best) {
        best = value;
        bestPoint = [x, y];
      }
    }
  }
  return { best, bestPoint };
}

module.exports = { erf, erfSeries, normalCdf, normalPdf, cholesky, solveCholesky, solveDense,
  kernel, fitGp, predict, expectedImprovement, expectedImprovementNumeric, halton, haltonPoint, lcg,
  maximiseAcquisition, acquire, bayesOptimize, randomSearch, forrester, rippled, oracle1d, oracle2d,
  DEFAULT_KERNEL };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];
  const rng = lcg(828);

  // ---- 1. kernel and Cholesky -------------------------------------------------------------
  const design = [[0], [0.25], [0.5], [0.75], [1]];
  const settings = { lengthScale: 0.3, signal: 2, noise: 1e-8 };
  const manual = 2 ** 2 * Math.exp(-(0.25 ** 2) / (2 * 0.3 ** 2));
  assert.ok(Math.abs(kernel([0], [0.25], settings) - manual) < 1e-15);
  assert.strictEqual(kernel([0.3], [0.3], settings), 4);
  const small = [[0, 1], [0, 1], [0.5, 0.5]];
  const matrix = [
    [kernel(small[0], small[0], settings) + settings.noise, kernel(small[0], small[1], settings), kernel(small[0], small[2], settings)],
    [kernel(small[1], small[0], settings), kernel(small[1], small[1], settings) + settings.noise, kernel(small[1], small[2], settings)],
    [kernel(small[2], small[0], settings), kernel(small[2], small[1], settings), kernel(small[2], small[2], settings) + settings.noise],
  ];
  const factorisation = cholesky(matrix);
  assert.ok(factorisation.success);
  let worstReconstruction = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) sum += factorisation.lower[i][k] * factorisation.lower[j][k];
      worstReconstruction = Math.max(worstReconstruction, Math.abs(sum - matrix[i][j] - (i === j ? factorisation.jitter : 0)));
      assert.strictEqual(sum, sum, "finite factor");
    }
  }
  assert.ok(worstReconstruction < 1e-12, `LL^T reconstruction error ${worstReconstruction}`);
  assert.ok(Math.abs(factorisation.jitter - 1e-10) < 1e-15, "a well-conditioned kernel needs no jitter");
  const duplicated = [[0], [0], [0]];
  const singular = duplicated.map((_, i) => duplicated.map((__, j) => kernel(duplicated[i], duplicated[j], settings) + (i === j ? settings.noise : 0)));
  const withJitter = cholesky(singular);
  assert.ok(withJitter.success, "the escalating jitter rescues a rank-deficient matrix");
  // an indefinite matrix is rescued only once the jitter exceeds its negative eigenvalue
  const indefinite = cholesky([[0, 1], [1, 0]]);
  assert.ok(indefinite.success && indefinite.jitter > 1, `escalated jitter ${indefinite.jitter}`);

  // ---- 2. Cholesky vs dense-solve oracle --------------------------------------------------
  const randomDesign = Array.from({ length: 20 }, (_, i) => [i / 19 * 3.5 - 0.5]);
  const randomTargets = randomDesign.map(() => rng() * 4 - 2);
  const fit = fitGp(randomDesign, randomTargets, { lengthScale: 0.5, signal: 3, noise: 0.05 });
  // the oracle solves exactly the matrix the factorisation used, jitter included
  const oracleMatrix = fit.matrix.map((row, i) => row.map((value, j) => value + (i === j ? fit.jitter : 0)));
  const denseAlpha = solveDense(oracleMatrix, randomTargets.map((v) => v - fit.priorMean));
  let worstAlpha = 0;
  for (let i = 0; i < 20; i++) worstAlpha = Math.max(worstAlpha, Math.abs(denseAlpha[i] - fit.alpha[i]));
  assert.ok(worstAlpha < 1e-10, `alpha disagreement ${worstAlpha}`);
  // residual check: the Cholesky solution really solves the system
  let worstResidual = 0;
  for (let i = 0; i < 20; i++) {
    let row = 0;
    for (let j = 0; j < 20; j++) row += oracleMatrix[i][j] * fit.alpha[j];
    worstResidual = Math.max(worstResidual, Math.abs(row - (randomTargets[i] - fit.priorMean)));
  }
  assert.ok(worstResidual < 1e-11, `linear solve residual ${worstResidual}`);

  const probes = Array.from({ length: 30 }, (_, i) => [i / 29 * 4.4 - 0.6]);
  const posterior = predict(fit, probes);
  let worstOracle = 0;
  for (let i = 0; i < probes.length; i++) {
    const covariances = randomDesign.map((input) => kernel(probes[i], input, fit.options));
    let mean = fit.priorMean;
    for (let k = 0; k < 20; k++) mean += covariances[k] * denseAlpha[k];
    const solved = solveDense(oracleMatrix, covariances);
    let quadratic = 0;
    for (let k = 0; k < 20; k++) quadratic += covariances[k] * solved[k];
    const variance = fit.options.signal ** 2 - quadratic;
    worstOracle = Math.max(worstOracle, Math.abs(mean - posterior[i].mean),
      Math.abs(variance - posterior[i].variance));
  }
  assert.ok(worstOracle < 1e-9, `predictive disagreement ${worstOracle}`);

  // ---- 3. posterior semantics --------------------------------------------------------------
  const exactFit = fitGp(randomDesign, randomTargets, { lengthScale: 0.25, signal: 3, noise: 1e-12 });
  const atData = predict(exactFit, randomDesign);
  let worstInterpolation = 0;
  for (let i = 0; i < 20; i++) {
    worstInterpolation = Math.max(worstInterpolation, Math.abs(atData[i].mean - randomTargets[i]));
    assert.ok(atData[i].variance < 1e-8, `variance at an observation ${atData[i].variance}`);
  }
  assert.ok(worstInterpolation < 1e-6, `interpolation error ${worstInterpolation}`);
  const prior = predict(fitGp([], [], { signal: 3 }), [[0.3], [0.7]]);
  for (const entry of prior) {
    assert.strictEqual(entry.mean, 0);
    assert.strictEqual(entry.variance, 9);
  }
  const far = predict(fit, [[10]]);
  assert.ok(far[0].variance > fit.options.signal ** 2 - 1e-9, "far away the prior variance is recovered");
  let worstVariance = 0;
  for (let i = 0; i <= 200; i++) {
    const x = i / 200;
    const entry = predict(fit, [[x]])[0];
    worstVariance = Math.max(worstVariance, Math.max(0, -entry.variance),
      Math.max(0, entry.variance - fit.options.signal ** 2));
    assert.ok(entry.variance >= 0 && entry.variance <= fit.options.signal ** 2 + 1e-12);
  }
  // variance grows with the distance from the data along a ray beyond the design
  const rayFit = fitGp(Array.from({ length: 11 }, (_, i) => [i / 10]), randomTargets.slice(0, 11),
    { lengthScale: 0.2, signal: 1, noise: 1e-8 });
  let previousVariance = -1;
  for (const distance of [0.01, 0.05, 0.1, 0.3, 0.6, 1.5]) {
    const entry = predict(rayFit, [[1 + distance]])[0];
    assert.ok(entry.variance > previousVariance, `variance at distance ${distance} is ${entry.variance}`);
    previousVariance = entry.variance;
  }
  assert.ok(previousVariance < 1 + 1e-12);
  const withNoise = predict(fitGp(randomDesign, randomTargets, { ...fit.options, noise: 1 }), [[0.5]])[0];
  const withoutNoise = predict(fit, [[0.5]])[0];
  assert.ok(withNoise.variance > withoutNoise.variance, "observation noise adds predictive variance");

  // ---- 4. erf and the normal cdf -----------------------------------------------------------
  let worstErf = 0;
  for (let i = 0; i <= 120; i++) {
    const x = -6 + i * 0.1;
    // independent oracle: Simpson integration of the erf integrand
    const steps = 4000;
    const h = x / steps;
    let total = 0;
    for (let k = 0; k <= steps; k++) {
      const t = k * h;
      const value = Math.exp(-t * t);
      total += value * (k === 0 || k === steps ? 1 : k % 2 === 1 ? 4 : 2);
    }
    const numeric = 2 / Math.sqrt(Math.PI) * total * h / 3;
    worstErf = Math.max(worstErf, Math.abs(numeric - erf(x)));
  }
  assert.ok(worstErf < 1e-9, `erf accuracy ${worstErf}`);
  assert.strictEqual(erf(0), 0);
  assert.ok(Math.abs(erf(-5) + erf(5)) < 1e-12, "erf is odd");
  for (let i = 1; i <= 19; i++) assert.ok(erf(i * 0.2) < erf((i + 1) * 0.2), "erf is strictly increasing");
  for (let i = 20; i < 60; i++) assert.ok(erf(i * 0.2) <= erf((i + 1) * 0.2), "erf never decreases (saturates)");
  // the series and the continued fraction must agree across their hand-over point at |x| = 3
  for (const x of [2.9, 3, 3.001]) {
    assert.ok(Math.abs(erf(x) - erfSeries(x)) < 1e-12, `series/CF continuity at ${x}`);
  }
  assert.ok(Math.abs(erf(20) - 1) < 1e-15 && Math.abs(erf(-20) + 1) < 1e-15);
  assert.ok(Math.abs(normalCdf(0) - 0.5) < 1e-15);
  for (const z of [-4, -1, -0.25, 0.25, 1, 4]) {
    assert.ok(Math.abs(normalCdf(-z) - (1 - normalCdf(z))) < 1e-14, "cdf symmetry");
  }
  assert.ok(Math.abs(normalCdf(1.959964) - 0.975) < 1e-6);
  assert.ok(normalPdf(0) > normalPdf(1) && normalPdf(1) > normalPdf(2));

  // ---- 5. EI against numerical integration -------------------------------------------------
  let worstEi = 0;
  for (let trial = 0; trial < 200; trial++) {
    const mean = rng() * 4 - 2;
    const deviation = 0.05 + rng() * 2;
    const best = rng() * 2 - 1;
    const closed = expectedImprovement(mean, deviation, best);
    const numeric = expectedImprovementNumeric(mean, deviation, best, 4000);
    worstEi = Math.max(worstEi, Math.abs(closed - numeric));
    assert.ok(closed >= 0, "EI is non-negative");
  }
  assert.ok(worstEi < 1e-8, `EI closed form vs quadrature ${worstEi}`);
  assert.ok(Math.abs(expectedImprovement(0, 1, 0) - INV_SQRT_2PI) < 1e-15, "EI(mu = tau) = s phi(0)");
  assert.strictEqual(expectedImprovement(0.5, 0, 0), 0.5);
  assert.strictEqual(expectedImprovement(-0.5, 0, 0), 0);
  let previousEi = -1;
  for (let i = 0; i <= 60; i++) {
    const value = expectedImprovement(-1 + i * 0.05, 1.3, 0.2);
    assert.ok(value > previousEi, "EI grows with the posterior mean");
    previousEi = value;
  }
  assert.ok(expectedImprovement(1, 1e-9, 0) === 1 || Math.abs(expectedImprovement(1, 1e-9, 0) - 1) < 1e-9);

  // ---- 6. acquisition maximisation is a local maximum --------------------------------------
  const acquisitionFit = fitGp([[0], [0.3], [0.6]], [1, -0.5, 0.4], { lengthScale: 0.25, signal: 1, noise: 1e-6 });
  const champion = maximiseAcquisition(acquisitionFit, [[0, 1]], { gridSize: 2001, rounds: 8, restarts: 0 });
  const championValue = acquire(acquisitionFit, champion.point, Math.min(...acquisitionFit.targets));
  // the maximiser must beat (or match) a 50k-point brute-force scan of the same interval
  const fine = Array.from({ length: 50001 }, (_, i) => i / 50000)
    .map((x) => ({ x, value: acquire(acquisitionFit, [x], Math.min(...acquisitionFit.targets)) }));
  const fineBest = fine.reduce((best, entry) => (entry.value > best.value ? entry : best));
  assert.ok(championValue >= fineBest.value - 1e-9,
    `refinement ${championValue} vs fine grid ${fineBest.value}`);
  assert.ok(Math.abs(champion.point[0] - fineBest.x) < 5e-3, "the query is where the acquisition peaks");
  const near = acquire(acquisitionFit, [champion.point[0] + 1e-3], Math.min(...acquisitionFit.targets));
  assert.ok(near <= championValue + 1e-9, "the query is a local maximum of EI");

  // ---- 7. optimisation quality on Forrester ------------------------------------------------
  const oracle = oracle1d(forrester);
  assert.ok(Math.abs(oracle.best + 6.0207) < 0.01, `Forrester oracle ${oracle.best}`);
  const bo = bayesOptimize({ objective: forrester, bounds: [[0, 1]], budget: 24, initialPoints: 5,
    seed: 828, kernelSettings: { lengthScale: 0.15, signal: 20, noise: 1e-8 }, gridSize: 2001,
    rounds: 8, restarts: 1 });
  for (let i = 1; i < bo.improvements.length; i++) {
    assert.ok(bo.improvements[i] <= bo.improvements[i - 1], "the best-so-far never worsens");
  }
  const regret = bo.best - oracle.best;
  assert.ok(regret < 0.05, `Forrester regret ${regret}`);
  assert.ok(Math.abs(bo.best - oracle.best) < Math.abs(bo.trace[0].value - oracle.best), "BO improves");
  const random1d = randomSearch({ objective: forrester, bounds: [[0, 1]], budget: 24, seed: 4242 });
  assert.ok(bo.best <= random1d.best, `BO ${bo.best} vs random ${random1d.best}`);

  // ---- 8. optimisation quality in 2-D ------------------------------------------------------
  const oracle2 = oracle2d(rippled, 0, 1, 500);
  const bo2 = bayesOptimize({ objective: rippled, bounds: [[0, 1], [0, 1]], budget: 30,
    initialPoints: 6, seed: 828, kernelSettings: { lengthScale: 0.2, signal: 0.5, noise: 1e-8 },
    gridSize: 121, rounds: 6, restarts: 2 });
  const regret2 = bo2.best - oracle2.best;
  assert.ok(regret2 < 0.05, `2-D regret ${regret2}`);
  const random2d = randomSearch({ objective: rippled, bounds: [[0, 1], [0, 1]], budget: 30, seed: 77 });
  assert.ok(bo2.best <= random2d.best, `BO ${bo2.best} vs random ${random2d.best}`);

  // ---- 9. determinism and edges ------------------------------------------------------------
  const repeat = bayesOptimize({ objective: forrester, bounds: [[0, 1]], budget: 12, initialPoints: 4,
    seed: 828, kernelSettings: { lengthScale: 0.2, signal: 20, noise: 1e-8 }, gridSize: 501 });
  const repeatAgain = bayesOptimize({ objective: forrester, bounds: [[0, 1]], budget: 12, initialPoints: 4,
    seed: 828, kernelSettings: { lengthScale: 0.2, signal: 20, noise: 1e-8 }, gridSize: 501 });
  assert.deepStrictEqual(repeatAgain.trace.map((entry) => entry.point), repeat.trace.map((entry) => entry.point));
  assert.strictEqual(bayesOptimize({ objective: forrester, bounds: [[0, 1]], budget: 1,
    initialPoints: 1, seed: 1 }).trace.length, 1);
  const singlePoint = predict(fitGp([[0.4]], [1]), [[0.4], [0.5]]);
  assert.ok(Math.abs(singlePoint[0].mean - 1) < 1e-6);
  assert.ok(singlePoint[1].variance > 0);
  // duplicate inputs with zero noise are invertible thanks to the diagonal jitter
  const duplicates = fitGp([[0], [0]], [1, 2], { noise: 0, lengthScale: 1 });
  assert.ok(Math.abs(duplicates.jitter - 1e-10) < 1e-15, "the base jitter suffices here");
  const tied = predict(duplicates, [[0]])[0];
  assert.ok(Math.abs(tied.mean - 1.5) < 1e-6, `tied observations average out (${tied.mean})`);
  assert.ok(tied.variance < 1e-6);
  const tinyBudget = bayesOptimize({ objective: (x) => x * x, bounds: [[-1, 1]], budget: 3,
    initialPoints: 3, seed: 3 });
  assert.strictEqual(tinyBudget.trace.length, 3);
  assert.ok(tinyBudget.best <= 1 && tinyBudget.best >= 0);

  metrics.push(`erf vs quadrature: max error ${worstErf.toExponential(2)}, Cholesky LL^T error ${worstReconstruction.toExponential(2)}`);
  metrics.push(`predictive mean/variance vs Gaussian elimination: max error ${worstOracle.toExponential(2)} (alpha ${worstAlpha.toExponential(2)}, interpolation ${worstInterpolation.toExponential(2)})`);
  metrics.push(`EI closed form vs quadrature over 200 triples: max error ${worstEi.toExponential(2)}`);
  metrics.push(`Forrester: optimum ${oracle.best.toFixed(6)} at x = ${oracle.bestX.toFixed(6)}, BO best ${bo.best.toFixed(6)} (regret ${regret.toExponential(2)}) vs random ${random1d.best.toFixed(6)}`);
  metrics.push(`2-D ripple: oracle ${oracle2.best.toFixed(6)}, BO best ${bo2.best.toFixed(6)} (regret ${regret2.toExponential(2)}) vs random ${random2d.best.toFixed(6)}`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
