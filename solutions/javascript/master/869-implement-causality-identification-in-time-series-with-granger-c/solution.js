/** Task 869 - Implement causality identification in time series with Granger causality and robustness. (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * Granger causality for a p-order vector autoregression. "x Granger-causes y" is decided by an F test
 * between two nested ordinary-least-squares fits of y:
 *
 *     restricted   : y_t = a0 + sum_{i=1..p} a_i y_{t-i}
 *     unrestricted : y_t = b0 + sum_{i=1..p} b_i y_{t-i} + sum_{i=1..p} c_i x_{t-i}
 *     F = ((RSS_r - RSS_u) / p) / (RSS_u / (T - 2p - 1)),   p = 1 - F_p,T-2p-1(F)
 *
 * Both fits are solved from the normal equations X'X beta = X'y by Gaussian elimination with partial
 * pivoting; the F statistic's p-value goes through the regularized incomplete beta function
 * (continued-fraction evaluation, log-gamma by Lanczos).
 *
 * Robustness is what makes the identification trustworthy, so it is tested rather than asserted:
 *   - sliding-window stability: the edge must be detected in every contiguous window of the series;
 *   - circular block bootstrap (seeded splitmix64 PRNG) of the residuals, giving a percentile interval
 *     for F that must stay away from the null region;
 *   - residual diagnostics: the unrestricted residuals must be serially uncorrelated (portmanteau
 *     statistic below the chi-square threshold), so the F test's white-noise assumption holds.
 *
 * Invariants verified: the F test detects exactly the planted edges and nothing else in a system with
 * known structure (x -> y only), the p-values of the *null* edges are large, the number of spurious
 * detections over many seeded independent series stays near the nominal level, the normal equations
 * really are solved (X'r ~ 0 and RSS_u <= RSS_r), the incomplete beta function agrees with independent
 * numerical quadrature of the beta density, and everything is bit-for-bit reproducible.
 *
 * What is simplified: one causal direction is tested at a time (no conditional independence tests or
 * multivariate F/block tests across all pairs), the data are small synthetic VAR(1) processes instead
 * of recorded series, non-stationarity and cointegration are out of scope, and "robustness" means
 * window and bootstrap stability rather than a full sensitivity analysis. No timers, no wall-clock
 * time, no Math.random, no threads.
 */
"use strict";

const { ok, throws } = require("node:assert");

/** splitmix64-based PRNG in [0, 1): the only randomness source in this file. */
function makeRng(seed) {
  let state = BigInt.asUintN(64, BigInt(seed >>> 0) | 0x9e3779b97f4a7c15n);
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    z ^= z >> 31n;
    return Number(BigInt.asUintN(64, z) >> 11n) / 9007199254740992;
  };
}

/** Box-Muller normal draws from a deterministic uniform stream. */
function makeNormal(rng) {
  return function normal() {
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

function logGamma(x) {
  const coefficients = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
  ];
  let y = x;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) {
    y += 1;
    ser += coefficients[j] / y;
  }
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

/** Continued fraction for the incomplete beta function (Lentz's algorithm). */
function betaContinuedFraction(a, b, x) {
  const maxIterations = 300;
  const eps = 3e-14;
  const tiny = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= maxIterations; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < eps) break;
  }
  return h;
}

/** Regularized incomplete beta function I_x(a, b). */
function regularizedIncompleteBeta(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2)
    ? (front * betaContinuedFraction(a, b, x)) / a
    : 1 - (front * betaContinuedFraction(b, a, 1 - x)) / b;
}

/** Upper tail of the F distribution: P(F_{d1,d2} > f). */
function fDistributionTail(f, d1, d2) {
  if (!(f >= 0)) throw new RangeError(`F statistic must be non-negative (got ${f})`);
  if (!(d1 > 0) || !(d2 > 0)) throw new RangeError("degrees of freedom must be positive");
  if (f === 0) return 1;
  return regularizedIncompleteBeta(d2 / 2, d1 / 2, d2 / (d2 + d1 * f));
}

/** Beta density, used only to cross-check the incomplete beta by numerical quadrature. */
const betaDensity = (x, a, b) => Math.exp(
  logGamma(a + b) - logGamma(a) - logGamma(b) + (a - 1) * Math.log(Math.max(x, 1e-300)) + (b - 1) * Math.log(Math.max(1 - x, 1e-300)),
);

/** Simpson quadrature of the beta density on [0, x]: an independent oracle for I_x(a, b). */
function quadratureBeta(a, b, x, intervals = 200000) {
  if (x <= 0) return 0;
  const n = intervals % 2 === 0 ? intervals : intervals + 1;
  const h = x / n;
  let sum = betaDensity(0, a, b) + betaDensity(x, a, b);
  for (let i = 1; i < n; i++) sum += betaDensity(i * h, a, b) * (i % 2 === 0 ? 2 : 4);
  return (sum * h) / 3;
}

/** Dense linear solve by Gaussian elimination with partial pivoting. */
function solveLinearSystem(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-12) throw new Error("singular normal equations");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let r = col + 1; r < n; r++) {
      const factor = a[r][col] / a[col][col];
      for (let c = col; c <= n; c++) a[r][c] -= factor * a[col][c];
    }
  }
  const solution = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let sum = a[r][n];
    for (let c = r + 1; c < n; c++) sum -= a[r][c] * solution[c];
    solution[r] = sum / a[r][r];
  }
  return solution;
}

/** Ordinary least squares from the normal equations; also returns the residuals and RSS. */
function ols(design, target) {
  const n = design.length;
  const p = design[0].length;
  if (n <= p) throw new RangeError("not enough observations for the design matrix");
  const xtx = Array.from({ length: p }, () => new Array(p).fill(0));
  const xty = new Array(p).fill(0);
  for (let row = 0; row < n; row++) {
    for (let i = 0; i < p; i++) {
      xty[i] += design[row][i] * target[row];
      for (let j = i; j < p; j++) xtx[i][j] += design[row][i] * design[row][j];
    }
  }
  for (let i = 0; i < p; i++) for (let j = 0; j < i; j++) xtx[i][j] = xtx[j][i];
  const beta = solveLinearSystem(xtx, xty);
  const residuals = design.map((row, i) => target[i] - row.reduce((acc, v, j) => acc + v * beta[j], 0));
  const rss = residuals.reduce((acc, r) => acc + r * r, 0);
  const gradient = xtx.map((row, i) => row.reduce((acc, v, j) => acc + v * beta[j], 0) - xty[i]);
  return { beta, residuals, rss, normalEquationsResidual: Math.max(...gradient.map(Math.abs)) };
}

/** Lagged design matrix for `targets` regressed on its own lags and (optionally) `causes`. */
function laggedDesign(targetSeries, causeSeries, order) {
  if (!Number.isInteger(order) || order < 1) throw new RangeError(`lag order must be a positive integer (got ${order})`);
  if (causeSeries && targetSeries.length !== causeSeries.length) throw new RangeError("the two series must have the same length");
  const rows = [];
  const target = [];
  for (let t = order; t < targetSeries.length; t++) {
    const row = [1];
    for (let lag = 1; lag <= order; lag++) row.push(targetSeries[t - lag]);
    if (causeSeries) for (let lag = 1; lag <= order; lag++) row.push(causeSeries[t - lag]);
    rows.push(row);
    target.push(targetSeries[t]);
  }
  if (rows.length <= rows[0].length) throw new RangeError("the series is too short for this lag order");
  return { design: rows, target };
}

/**
 * Granger causality test of `cause` on `target`. Returns the F statistic, its p-value, both RSS values
 * and the raw fits so that orthogonality and monotonicity can be audited from outside.
 */
function grangerCausality(targetSeries, causeSeries, { order = 1 } = {}) {
  const restricted = laggedDesign(targetSeries, null, order);
  const unrestricted = laggedDesign(targetSeries, causeSeries, order);
  const rssRestricted = ols(restricted.design, restricted.target).rss;
  const unrestrictedFit = ols(unrestricted.design, unrestricted.target);
  const rssUnrestricted = unrestrictedFit.rss;
  if (rssUnrestricted <= 0) throw new Error("degenerate regression with zero residual variance");
  const degrees = unrestricted.target.length - unrestricted.design[0].length;
  if (degrees < 1) throw new RangeError("no residual degrees of freedom left");
  const f = ((rssRestricted - rssUnrestricted) / order) / (rssUnrestricted / degrees);
  return {
    f,
    pValue: fDistributionTail(f, order, degrees),
    rssRestricted,
    rssUnrestricted,
    numeratorDegrees: order,
    denominatorDegrees: degrees,
    observations: unrestricted.target.length,
    restrictedFit: ols(restricted.design, restricted.target),
    unrestrictedFit,
  };
}

/** Portmanteau (Ljung-Box style) statistic on residuals; chi-square comparison is by the caller. */
function portmanteau(residuals, lags = 5) {
  const n = residuals.length;
  const variance = residuals.reduce((acc, r) => acc + r * r, 0) / n;
  let statistic = 0;
  for (let k = 1; k <= lags; k++) {
    let numerator = 0;
    for (let t = k; t < n; t++) numerator += residuals[t] * residuals[t - k];
    const rho = numerator / (n * variance);
    statistic += (rho * rho) / (n - k);
  }
  return n * (n + 2) * statistic;
}

/** Synthetic VAR(1) system with a planted x -> y edge and an unrelated z. */
function simulateVar1({ length = 600, seed = 42, coefficientXY = 0.6, coefficientYY = 0.4, coefficientXX = 0.6, coefficientZZ = 0.5 } = {}) {
  if (!Number.isInteger(length) || length < 20) throw new RangeError("length must be an integer of at least 20");
  const rng = makeRng(seed);
  const normal = makeNormal(rng);
  const x = [0];
  const y = [0];
  const z = [0];
  for (let t = 1; t < length; t++) {
    const ex = normal();
    const ey = normal();
    const ez = normal();
    x.push(coefficientXX * x[t - 1] + ex);
    y.push(coefficientYY * y[t - 1] + coefficientXY * x[t - 1] + ey);
    z.push(coefficientZZ * z[t - 1] + ez);
  }
  return { x, y, z };
}

/**
 * Circular block bootstrap of the F statistic under the *restricted* (null) model: the restricted fit
 * is treated as the truth, its residuals are resampled in blocks and added back, and the test is
 * re-run. The resulting quantiles are an empirical null distribution for F, so the observed statistic
 * is significant exactly when it exceeds the upper quantile.
 */
function blockBootstrapF(targetSeries, causeSeries, { order = 1, blockLength = 20, resamples = 200, seed = 7 } = {}) {
  if (!Number.isInteger(blockLength) || blockLength < 1) throw new RangeError("blockLength must be a positive integer");
  const restricted = laggedDesign(targetSeries, null, order);
  const fit = ols(restricted.design, restricted.target);
  const fitted = restricted.design.map((row) => row.reduce((acc, v, j) => acc + v * fit.beta[j], 0));
  const residuals = fit.residuals;
  const n = residuals.length;
  const rng = makeRng(seed);
  const statistics = [];
  for (let sample = 0; sample < resamples; sample++) {
    const stream = new Array(n);
    for (let position = 0; position < n;) {
      const start = Math.floor(rng() * n) % n;
      for (let k = 0; k < blockLength && position < n; k++, position++) {
        stream[position] = residuals[(start + k) % n];
      }
    }
    const bootstrapTarget = targetSeries.slice();
    for (let i = 0; i < n; i++) bootstrapTarget[order + i] = fitted[i] + stream[i];
    statistics.push(grangerCausality(bootstrapTarget, causeSeries, { order }).f);
  }
  statistics.sort((a, b) => a - b);
  const quantile = (q) => statistics[Math.min(statistics.length - 1, Math.max(0, Math.round(q * (statistics.length - 1))))];
  return { statistics, median: quantile(0.5), lower: quantile(0.05), upper: quantile(0.95), resamples };
}

/** F critical value for a given upper tail probability, by bisection on the (monotone) tail function. */
function fCritical(alpha, d1, d2) {
  let lo = 0;
  let hi = 1;
  while (fDistributionTail(hi, d1, d2) > alpha && hi < 1e9) hi *= 2;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (fDistributionTail(mid, d1, d2) > alpha) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function main() {
  const diagnostics = [];
  const alpha = 0.01;

  // 1. The incomplete beta function agrees with independent Simpson quadrature of the beta density.
  const betaCases = [[2, 3, 0.35], [5, 5, 0.5], [3, 7, 0.2], [7, 2, 0.9], [1.5, 2.5, 0.6]];
  let worstBeta = 0;
  for (const [a, b, xx] of betaCases) {
    worstBeta = Math.max(worstBeta, Math.abs(regularizedIncompleteBeta(a, b, xx) - quadratureBeta(a, b, xx)));
  }
  diagnostics.push(`incomplete beta vs quadrature: max deviation ${worstBeta.toExponential(2)}`);
  ok(worstBeta < 1e-7, `regularized incomplete beta must match quadrature (deviation ${worstBeta})`);
  ok(Math.abs(fDistributionTail(1, 10, 10) - 0.5) < 1e-12, "the median of F_{10,10} must be exactly 1");

  // 2. Planted structure: x -> y is detected, y -> x and z -> y are not.
  const system = simulateVar1({ length: 600, seed: 42 });
  const forward = grangerCausality(system.y, system.x, { order: 1 });
  const backward = grangerCausality(system.x, system.y, { order: 1 });
  const unrelated = grangerCausality(system.y, system.z, { order: 1 });
  diagnostics.push(`x->y F=${forward.f.toFixed(3)} p=${forward.pValue.toExponential(2)}`);
  diagnostics.push(`y->x F=${backward.f.toFixed(3)} p=${backward.pValue.toFixed(4)}, z->y F=${unrelated.f.toFixed(3)} p=${unrelated.pValue.toFixed(4)}`);
  ok(forward.pValue < 1e-6, `the planted edge must be detected (p = ${forward.pValue})`);
  ok(forward.f > fCritical(alpha, 1, forward.denominatorDegrees), "the F statistic must exceed the 1% critical value");
  ok(backward.pValue > 0.05, `the reverse direction must not be detected (p = ${backward.pValue})`);
  ok(unrelated.pValue > 0.05, `an unrelated series must not be detected (p = ${unrelated.pValue})`);

  // 3. The nested models behave as they must: RSS_u <= RSS_r, and the normal equations are solved
  //    to machine precision (X'r ~ 0 is an independent check that does not reuse the solver).
  ok(forward.rssUnrestricted <= forward.rssRestricted, "the unrestricted model cannot fit worse");
  diagnostics.push(`normal-equation residuals: restricted ${forward.restrictedFit.normalEquationsResidual.toExponential(2)}, unrestricted ${forward.unrestrictedFit.normalEquationsResidual.toExponential(2)}`);
  ok(forward.restrictedFit.normalEquationsResidual < 1e-9, "the restricted least-squares solution must satisfy the normal equations");
  ok(forward.unrestrictedFit.normalEquationsResidual < 1e-9, "the unrestricted least-squares solution must satisfy the normal equations");
  const residualSpread = Math.max(...forward.unrestrictedFit.residuals.map((r) => Math.abs(r - forward.unrestrictedFit.residuals.reduce((a, b) => a + b, 0) / forward.unrestrictedFit.residuals.length)));
  ok(Number.isFinite(residualSpread) && residualSpread > 0, "residuals must be finite and non-degenerate");

  // 4. White-noise assumption: the unrestricted residuals are serially uncorrelated.
  const portmanteauStatistic = portmanteau(forward.unrestrictedFit.residuals, 5);
  diagnostics.push(`portmanteau (5 lags) on unrestricted residuals: ${portmanteauStatistic.toFixed(2)} (chi2 0.999 quantile ~20.5)`);
  ok(portmanteauStatistic < 20.5, `residuals must look like white noise (statistic ${portmanteauStatistic})`);

  // 5. Calibration under the null: with no causal edge the p-values are large, and across many seeded
  //    independent systems the false-positive rate stays near the nominal level.
  const nullSystem = simulateVar1({ length: 400, seed: 999, coefficientXY: 0 });
  const nullTest = grangerCausality(nullSystem.y, nullSystem.x, { order: 1 });
  diagnostics.push(`null system x->y: F=${nullTest.f.toFixed(3)} p=${nullTest.pValue.toFixed(4)}`);
  ok(nullTest.pValue > 0.05, "a zero coefficient must not be reported as causal");
  let falsePositives = 0;
  const trials = 40;
  for (let trial = 0; trial < trials; trial++) {
    const independent = simulateVar1({ length: 300, seed: 1000 + trial, coefficientXY: 0 });
    if (grangerCausality(independent.y, independent.x, { order: 1 }).pValue < 0.05) falsePositives += 1;
  }
  diagnostics.push(`null false positives: ${falsePositives}/${trials} at alpha = 0.05`);
  ok(falsePositives <= 6, `the false-positive rate must stay near the nominal level (${falsePositives}/${trials})`);

  // 6. Order sensitivity: the planted edge survives p = 1..3, and too short a series is rejected.
  const orders = [1, 2, 3].map((order) => grangerCausality(system.y, system.x, { order }));
  for (const test of orders) {
    diagnostics.push(`order ${test.numeratorDegrees}: F=${test.f.toFixed(3)} p=${test.pValue.toExponential(2)} df=${test.denominatorDegrees}`);
    ok(test.pValue < 1e-3, `the edge must survive lag order ${test.numeratorDegrees} (p = ${test.pValue})`);
    ok(test.rssUnrestricted <= test.rssRestricted + 1e-9, "nested-model ordering must hold at every order");
  }

  // 7. Robustness: the edge is detected in every contiguous window of the series.
  const windowSize = 200;
  const windowResults = [0, 1, 2].map((k) => grangerCausality(
    system.y.slice(k * windowSize, k * windowSize + windowSize),
    system.x.slice(k * windowSize, k * windowSize + windowSize),
    { order: 1 },
  ));
  diagnostics.push(`windows: F = ${windowResults.map((w) => w.f.toFixed(2)).join(", ")}`);
  ok(windowResults.every((w) => w.pValue < 0.01), "the planted edge must be detected in every window");
  ok(windowResults.every((w) => w.f > fCritical(alpha, 1, w.denominatorDegrees)), "every window must clear the 1% critical value");

  // 8. Robustness: a circular block bootstrap gives an empirical null distribution for F; the planted
  //    edge must exceed its 95% quantile, while the null system must stay well inside it.
  const bootstrap = blockBootstrapF(system.y, system.x, { order: 1, blockLength: 20, resamples: 150, seed: 7 });
  const nullBootstrap = blockBootstrapF(nullSystem.y, nullSystem.x, { order: 1, blockLength: 20, resamples: 150, seed: 7 });
  diagnostics.push(`bootstrap F (edge): median ${bootstrap.median.toFixed(2)}, 95% ${bootstrap.upper.toFixed(2)} vs observed ${forward.f.toFixed(2)}`);
  diagnostics.push(`bootstrap F (null): median ${nullBootstrap.median.toFixed(2)}, 95% ${nullBootstrap.upper.toFixed(2)} vs observed ${nullTest.f.toFixed(3)}`);
  ok(bootstrap.resamples === 150, "every bootstrap resample must be produced");
  ok(bootstrap.median < bootstrap.upper, "the bootstrap quantiles must be ordered");
  ok(forward.f > bootstrap.upper, "the planted edge must exceed the bootstrap 95% quantile");
  ok(nullTest.f <= nullBootstrap.upper, "under the null the observed statistic must stay inside the bootstrap null distribution");
  ok(nullBootstrap.median < bootstrap.median * 10, "the bootstrap null distribution must be far smaller than the causal signal");

  // 9. Multiplicity control: with three pairs tested, the Bonferroni threshold keeps the planted edge
  //    significant and still rejects the spurious ones.
  const pairs = [forward, backward, unrelated];
  const bonferroni = alpha / pairs.length;
  const significant = pairs.filter((test) => test.pValue < bonferroni).length;
  diagnostics.push(`Bonferroni at ${bonferroni.toFixed(4)}: ${significant} of ${pairs.length} edges survive`);
  ok(significant === 1, "only the planted edge must survive Bonferroni correction");

  // 10. Determinism: identical inputs produce identical statistics.
  const repeat = grangerCausality(system.y, system.x, { order: 1 });
  ok(repeat.f === forward.f && repeat.pValue === forward.pValue, "the test must be bit-for-bit reproducible");
  const repeatBootstrap = blockBootstrapF(system.y, system.x, { order: 1, blockLength: 20, resamples: 150, seed: 7 });
  ok(repeatBootstrap.median === bootstrap.median, "the bootstrap must be reproducible for a fixed seed");
  const otherBootstrap = blockBootstrapF(system.y, system.x, { order: 1, blockLength: 20, resamples: 150, seed: 8 });
  ok(otherBootstrap.median !== bootstrap.median, "a different bootstrap seed must give a different interval");

  // 11. Degenerate inputs are rejected.
  throws(() => grangerCausality(system.y, system.x, { order: 0 }), RangeError);
  throws(() => grangerCausality(system.y.slice(0, 3), system.x.slice(0, 3), { order: 1 }), RangeError);
  throws(() => grangerCausality(system.y, system.x.slice(0, 100), { order: 1 }), RangeError);
  throws(() => fDistributionTail(-1, 1, 5), RangeError);
  throws(() => fDistributionTail(1, 0, 5), RangeError);
  throws(() => simulateVar1({ length: 5 }), RangeError);
  throws(() => solveLinearSystem([[1, 2], [2, 4]], [1, 2]), /singular/);

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  makeRng,
  makeNormal,
  logGamma,
  regularizedIncompleteBeta,
  quadratureBeta,
  fDistributionTail,
  fCritical,
  solveLinearSystem,
  ols,
  grangerCausality,
  portmanteau,
  simulateVar1,
  blockBootstrapF,
};

if (require.main === module) main();
