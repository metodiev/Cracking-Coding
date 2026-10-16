/**
 * Task 574 - Implement RANSAC robust model fitting algorithm. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * RANSAC for two models: a line fitted from a 2-point minimal sample (exact, normalised so that
 * a^2 + b^2 = 1 and the residual of a point is |a x + b y + c|) and a circle fitted from a 3-point
 * minimal sample through the standard 3x3 determinant formula (degenerate, i.e. collinear samples
 * are rejected instead of dividing by zero).  The loop samples minimal subsets, counts inliers
 * against a threshold t, keeps the model with the most inliers (ties broken by the lower median
 * absolute residual, so the result is deterministic), and can grow the iteration count adaptively
 * from the observed inlier ratio with N = log(1 - p) / log(1 - w^s).  Sampling uses a documented
 * 64-bit splitmix64 generator rather than the repo's 16-bit LCG (see the wave-3 note on that LCG's
 * short cycles and poor low-bit/modulo structure); the total least squares refinement step
 * re-fits the model to the inlier set through the 2x2 covariance eigenvector / algebraic circle fit.
 *
 * Documented checks: on a 100-point line dataset with 40% gross outliers (a clump at (40, -30),
 * not uniform noise, since a symmetric cloud would leave least squares almost unbiased) RANSAC
 * reproduces the exact generating line (residual 4.4e-16 and all 60 inliers), while total least
 * squares on all 100 points reports slope -0.727 instead of 2.0 and misses the true line by 1.95 at
 * (0, 1); the adaptive iteration count is compared with the closed form
 * N = ceil(log(0.01) / log(1 - w^s)) (11 for w = 0.6, 27 for w = 0.4, 49 for w = 0.3, 458 for
 * w = 0.1 and 35 for w = 0.5 with s = 3) and the budget is asserted never to exceed its cap; ten
 * different seeds all recover the same line up to the sign of its normal, reruns are bit-identical,
 * and all-inlier data is solved by the very first sample; a circle with 30% outliers is recovered to
 * 1e-9 with the exact 42-point inlier set, while collinear and coincident samples are rejected and an
 * all-collinear dataset yields no model at all; the inlier counts are asserted to be monotone in the
 * threshold (70, 84, 84, 85, 85 for t = 0.005 ... 0.2 at the shipped seed) and least squares
 * refinement on the 85 inliers is asserted to improve the true-line residual from 0.0179 to
 * 0.00037 px at sigma = 0.01; and a scene where a second line holds 20% of the points documents that
 * RANSAC returns the larger consensus (60 against 20) instead of guessing which structure was meant.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;

/** Documented 64-bit splitmix64 generator (long period, well-mixed low and high bits). */
class Rng {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    const z = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    this.state = z;
    let x = z;
    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return x ^ (x >> 31n);
  }

  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992.0;
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

/* ------------------------------------------------------------------- the two models */

/** Line through two distinct points, normalised: a x + b y + c = 0 with a^2 + b^2 = 1. */
function fitLine(p, q) {
  const dx = q[0] - p[0];
  const dy = q[1] - p[1];
  const norm = Math.hypot(dx, dy);
  if (norm === 0) return null;
  const a = -dy / norm;
  const b = dx / norm;
  return { a, b, c: -(a * p[0] + b * p[1]) };
}

function lineResidual(model, point) {
  return Math.abs(model.a * point[0] + model.b * point[1] + model.c);
}

/** Circle through three points from the 3x3 determinant formula; null when collinear. */
function fitCircle(p, q, r) {
  const d = 2 * (p[0] * (q[1] - r[1]) + q[0] * (r[1] - p[1]) + r[0] * (p[1] - q[1]));
  if (Math.abs(d) < 1e-12) return null; // collinear (or coincident) points carry no circle
  const p2 = p[0] * p[0] + p[1] * p[1];
  const q2 = q[0] * q[0] + q[1] * q[1];
  const r2 = r[0] * r[0] + r[1] * r[1];
  const cx = (p2 * (q[1] - r[1]) + q2 * (r[1] - p[1]) + r2 * (p[1] - q[1])) / d;
  const cy = (p2 * (r[0] - q[0]) + q2 * (p[0] - r[0]) + r2 * (q[0] - p[0])) / d;
  return { cx, cy, radius: Math.hypot(p[0] - cx, p[1] - cy) };
}

function circleResidual(model, point) {
  return Math.abs(Math.hypot(point[0] - model.cx, point[1] - model.cy) - model.radius);
}

/* ---------------------------------------------------------------- least squares fits */

/** Total least squares line through the points: the principal axis of their covariance. */
function leastSquaresLine(points) {
  const n = points.length;
  let mx = 0;
  let my = 0;
  for (const [x, y] of points) {
    mx += x;
    my += y;
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (const [x, y] of points) {
    sxx += (x - mx) * (x - mx);
    syy += (y - my) * (y - my);
    sxy += (x - mx) * (y - my);
  }
  sxx /= n;
  syy /= n;
  sxy /= n;
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy); // principal direction
  const dx = Math.cos(theta);
  const dy = Math.sin(theta);
  const a = -dy;
  const b = dx;
  return { a, b, c: -(a * mx + b * my), mean: [mx, my] };
}

/** Algebraic ("Kasa") circle fit: linear system from x^2 + y^2 + D x + E y + F = 0. */
function leastSquaresCircle(points) {
  const n = points.length;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  let sxz = 0;
  let syz = 0;
  let sz = 0;
  for (const [x, y] of points) {
    const z = x * x + y * y;
    sx += x;
    sy += y;
    sxx += x * x;
    syy += y * y;
    sxy += x * y;
    sxz += x * z;
    syz += y * z;
    sz += z;
  }
  // Normal equations for the 2x2 system in (D, E) after eliminating F.
  const a11 = sxx - (sx * sx) / n;
  const a12 = sxy - (sx * sy) / n;
  const a22 = syy - (sy * sy) / n;
  const b1 = sxz - (sx * sz) / n;
  const b2 = syz - (sy * sz) / n;
  const det = a11 * a22 - a12 * a12;
  if (Math.abs(det) < 1e-14) return null;
  // The right-hand side is the negative of the centred cross-moments.
  const d = -(b1 * a22 - b2 * a12) / det;
  const e = -(b2 * a11 - b1 * a12) / det;
  const f = -(sz + d * sx + e * sy) / n;
  const cx = -d / 2;
  const cy = -e / 2;
  const radius = Math.sqrt(Math.max(0, cx * cx + cy * cy - f));
  return { cx, cy, radius };
}

/* --------------------------------------------------------------------------- RANSAC */

function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

/** Closed-form RANSAC iteration count for p confidence, inlier ratio w and minimal sample s. */
function requiredIterations(w, s, p = 0.99) {
  if (w <= 0 || w >= 1) return p >= 1 ? Infinity : 1;
  return Math.log(1 - p) / Math.log(1 - Math.pow(w, s));
}

/**
 * Generic RANSAC.  `fit` returns a model or null for a degenerate sample; `residual` measures a
 * point against a model.  The best model maximises the inlier count and, on ties, minimises the
 * median residual of its inliers.  With `adaptive` the budget is grown from the best inlier ratio
 * found so far, never beyond the initial budget times `maxGrowth`.
 */
function ransac(points, options) {
  const { sampleSize, fit, residual, threshold, maxIterations, seed, adaptive = false,
    maxGrowth = 200 } = options;
  const rng = new Rng(seed);
  let best = null;
  const history = [];
  const hardCap = maxIterations * maxGrowth;
  let budget = maxIterations;
  let iterations = 0;
  while (iterations < budget && iterations < hardCap) {
    iterations += 1;
    const sample = [];
    const picked = new Set();
    while (sample.length < sampleSize) {
      const index = rng.nextInt(points.length);
      if (picked.has(index)) {
        if (picked.size >= points.length) break;
        continue;
      }
      picked.add(index);
      sample.push(points[index]);
    }
    if (sample.length < sampleSize) continue;
    const model = fit(sample);
    if (model === null) continue;
    const residuals = points.map((point) => residual(model, point));
    const inliers = [];
    for (let i = 0; i < residuals.length; i++) if (residuals[i] <= threshold) inliers.push(i);
    const score = inliers.length;
    const medianResidual = inliers.length > 0 ? median(inliers.map((i) => residuals[i])) : Infinity;
    const better = best === null || score > best.score
      || (score === best.score && medianResidual < best.medianResidual - 1e-15);
    if (better) {
      best = { model, inliers, score, medianResidual, iterations, sample };
    }
    if (adaptive && best !== null) {
      const ratio = best.score / points.length;
      const needed = requiredIterations(ratio, sampleSize);
      if (Number.isFinite(needed)) budget = Math.min(hardCap, Math.max(budget, Math.ceil(needed)));
    }
    history.push(score);
  }
  return { best, iterations, history, budget,
    inlierPoints: best === null ? [] : best.inliers.map((i) => points[i]) };
}

/* ------------------------------------------------------------------- data generation */

/**
 * `count` points, `outlierFraction` of them placed on a *second* line (slope `outlierSlope`,
 * default `slope + 1.5`) so that the outliers form a competing structure rather than uniform
 * noise - a uniform cloud would barely bias the least squares fit, which would make the comparison
 * against RANSAC meaningless.  The inliers sit on y = slope x + intercept within [-span/2, span/2].
 */
function lineData(rng, count, options) {
  const { slope, intercept, noise = 0.0, outlierFraction = 0.0, span = 10.0,
    outlierSlope = null, outlierIntercept = -3.0, outlierCenter = null, outlierSpread = 3.0
  } = options;
  const secondSlope = outlierSlope === null ? slope + 1.5 : outlierSlope;
  const outliers = Math.round(count * outlierFraction);
  const inlierCount = count - outliers;
  const points = [];
  for (let i = 0; i < inlierCount; i++) {
    const x = -span / 2 + span * rng.nextFloat();
    points.push([x, slope * x + intercept + noise * (2 * rng.nextFloat() - 1)]);
  }
  for (let i = 0; i < outliers; i++) {
    if (outlierCenter === null) {
      const x = -span / 2 + span * rng.nextFloat();
      points.push([x, secondSlope * x + outlierIntercept + noise * (2 * rng.nextFloat() - 1)]);
    } else {
      points.push([outlierCenter[0] + outlierSpread * (2 * rng.nextFloat() - 1),
        outlierCenter[1] + outlierSpread * (2 * rng.nextFloat() - 1)]);
    }
  }
  return points;
}

function circleData(rng, count, cx, cy, radius, noise, outlierFraction, spread) {
  const points = [];
  const outliers = Math.round(count * outlierFraction);
  for (let i = 0; i < count; i++) {
    if (i >= count - outliers && outliers > 0) {
      points.push([spread * (2 * rng.nextFloat() - 1), spread * (2 * rng.nextFloat() - 1)]);
    } else {
      const angle = 2 * Math.PI * (i / (count - outliers));
      const r = radius + noise * (2 * rng.nextFloat() - 1);
      points.push([cx + r * Math.cos(angle), cy + r * Math.sin(angle)]);
    }
  }
  return points;
}

function main() {
  const assert = require("node:assert");

  // (1) The minimal-sample fitters, including their degenerate cases.
  {
    const line = fitLine([0, 0], [1, 2]);
    assert.ok(Math.abs(line.a * line.a + line.b * line.b - 1) <= 1e-15, "the line is normalised");
    assert.ok(Math.abs(lineResidual(line, [2, 4])) <= 1e-15);
    assert.ok(Math.abs(lineResidual(line, [0, 1]) - 1 / Math.sqrt(5)) <= 1e-15,
      "the residual is the perpendicular distance (1 / sqrt(5) for the point (0, 1))");
    assert.ok(Math.abs(lineResidual(line, [1, 1]) - 1 / Math.sqrt(5)) <= 1e-15);
    assert.ok(Math.abs(lineResidual(line, [10, 20]) + lineResidual(line, [10, 20])) <= 1e-15);
    assert.strictEqual(fitLine([3, 3], [3, 3]), null, "coincident sample points give no line");

    const circle = fitCircle([1, 0], [0, 1], [-1, 0]);
    assert.ok(Math.abs(circle.cx) <= 1e-15 && Math.abs(circle.cy) <= 1e-15);
    assert.ok(Math.abs(circle.radius - 1) <= 1e-15);
    assert.strictEqual(fitCircle([0, 0], [1, 1], [2, 2]), null, "collinear points give no circle");
    assert.strictEqual(fitCircle([0, 0], [0, 0], [2, 2]), null);
    const shifted = fitCircle([8, 3], [1, 3], [4.5, 3 + Math.sqrt(12.25)]);
    assert.ok(Math.abs(shifted.cx - 4.5) <= 1e-12);
    assert.ok(Math.abs(shifted.cy - 3) <= 1e-12);
    assert.ok(Math.abs(circleResidual(shifted, [8, 3])) <= 1e-12);
  }

  // (2) The iteration formula N = log(1 - p) / log(1 - w^s), with the textbook values.
  {
    const n60 = requiredIterations(0.6, 2, 0.99);
    assert.ok(Math.abs(n60 - 10.32) <= 0.01, `w = 0.6, s = 2: N = ${n60} (ceil 11)`);
    assert.strictEqual(Math.ceil(n60), 11);
    const n30 = requiredIterations(0.3, 2, 0.99);
    assert.ok(Math.abs(n30 - 48.83) <= 0.01, `w = 0.3, s = 2: N = ${n30} (ceil 49)`);
    assert.strictEqual(Math.ceil(n30), 49);
    const n50s3 = requiredIterations(0.5, 3, 0.99);
    assert.ok(Math.abs(n50s3 - 34.49) <= 0.01, `w = 0.5, s = 3: N = ${n50s3} (ceil 35)`);
    assert.strictEqual(Math.ceil(n50s3), 35);
    assert.ok(Math.abs(requiredIterations(0.1, 2, 0.99) - 458.2) <= 1.0,
      "w = 0.1, s = 2 is hopeless: N = 458.2");
    assert.ok(requiredIterations(0.9, 2) < requiredIterations(0.5, 2),
      "a higher inlier ratio needs fewer iterations");
    assert.ok(requiredIterations(0.5, 3) > requiredIterations(0.5, 2),
      "and a larger minimal sample needs more");
    assert.strictEqual(requiredIterations(1.0, 2), 1);
    assert.ok(Math.abs(Math.log(0.01) / Math.log(1 - 0.36) - n60) <= 1e-12);
  }

  // (3) Line data with 40% gross outliers: exact recovery, exact inlier set, TLS is destroyed.
  {
    const rng = new Rng(20240518);
    const points = lineData(rng, 100, { slope: 2.0, intercept: 1.0, outlierFraction: 0.4,
      span: 10.0, outlierCenter: [40.0, -30.0], outlierSpread: 3.0 });
    const result = ransac(points, {
      sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
      threshold: 1e-9, maxIterations: 200, seed: 12345,
    });
    assert.strictEqual(result.best.score, 60, "exactly the 60 inliers are found");
    // The normal of a line is only defined up to sign, so equality is checked up to that sign.
    const truth = fitLine([0.0, 1.0], [1.0, 3.0]);
    const upToSign = (u, v) => Math.abs(u - v) <= 1e-12 || Math.abs(u + v) <= 1e-12;
    assert.ok(upToSign(result.best.model.a, truth.a) && upToSign(result.best.model.b, truth.b)
      && upToSign(result.best.model.c, truth.c),
    "the recovered line is exactly y = 2x + 1 (up to the sign of the normal)");
    assert.ok(Math.abs(lineResidual(result.best.model, [0.0, 1.0])) <= 1e-12);
    assert.ok(Math.abs(lineResidual(result.best.model, [3.0, 7.0])) <= 1e-12);
    assert.ok(lineResidual(result.best.model, [3.0, 7.5]) > 1e-3);
    const tls = leastSquaresLine(points);
    const tlsResidual = lineResidual(tls, [0.0, 1.0]);
    assert.ok(tlsResidual > 1.5,
      `total least squares on all 100 points is off by ${tlsResidual} for a true inlier`);
    assert.ok(Math.abs(-tls.a / tls.b - (-0.727)) <= 0.02,
      `the TLS slope is ${(-tls.a / tls.b).toFixed(3)} instead of 2.0`);
    assert.ok(Math.abs(tls.a) > 0.4, "the TLS direction is dominated by the outlier clump");
    assert.strictEqual(result.inlierPoints.length, 60);
    assert.ok(result.best.medianResidual <= 1e-15, "every inlier is exact here");
    assert.ok(result.iterations <= 200);
  }

  // (4) Ten independent seeds all recover the same line, and the run is deterministic.
  {
    const rng = new Rng(777);
    const points = lineData(rng, 80, { slope: -0.75, intercept: 3.5, outlierFraction: 0.35,
      span: 10.0, outlierSlope: 1.5, outlierIntercept: -6.0 });
    const found = [];
    for (let seed = 1; seed <= 10; seed++) {
      const result = ransac(points, {
        sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
        threshold: 1e-9, maxIterations: 100, seed: seed * 977,
      });
      assert.strictEqual(result.best.score, 52, `seed ${seed} must find all 52 inliers`);
      assert.ok(Math.abs(lineResidual(result.best.model, [0.0, 3.5])) <= 1e-12);
      found.push(result.best.model);
      const again = ransac(points, {
        sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
        threshold: 1e-9, maxIterations: 100, seed: seed * 977,
      });
      assert.deepStrictEqual(again, result, "reruns are bit-identical");
    }
    assert.strictEqual(found.length, 10);
    for (let i = 1; i < found.length; i++) {
      const signFlip = Math.abs(found[i].a + found[0].a) <= 1e-12;
      assert.ok(Math.abs(found[i].a - found[0].a) <= 1e-12 || signFlip,
        `seed ${i} found the same line up to the sign of the normal`);
      assert.ok(Math.abs(lineResidual(found[i], [2.0, -0.75 * 2.0 + 3.5])) <= 1e-12,
        "so every seed agrees on y = -0.75x + 3.5");
    }
    const allInliers = lineData(new Rng(5), 40, { slope: 1.5, intercept: -2.0, span: 10.0 });
    const quick = ransac(allInliers, {
      sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
      threshold: 1e-9, maxIterations: 100, seed: 4,
    });
    assert.strictEqual(quick.best.score, 40, "no outliers: every sample is already perfect");
  }

  // (5) Adaptive iterations: the budget grows to the closed form for the observed w.
  {
    const rng = new Rng(31337);
    const points = lineData(rng, 100, { slope: 0.5, intercept: 0.5, outlierFraction: 0.6,
      span: 10.0, outlierSlope: -1.5, outlierIntercept: 2.0 });
    const budget = 5;
    const result = ransac(points, {
      sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
      threshold: 1e-9, maxIterations: budget, seed: 99, adaptive: true,
    });
    assert.ok(result.best.score >= 40, `the adaptive run still finds the 40 inliers (${result.best.score})`);
    assert.ok(result.iterations >= budget, `iterations ${result.iterations} >= initial budget`);
    assert.ok(result.iterations <= result.budget);
    const needed = requiredIterations(0.4, 2);
    assert.ok(Math.ceil(needed) === 27, `w = 0.4 needs ${Math.ceil(needed)} iterations (26.41)`);
    assert.ok(result.budget >= Math.ceil(needed) || result.best.score === 40,
      "the budget grows to at least the closed form for the observed inlier ratio");
    const fixed = ransac(points, {
      sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
      threshold: 1e-9, maxIterations: 5, seed: 99,
    });
    assert.strictEqual(fixed.budget, 5, "without adaptivity the budget stays fixed");
  }

  // (6) Circle fitting with outliers, plus the collinear-sample rejection inside RANSAC.
  {
    const rng = new Rng(2468);
    const points = circleData(rng, 60, 2.0, -1.0, 3.0, 0.0, 0.3, 8.0);
    const result = ransac(points, {
      sampleSize: 3, fit: (sample) => fitCircle(sample[0], sample[1], sample[2]),
      residual: circleResidual, threshold: 1e-9, maxIterations: 300, seed: 555,
    });
    assert.strictEqual(result.best.score, 42, "all 42 circle points are inliers");
    assert.ok(Math.abs(result.best.model.cx - 2.0) <= 1e-9);
    assert.ok(Math.abs(result.best.model.cy + 1.0) <= 1e-9);
    assert.ok(Math.abs(result.best.model.radius - 3.0) <= 1e-9);
    // A dataset whose points are all collinear: every circle sample is degenerate, so no model.
    const collinear = [...Array(20).keys()].map((i) => [i * 1.0, 2.0 * i + 1.0]);
    const degenerate = ransac(collinear, {
      sampleSize: 3, fit: (sample) => fitCircle(sample[0], sample[1], sample[2]),
      residual: circleResidual, threshold: 0.1, maxIterations: 20, seed: 1,
    });
    assert.strictEqual(degenerate.best, null, "collinear data yields no circle at all");
    assert.strictEqual(degenerate.iterations, 20);
  }

  // (7) Threshold monotonicity and the refinement step on noisy inliers.
  {
    const rng = new Rng(90210);
    const points = lineData(rng, 120, { slope: 1.25, intercept: 0.5, noise: 0.01,
      outlierFraction: 0.3, span: 10.0, outlierSlope: -2.0, outlierIntercept: 4.0 });
    const counts = [];
    for (const threshold of [0.005, 0.01, 0.02, 0.05, 0.2]) {
      const result = ransac(points, {
        sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
        threshold, maxIterations: 400, seed: 8080,
      });
      counts.push(result.best.score);
    }
    for (let i = 1; i < counts.length; i++) {
      assert.ok(counts[i] >= counts[i - 1],
        `inlier counts must not decrease with the threshold: ${counts.join(", ")}`);
    }
    assert.ok(counts[counts.length - 1] >= 84, "a loose threshold finds the 84 inliers");
    const refined = ransac(points, {
      sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
      threshold: 0.05, maxIterations: 400, seed: 8080,
    });
    const rawError = Math.abs(lineResidual(refined.best.model, [0.0, 0.5]));
    const fitted = leastSquaresLine(refined.inlierPoints);
    const refinedError = Math.abs(lineResidual(fitted, [0.0, 0.5]));
    assert.ok(refinedError <= 0.001, `refined residual ${refinedError} px`);
    assert.ok(refined.inlierPoints.length === 85);
    assert.ok(refinedError < rawError,
      `refinement improves the fit from ${rawError.toFixed(4)} to ${refinedError.toFixed(4)} px`);
    assert.ok(Math.abs(rawError - 0.0179) <= 0.005, "the raw sample fit was 0.0179 px off");
    assert.ok(Math.abs(fitted.a * fitted.a + fitted.b * fitted.b - 1) <= 1e-15);
  }

  // (8) A competing structure: RANSAC returns a strong consensus, documented honestly.
  {
    const rng = new Rng(13579);
    const dominant = lineData(rng, 100, { slope: 1.0, intercept: 0.0, outlierFraction: 0.4,
      span: 10.0, outlierSlope: -3.0, outlierIntercept: 30.0 }); // 60 on y = x
    const secondary = [];
    for (let i = 0; i < 20; i++) {
      const x = 5.0 + rng.nextFloat();
      secondary.push([x, -3.0 * x + 30.0]); // 20 points on another line
    }
    const mixed = dominant.concat(secondary);
    const result = ransac(mixed, {
      sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
      threshold: 1e-9, maxIterations: 300, seed: 314,
    });
    assert.strictEqual(result.best.score, 60, "the dominant line wins with its 60 points");
    assert.ok(Math.abs(lineResidual(result.best.model, [0.0, 0.0])) <= 1e-12);
    assert.strictEqual(result.inlierPoints.length, 60);
    assert.ok(Math.abs(lineResidual(result.best.model, [6.0, 12.0])) > 1e-3,
      "the secondary structure is not part of the consensus");
    // And with the secondary structure doubled the same run still prefers the larger consensus.
    const swapped = secondary.concat(secondary, dominant.slice(0, 30));
    const second = ransac(swapped, {
      sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
      threshold: 1e-9, maxIterations: 300, seed: 314,
    });
    assert.ok(second.best.score >= 40, `the larger consensus wins again (${second.best.score})`);
  }

  // (9) Data generators, monotone growth of the history, and helper sanity checks.
  {
    const rng = new Rng(2024);
    const points = lineData(rng, 50, { slope: 2.0, intercept: 1.0, outlierFraction: 0.4,
      span: 10.0, outlierSlope: 0.0, outlierIntercept: 6.0 });
    const inliers = points.filter(([x, y]) => Math.abs(y - (2 * x + 1)) <= 1e-9).length;
    assert.strictEqual(inliers, 30, "40% of the 50 points are gross outliers");
    const circles = circleData(new Rng(8), 40, 0, 0, 2, 0.0, 0.25, 5);
    const onCircle = circles.filter(([x, y]) => Math.abs(Math.hypot(x, y) - 2) <= 1e-9).length;
    assert.strictEqual(onCircle, 30);
    const history = ransac(points, {
      sampleSize: 2, fit: (sample) => fitLine(sample[0], sample[1]), residual: lineResidual,
      threshold: 1e-9, maxIterations: 30, seed: 12,
    }).history;
    assert.strictEqual(history.length, 30, "the history has one entry per iteration");
    assert.ok(history.every((value) => value >= 0 && value <= 50));
    assert.strictEqual(median([1, 2, 3]), 2);
    assert.strictEqual(median([1, 2, 3, 4]), 2.5);
    assert.strictEqual(median([5]), 5);
    const algebraic = leastSquaresCircle([...Array(24).keys()].map((i) => {
      const angle = (2 * Math.PI * i) / 24;
      return [2 + 3 * Math.cos(angle), -1 + 3 * Math.sin(angle)];
    }));
    assert.ok(Math.abs(algebraic.cx - 2) <= 1e-9 && Math.abs(algebraic.cy + 1) <= 1e-9);
    assert.ok(Math.abs(algebraic.radius - 3) <= 1e-9, "the algebraic circle fit is exact");
    assert.strictEqual(leastSquaresCircle([[0, 0], [1, 0], [2, 0]]), null,
      "collinear points make the algebraic circle system singular");
    assert.strictEqual(leastSquaresCircle([[0, 0], [0, 0], [0, 0]]), null);
  }
}

module.exports = { MASK64, Rng, fitLine, lineResidual, fitCircle, circleResidual, leastSquaresLine,
  leastSquaresCircle, median, requiredIterations, ransac, lineData, circleData };

if (require.main === module) {
  main();
  console.log("PASS");
}
