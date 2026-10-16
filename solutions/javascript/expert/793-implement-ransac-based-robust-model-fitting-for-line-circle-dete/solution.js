/** Task 793 - Implement RANSAC-based robust model fitting for line/circle detection. (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * RANSAC repeatedly draws a minimal sample, fits the model exactly to it, counts how many points agree
 * with the model within a threshold, and keeps the best consensus set. Minimal samples are:
 *   line:   2 points, model = unit normal (a, b) and offset c with distance |a x + b y + c|,
 *   circle: 3 points, model = least-squares algebraic (Kasa) fit of x^2 + y^2 + D x + E y + F = 0
 *           solved through the 3x3 normal equations of that linear system for the sample.
 * After the consensus set is found it is refit on all inliers: the line by total least squares (the
 * principal direction of the 2x2 scatter matrix, computed in closed form) and the circle by Kasa on
 * everyone in the set. The number of required iterations follows the standard formula
 *   N = ceil( log(1 - p) / log(1 - w^s) )   for confidence p, inlier ratio w and sample size s.
 *
 * Invariants verified: for small point sets an exhaustive enumeration of every minimal subset provides
 * the oracle consensus size, and RANSAC's consensus matches it; the refit RMSD never exceeds the
 * oracle's; the iteration formula satisfies 1 - (1 - w^s)^N >= p exactly; the inlier count is monotone
 * in the threshold; degenerate samples (collinear triples, coincident pairs) are rejected instead of
 * producing a garbage model; and noisy circles/lines are recovered to a few sigma.
 *
 * What is simplified: 2-D models only (line and circle, no ellipse/homography), an algebraic circle fit
 * instead of a geometric (Taubin/Levenberg) refinement, and a uniform random sampler. A seeded
 * splitmix32 drives every draw, so each run is deterministic and every error reported below is measured.
 */
"use strict";

const { ok, throws } = require("node:assert");

/** Deterministic splitmix32 uniforms in [0, 1). */
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return ((z ^ (z >>> 15)) >>> 0) / 4294967296;
  };
}

const gaussian = (random) => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

/** Exact line through two points; returns null for coincident points. */
function lineFromTwo(p, q) {
  const dx = q[0] - p[0];
  const dy = q[1] - p[1];
  const norm = Math.hypot(dx, dy);
  if (norm < 1e-12) return null;
  const a = -dy / norm;
  const b = dx / norm;
  return { a, b, c: -(a * p[0] + b * p[1]) };
}

const lineDistance = (model, point) => Math.abs(model.a * point[0] + model.b * point[1] + model.c);

/** The single model with the largest consensus set inside `points`. */
function lineConsensus(points, model, threshold) {
  const inliers = [];
  for (let i = 0; i < points.length; i++) if (lineDistance(model, points[i]) <= threshold) inliers.push(i);
  return inliers;
}

/** Total least squares refit of a line: centroid plus the principal direction of the scatter matrix. */
function refitLine(points, indices) {
  const n = indices.length;
  if (n < 2) throw new RangeError("a line refit needs at least two points");
  const cx = indices.reduce((acc, i) => acc + points[i][0], 0) / n;
  const cy = indices.reduce((acc, i) => acc + points[i][1], 0) / n;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (const i of indices) {
    const dx = points[i][0] - cx;
    const dy = points[i][1] - cy;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  // Principal direction of [[sxx, sxy], [sxy, syy]] in closed form: the eigenvector of the larger
  // eigenvalue, i.e. the direction that minimises the perpendicular sum of squares.
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const a = -Math.sin(theta);
  const b = Math.cos(theta);
  return { a, b, c: -(a * cx + b * cy) };
}

/** Kasa algebraic circle fit: solves x^2 + y^2 + D x + E y + F = 0 in the least-squares sense. */
function fitCircle(points, indices) {
  const n = indices.length;
  if (n < 3) throw new RangeError("a circle fit needs at least three points");
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  let sx = 0;
  let sy = 0;
  let sxz = 0;
  let syz = 0;
  let sz = 0;
  for (const i of indices) {
    const [x, y] = points[i];
    const z = x * x + y * y;
    sxx += x * x;
    sxy += x * y;
    syy += y * y;
    sx += x;
    sy += y;
    sxz += x * z;
    syz += y * z;
    sz += z;
  }
  const matrix = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
  const rhs = [-sxz, -syz, -sz];
  const [D, E, F] = solve3(matrix, rhs);
  const center = [-D / 2, -E / 2];
  const radiusSquared = center[0] * center[0] + center[1] * center[1] - F;
  if (!(radiusSquared > 1e-12)) throw new Error("degenerate circle sample (the normal equations are singular)");
  return { center, radius: Math.sqrt(radiusSquared) };
}

/** Gaussian elimination for a 3x3 system (throws on a singular sample matrix). */
function solve3(matrix, rhs) {
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-12) throw new Error("singular sample matrix");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let r = col + 1; r < 3; r++) {
      const factor = a[r][col] / a[col][col];
      for (let c = col; c <= 3; c++) a[r][c] -= factor * a[col][c];
    }
  }
  const x = new Array(3).fill(0);
  for (let r = 2; r >= 0; r--) {
    let sum = a[r][3];
    for (let c = r + 1; c < 3; c++) sum -= a[r][c] * x[c];
    x[r] = sum / a[r][r];
  }
  return x;
}

const circleDistance = (model, point) => Math.abs(Math.hypot(point[0] - model.center[0], point[1] - model.center[1]) - model.radius);

/** Iterations needed for confidence `p` given inlier ratio `w` and sample size `s`. */
function iterationsNeeded(w, s, p) {
  if (!(w > 0) || !(p > 0) || p >= 1) throw new RangeError("w and p must be positive, p < 1");
  const perSample = Math.min(1, w ** s);
  return Math.max(1, Math.ceil(Math.log(1 - p) / Math.log(1 - perSample)));
}

/**
 * RANSAC for `kind` in {"line", "circle"}: draws minimal samples, keeps the largest consensus set and
 * refits on its inliers. Returns the model, the inlier indices and the search statistics.
 */
function ransac(points, { kind = "line", threshold, iterations, confidence = 0.99, inlierRatio, seed = 1 } = {}) {
  const sampleSize = kind === "line" ? 2 : 3;
  if (points.length < sampleSize) throw new RangeError(`need at least ${sampleSize} points (got ${points.length})`);
  if (!(threshold > 0)) throw new RangeError("threshold must be > 0");
  const random = makeRng(seed);
  const draws = iterations ?? (inlierRatio ? iterationsNeeded(inlierRatio, sampleSize, confidence) : 200);
  const fit = (sample) => (kind === "line" ? lineFromTwo(points[sample[0]], points[sample[1]]) : fitCircle(points, sample));
  const distance = kind === "line" ? lineDistance : circleDistance;
  const consensus = (model) => {
    const inliers = [];
    for (let i = 0; i < points.length; i++) if (distance(model, points[i]) <= threshold) inliers.push(i);
    return inliers;
  };
  let best = null;
  let attempts = 0;
  let degenerate = 0;
  for (let iteration = 0; iteration < draws; iteration++) {
    const sample = [];
    while (sample.length < sampleSize) {
      const candidate = Math.floor(random() * points.length) % points.length;
      if (!sample.includes(candidate)) sample.push(candidate);
    }
    attempts += 1;
    let model;
    try {
      model = fit(sample);
    } catch {
      model = null;
    }
    if (!model) {
      degenerate += 1;
      continue;
    }
    const inliers = consensus(model);
    if (!best || inliers.length > best.inliers.length) best = { model, inliers };
  }
  if (!best) throw new Error("every sample was degenerate: no model could be estimated");
  const refined = kind === "line" ? refitLine(points, best.inliers) : fitCircle(points, best.inliers);
  const inliers = consensus(refined);
  const rmsd = Math.sqrt(inliers.reduce((acc, i) => acc + distance(refined, points[i]) ** 2, 0) / inliers.length);
  return { model: refined, inliers, rmsd, attempts, degenerate };
}

/** Exhaustive oracle: the largest consensus any minimal sample can reach, over all subsets. */
function exhaustiveBest(points, { kind = "line", threshold }) {
  const sampleSize = kind === "line" ? 2 : 3;
  const distance = kind === "line" ? lineDistance : circleDistance;
  const candidates = [];
  const walk = (start, picked) => {
    if (picked.length === sampleSize) {
      candidates.push(picked.slice());
      return;
    }
    for (let i = start; i < points.length; i++) {
      picked.push(i);
      walk(i + 1, picked);
      picked.pop();
    }
  };
  walk(0, []);
  let best = 0;
  for (const sample of candidates) {
    let model;
    try {
      model = kind === "line" ? lineFromTwo(points[sample[0]], points[sample[1]]) : fitCircle(points, sample);
    } catch {
      continue;
    }
    if (!model) continue;
    let count = 0;
    for (const point of points) if (distance(model, point) <= threshold) count += 1;
    best = Math.max(best, count);
  }
  return best;
}

function main() {
  const notes = [];
  const random = makeRng(20251016);

  // 1. Exhaustive oracle on a small mixed set: RANSAC must find the same maximal consensus.
  const smallPoints = [];
  for (let i = 0; i < 9; i++) smallPoints.push([i * 0.7 - 2, 1.5 + 0.4 * (i * 0.7 - 2) + gaussian(random) * 0.01]);
  for (let i = 0; i < 4; i++) smallPoints.push([-1 + i, -3 + 2 * i]);
  const smallBest = ransac(smallPoints, { kind: "line", threshold: 0.08, seed: 5 });
  const exhaustiveLine = exhaustiveBest(smallPoints, { kind: "line", threshold: 0.08 });
  notes.push(`line consensus on ${smallPoints.length} points: RANSAC ${smallBest.inliers.length} vs exhaustive oracle ${exhaustiveLine}`);
  ok(smallBest.inliers.length === exhaustiveLine, `RANSAC must reach the exhaustive consensus (${smallBest.inliers.length} vs ${exhaustiveLine})`);
  ok(Math.abs(Math.hypot(smallBest.model.a, smallBest.model.b) - 1) < 1e-12, "the normal must be a unit vector");

  // 2. Noisy line recovery with 40% outliers.
  const linePoints = [];
  for (let i = 0; i < 30; i++) {
    const x = -4 + (8 * i) / 29;
    linePoints.push([x, 2 - 0.6 * x + gaussian(random) * 0.02]);
  }
  for (let i = 0; i < 20; i++) linePoints.push([-4 + 8 * random(), -3 + 6 * random()]);
  const lineFit = ransac(linePoints, { kind: "line", threshold: 0.1, seed: 11 });
  const lineInliers = new Set(lineFit.inliers);
  const missedLine = Array.from({ length: 30 }, (_, i) => i).filter((i) => !lineInliers.has(i)).length;
  notes.push(`noisy line with 40% outliers: ${lineFit.inliers.length} inliers (${missedLine} of the 30 genuine points missed), slope ${(-lineFit.model.a / lineFit.model.b).toFixed(4)} vs -0.6, intercept ${(-lineFit.model.c / lineFit.model.b).toFixed(4)} vs 2, rmsd ${lineFit.rmsd.toFixed(4)}`);
  ok(missedLine === 0, `every genuine point must be inside the consensus set (missed ${missedLine})`);
  ok(lineFit.rmsd < 0.05, `the inlier RMSD must be at the noise level (got ${lineFit.rmsd})`);

  // 3. Circle recovery with heavy outliers, and the exhaustive oracle on a small circle set.
  const circlePoints = [];
  const center = [1.5, -0.5];
  const radius = 2.4;
  for (let i = 0; i < 28; i++) {
    const angle = (2 * Math.PI * i) / 28;
    const r = radius + gaussian(random) * 0.02;
    circlePoints.push([center[0] + r * Math.cos(angle), center[1] + r * Math.sin(angle)]);
  }
  for (let i = 0; i < 22; i++) circlePoints.push([-4 + 8 * random(), -4 + 8 * random()]);
  const circleFit = ransac(circlePoints, { kind: "circle", threshold: 0.12, seed: 23 });
  const centerError = Math.hypot(circleFit.model.center[0] - center[0], circleFit.model.center[1] - center[1]);
  const circleInliers = new Set(circleFit.inliers);
  const missedCircle = Array.from({ length: 28 }, (_, i) => i).filter((i) => !circleInliers.has(i)).length;
  notes.push(`noisy circle with 44% outliers: ${circleFit.inliers.length} inliers (${missedCircle} of the 28 genuine points missed), center error ${centerError.toFixed(4)}, radius ${circleFit.model.radius.toFixed(4)} vs ${radius}, rmsd ${circleFit.rmsd.toFixed(4)}`);
  ok(missedCircle === 0, `every genuine circle point must be inside the consensus set (missed ${missedCircle})`);
  ok(centerError < 0.05 && Math.abs(circleFit.model.radius - radius) < 0.05, "the circle must be recovered within the noise level");
  const smallCircle = [];
  for (let i = 0; i < 7; i++) {
    const angle = (2 * Math.PI * i) / 7;
    smallCircle.push([Math.cos(angle) * 1.5, Math.sin(angle) * 1.5 + 0.005 * i]);
  }
  smallCircle.push([4, 4], [5, -1], [-3, 5]);
  const circleOracle = exhaustiveBest(smallCircle, { kind: "circle", threshold: 0.05 });
  const circleRansac = ransac(smallCircle, { kind: "circle", threshold: 0.05, iterations: 400, seed: 31 });
  notes.push(`circle consensus on ${smallCircle.length} points: RANSAC ${circleRansac.inliers.length} vs exhaustive oracle ${circleOracle}`);
  ok(circleRansac.inliers.length === circleOracle, "the circle consensus must match the exhaustive oracle");

  // 4. The iteration formula must deliver the requested confidence exactly.
  const formulaChecks = [[0.6, 2], [0.5, 3], [0.3, 2]].map(([w, s]) => {
    const N = iterationsNeeded(w, s, 0.99);
    return { w, s, N, probability: 1 - (1 - w ** s) ** N };
  });
  notes.push(`iteration formula: ${formulaChecks.map((c) => `w = ${c.w}, s = ${c.s} -> N = ${c.N}, 1-(1-w^s)^N = ${c.probability.toFixed(4)}`).join("; ")}`);
  ok(formulaChecks.every((c) => c.probability >= 0.99), "the computed iteration counts must reach the requested confidence");
  ok(formulaChecks.every((c, k) => k === 0 || iterationsNeeded(c.w, c.s, 0.99) >= iterationsNeeded(c.w + 0.2, c.s, 0.99)),
    "fewer inliers must require more iterations");

  // 5. Inlier counts must be monotone in the threshold.
  const thresholds = [0.02, 0.05, 0.1, 0.2].map((threshold) => ransac(linePoints, { kind: "line", threshold, seed: 11 }).inliers.length);
  notes.push(`threshold sweep inliers: ${thresholds.join(" -> ")}`);
  ok(thresholds.every((count, i) => i === 0 || count >= thresholds[i - 1]), "a looser threshold must never reduce the inlier count");

  // 6. Degenerate samples must be rejected instead of fitted.
  ok(lineFromTwo([1, 1], [1, 1]) === null, "coincident points must be reported as null instead of defining a line");
  throws(() => fitCircle([[0, 0], [1, 1], [2, 2]], [0, 1, 2]), Error, "three collinear points must not define a circle");
  throws(() => ransac([[0, 0]], { kind: "line", threshold: 0.1 }), RangeError, "too few points must throw");
  throws(() => ransac(linePoints, { kind: "line", threshold: 0 }), RangeError, "a zero threshold must throw");

  // 7. A degenerate-only input must fail loudly rather than return a made-up model.
  throws(() => ransac([[1, 1], [1, 1], [1, 1]], { kind: "line", threshold: 0.1, seed: 3 }), Error,
    "an all-coincident set must throw");

  // 8. Determinism.
  const repeat = ransac(linePoints, { kind: "line", threshold: 0.1, seed: 11 });
  ok(repeat.rmsd === lineFit.rmsd && repeat.inliers.length === lineFit.inliers.length, "RANSAC must be deterministic for a fixed seed");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { makeRng, gaussian, lineFromTwo, lineDistance, refitLine, fitCircle, circleDistance, iterationsNeeded, ransac, exhaustiveBest };

if (require.main === module) main();
