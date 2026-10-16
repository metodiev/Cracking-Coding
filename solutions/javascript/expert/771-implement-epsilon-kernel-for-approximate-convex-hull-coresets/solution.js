/** Task 771 - Implement epsilon-kernel for approximate convex hull (coresets). (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * A (1-eps)-kernel of a point set S is a small subset K such that, in every direction u,
 * max_{p in K} <p,u> >= (1 - eps) * max_{p in S} <p,u>. In 2-D the construction below is the
 * classical one: take the exact hull, then thin it with a Douglas-Peucker style recursion that
 * only keeps a hull vertex when it sticks out by more than the budget from the chord spanning the
 * chain it splits. Recursion stops on a chain (a, b) exactly when every skipped vertex lies
 * within `budget` of the chord a-b, and that chord is an edge of the final polygon, so
 *
 *     conv(S)  subseteq  conv(K) + B(delta),   delta = max distance of a dropped vertex to conv(K)
 *
 * with delta <= budget. Picking a centre c inside conv(S) and budget = eps * r, where r is the
 * exact inradius of conv(S) around c, gives delta <= eps * r <= eps * max_S <.,u> for every
 * direction, which is precisely the (1 - eps) kernel inequality. The recursion is what makes the
 * error additive over one chord instead of accumulating along the whole boundary.
 *
 * What is simplified: the construction starts from the *exact* hull (a real system would hull an
 * already sampled subset) and works only in 2-D, where kernels need O(1/sqrt(eps)) vertices.
 * What is verified: delta computed exactly as the max point-to-convex-polygon distance of every
 * dropped hull vertex; the kernel inequality checked over *all* critical directions (the normals
 * of every pair of input points - the ratio of two support functions is piecewise monotone between
 * them, so this finite set is complete, not sampled); width preservation; the achieved support gap
 * <= delta; kernel sizes shrink monotonically with eps; eps -> 0 returns the exact hull on strictly
 * convex input; plus degenerate sets (empty, 1, 2, collinear, all-identical, duplicated) and
 * input validation.
 */
"use strict";

/** Andrew's monotone chain; returns the extreme points counter-clockwise with no collinear points. */
function convexHull(points) {
  const pts = dedupe(points);
  if (pts.length <= 2) return pts;
  const sorted = pts.slice().sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const build = (seq) => {
    const out = [];
    for (const p of seq) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
      out.push(p);
    }
    return out;
  };
  const lower = build(sorted);
  const upper = build(sorted.slice().reverse());
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function dedupe(points) {
  const seen = new Map();
  for (const p of points) seen.set(`${p[0]},${p[1]}`, p);
  return [...seen.values()].sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
}

function validatePoints(points) {
  if (!Array.isArray(points)) throw new TypeError("points must be an array");
  for (const p of points) {
    if (!Array.isArray(p) || p.length !== 2) throw new TypeError("a 2-D point is [x, y]");
    if (typeof p[0] !== "number" || typeof p[1] !== "number") throw new TypeError("coordinates must be numbers");
  }
}

function distanceToSegment(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

/** 0 when the point is inside the (convex, counter-clockwise) polygon, else the distance to it. */
function distanceToConvexPolygon(p, polygon) {
  if (polygon.length === 0) return Infinity;
  if (polygon.length === 1) return Math.hypot(p[0] - polygon[0][0], p[1] - polygon[0][1]);
  if (polygon.length === 2) return distanceToSegment(p, polygon[0], polygon[1]);
  let inside = true;
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    const cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if (cross < 0) { inside = false; break; }
  }
  if (inside) return 0;
  let best = Infinity;
  for (let i = 0; i < polygon.length; i++) {
    best = Math.min(best, distanceToSegment(p, polygon[i], polygon[(i + 1) % polygon.length]));
  }
  return best;
}

/** Exact inradius of the polygon around its interior point c (min distance from c to an edge). */
function inradius(polygon) {
  if (polygon.length < 3) return 0;
  const c = centroid(polygon);
  let best = Infinity;
  for (let i = 0; i < polygon.length; i++) {
    best = Math.min(best, distanceToSegment(c, polygon[i], polygon[(i + 1) % polygon.length]));
  }
  return best;
}

function centroid(points) {
  const sum = points.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1]], [0, 0]);
  return [sum[0] / points.length, sum[1] / points.length];
}

/** distanceToSegment(c, edge) is the perpendicular distance only when the foot lies on the edge. */
function edgeDistance(c, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len === 0) return Math.hypot(c[0] - a[0], c[1] - a[1]);
  return Math.abs((c[0] - a[0]) * dy - (c[1] - a[1]) * dx) / len;
}

/** Centred support function: max over the points of <p - c, u>. */
function support(points, u, c = [0, 0]) {
  let best = -Infinity;
  for (const p of points) best = Math.max(best, (p[0] - c[0]) * u[0] + (p[1] - c[1]) * u[1]);
  return best;
}

/**
 * Complete direction set for support-function comparisons: the unit normals of every segment
 * between two input points. Between two such directions both support functions are linear, so
 * their ratio is monotone and its extremes sit on the endpoints of the arcs they cut out.
 */
function criticalDirections(points) {
  const out = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[j][0] - points[i][0];
      const dy = points[j][1] - points[i][1];
      const len = Math.hypot(dx, dy);
      if (len === 0) continue;
      out.push([-dy / len, dx / len], [dy / len, -dx / len]);
    }
  }
  return out;
}

/**
 * Recursive thinning of one hull chain (given as indices into `hull`): keep the point farthest
 * from the chord only when it sticks out by more than `budget`; otherwise the whole chain
 * collapses onto that chord, which stays an edge of the final polygon.
 */
function simplifyChain(chainIdx, hull, budget, kept) {
  const a = hull[chainIdx[0]];
  const b = hull[chainIdx[chainIdx.length - 1]];
  let far = -1;
  let farDistance = 0;
  for (let t = 1; t < chainIdx.length - 1; t++) {
    const d = edgeDistance(hull[chainIdx[t]], a, b);
    if (d > farDistance) { farDistance = d; far = t; }
  }
  if (far < 0 || farDistance <= budget) return;
  kept.add(chainIdx[far]);
  simplifyChain(chainIdx.slice(0, far + 1), hull, budget, kept);
  simplifyChain(chainIdx.slice(far), hull, budget, kept);
}

/**
 * (1 - eps)-kernel of `points` in 2-D. Returns { kernel, hull, budget, inradius, delta } where
 * `delta` is the exact worst-case distance of a dropped hull vertex to the kernel polygon.
 */
function epsilonKernel(points, eps) {
  validatePoints(points);
  if (!(eps > 0 && eps <= 1)) throw new RangeError("eps must lie in (0, 1]");
  const hull = convexHull(points);
  if (hull.length <= 3) {
    return { kernel: hull, hull, budget: 0, radius: 0, delta: 0 };
  }
  const radius = inradius(hull);
  const budget = eps * radius;
  if (budget <= 0) return { kernel: hull, hull, budget, radius, delta: 0 }; // degenerate hull
  // Split the cycle at hull[0] and the vertex farthest from it, then thin both chains.
  let split = 1;
  let splitDistance = -1;
  for (let i = 1; i < hull.length; i++) {
    const d = Math.hypot(hull[i][0] - hull[0][0], hull[i][1] - hull[0][1]);
    if (d > splitDistance) { splitDistance = d; split = i; }
  }
  const kept = new Set([0, split]);
  const first = Array.from({ length: split + 1 }, (_, i) => i);
  const second = Array.from({ length: hull.length - split }, (_, k) => split + k).concat([0]);
  simplifyChain(first, hull, budget, kept);
  simplifyChain(second, hull, budget, kept);
  const kernel = [...kept].sort((x, y) => x - y).map((i) => hull[i]);
  let delta = 0;
  for (let i = 0; i < hull.length; i++) {
    if (!kept.has(i)) delta = Math.max(delta, distanceToConvexPolygon(hull[i], kernel));
  }
  return { kernel, hull, budget, radius, delta };
}

/** Exact diameter by brute force (n is small here). */
function diameter(points) {
  let best = 0;
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      best = Math.max(best, Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]));
    }
  }
  return best;
}

function makeRng(seed) {
  let state = seed >>> 0 || 0x2545f491;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** Uniform-ish cloud inside a disk of radius R (rejection sampling, deterministic PRNG). */
function diskCloud(rng, n, R) {
  const out = [];
  while (out.length < n) {
    const x = (rng(20001) - 10000) / 10000;
    const y = (rng(20001) - 10000) / 10000;
    if (x * x + y * y <= 1) out.push([x * R, y * R]);
  }
  return out;
}

function circlePoints(n, r) {
  return Array.from({ length: n }, (_, i) => {
    const theta = (2 * Math.PI * i) / n;
    return [r * Math.cos(theta), r * Math.sin(theta)];
  });
}

module.exports = {
  convexHull, dedupe, distanceToSegment, distanceToConvexPolygon, inradius, edgeDistance,
  support, criticalDirections, epsilonKernel, diameter,
};

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, ok, throws } = assert;
  const rng = makeRng(771771);
  let worstRatios = [];
  let gaps = [];

  const sortedKeys = (pts) => pts.map((p) => `${p[0].toFixed(6)},${p[1].toFixed(6)}`).sort();
  const cube = [[0, 0], [4, 0], [4, 3], [0, 3], [2, 1], [1, 2]];

  // -- hull on degenerate and simple inputs -----------------------------------------
  strictEqual(convexHull([]).length, 0, "empty input has an empty hull");
  strictEqual(convexHull([[1, 1]]).length, 1, "a single point is its own hull");
  strictEqual(convexHull([[0, 0], [3, 3]]).length, 2, "two points form a degenerate hull");
  strictEqual(convexHull([[0, 0], [1, 1], [2, 2], [1, 1]]).length, 2, "collinear points collapse to the extremes");
  strictEqual(convexHull([[1, 1], [1, 1], [1, 1]]).length, 1, "identical points collapse to one");
  strictEqual(convexHull(cube).length, 4, "interior points are not hull vertices");
  strictEqual(convexHull(circlePoints(12, 5)).length, 12, "a strictly convex 12-gon keeps every vertex");
  for (const p of cube) ok(distanceToConvexPolygon(p, convexHull(cube)) === 0, "every point lies inside the hull");
  ok(convexHull(circlePoints(9, 1)).some((p) => p[0] > 0.9), "hull vertices really are on the boundary");

  // -- distance primitives -----------------------------------------------------------
  strictEqual(distanceToSegment([0.5, 0], [0, 0], [1, 0]), 0, "a point on the segment has distance 0");
  strictEqual(distanceToSegment([0, 2], [0, 0], [1, 0]), 2, "perpendicular distance");
  strictEqual(Number(distanceToSegment([-1, 0], [0, 0], [1, 0]).toFixed(9)), 1, "distance is clamped to the endpoint");
  strictEqual(distanceToConvexPolygon([0.5, 0.5], [[0, 0], [1, 0], [1, 1], [0, 1]]), 0, "inside -> 0");
  strictEqual(distanceToConvexPolygon([2, 1], [[0, 0], [1, 0], [1, 1], [0, 1]]), 1, "outside -> distance to the boundary");
  strictEqual(Number(distanceToConvexPolygon([2, 2], [[0, 0], [1, 0], [1, 1], [0, 1]]).toFixed(9)),
    Number(Math.SQRT2.toFixed(9)), "a diagonal neighbour is sqrt(2) away from the corner");
  ok(inradius(circlePoints(64, 1)) > 0.99, "the inradius of a unit circle is 1");
  throws(() => epsilonKernel([[0, 0]], 0), RangeError, "eps must be positive");
  throws(() => epsilonKernel([[0, 0, 0]], 0.1), TypeError, "3-D input is rejected by the 2-D routine");

  // -- kernels on clouds: kernel inequality over the complete direction set ----------
  for (let trial = 0; trial < 25; trial++) {
    const points = diskCloud(rng, 12 + rng(30), 10 + rng(40));
    const { kernel, hull, radius, delta } = epsilonKernel(points, 0.25);
    ok(kernel.length > 0 && kernel.length <= hull.length, "the kernel is a non-empty subset of the hull");
    for (const p of kernel) ok(hull.includes(p), "every kernel vertex comes from the hull");
    strictEqual(new Set(sortedKeys(kernel)).size, kernel.length, "kernel vertices are distinct");
    strictEqual(convexHull(kernel).length, kernel.length, "the kernel is in convex position");
    ok(radius > 0 && delta <= 0.25 * radius + 1e-9,
      `delta ${delta} must stay inside the budget ${0.25 * radius}`);
    const centre = centroid(hull);
    for (const u of criticalDirections(points)) {
      const full = support(points, u, centre);
      const small = support(kernel, u, centre);
      ok(small <= full + 1e-9, "a subset cannot support more than the whole cloud");
      ok(small >= 0.75 * full - 1e-9, `kernel inequality violated: ${small} < 0.75 * ${full}`);
      ok(full - small <= delta + 1e-9, "the support gap must be bounded by the exact delta");
    const widthFull = support(points, u) + support(points, [-u[0], -u[1]]);
    const widthSmall = support(kernel, u) + support(kernel, [-u[0], -u[1]]);
      ok(widthSmall >= 0.75 * widthFull - 1e-9,
        `widths must also be (1 - eps)-preserved (${widthSmall} < 0.75 * ${widthFull})`);
      ok(widthFull >= 2 * radius - 1e-9, "the width about the centre is at least twice the inradius");
      worstRatios.push(full === 0 ? 1 : small / full);
      gaps.push(full - small);
    }
    ok(diameter(kernel) >= 0.75 * diameter(points), "the kernel preserves the diameter to 1 - eps");
  }

  // -- monotonicity in eps and the eps -> 0 limit ------------------------------------
  const cloud = diskCloud(rng, 60, 30);
  const sizes = [0.5, 0.25, 0.1, 0.05, 0.01].map((eps) => {
    const { kernel } = epsilonKernel(cloud, eps);
    ok(convexHull(kernel).length === kernel.length, "every eps must still produce a convex kernel");
    return kernel.length;
  });
  for (let i = 1; i < sizes.length; i++) {
    ok(sizes[i] >= sizes[i - 1], `kernel size must grow as eps shrinks: ${sizes.join(" > ")}`);
  }
  ok(sizes[0] < convexHull(cloud).length, "a loose eps must actually compress the hull");
  const tiny = epsilonKernel(circlePoints(24, 7), 1e-6);
  strictEqual(tiny.kernel.length, 24, "a vanishing eps must return the whole strictly convex hull");

  // -- degenerate geometry ------------------------------------------------------------
  const degenerate = epsilonKernel([[0, 0], [1, 1], [2, 2]], 0.1);
  strictEqual(degenerate.kernel.length, 2, "collinear input keeps its two extremes");
  strictEqual(degenerate.radius, 0, "a collinear hull has no inradius");
  const pointLike = epsilonKernel([[3, 3], [3, 3]], 0.4);
  strictEqual(pointLike.kernel.length, 1, "identical points give a single-vertex kernel");
  strictEqual(epsilonKernel([], 0.3).kernel.length, 0, "an empty cloud has an empty kernel");
  const square = epsilonKernel([[0, 0], [10, 0], [10, 10], [0, 10]], 0.5);
  strictEqual(square.kernel.length, 4, "a square has no vertex that a budget of eps * r can drop");

  // -- order independence and determinism -------------------------------------------
  const shuffled = cloud.slice();
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = rng(i + 1);
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  assert.deepStrictEqual(sortedKeys(epsilonKernel(shuffled, 0.1).kernel), sortedKeys(epsilonKernel(cloud, 0.1).kernel),
    "the kernel must not depend on the input order");
  assert.deepStrictEqual(epsilonKernel(cloud, 0.1), epsilonKernel(cloud, 0.1), "the kernel must be deterministic");

  const minRatio = Math.min(...worstRatios);
  const maxGap = Math.max(...gaps);
  console.log(`directions=${worstRatios.length} min support ratio=${minRatio.toFixed(4)} max support gap=${maxGap.toFixed(4)} sizes(${sizes.join(",")})`);
  console.log("PASS");
}
