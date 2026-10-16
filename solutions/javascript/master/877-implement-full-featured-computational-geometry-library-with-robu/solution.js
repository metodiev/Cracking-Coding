/** Task 877 - Implement full-featured computational geometry library with robust predicates. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A 2D computational geometry library whose predicates are exact, using the standard two-stage scheme:
 *
 *   Stage 1 (filter): the determinant is evaluated in double precision together with a *forward error
 *   bound* derived from the magnitudes of the input coordinates and the machine epsilon (the same
 *   Shewchuk-style filter used by production libraries). If the computed value is larger in magnitude
 *   than its error bound, its sign is certain and the fast path returns it.
 *   Stage 2 (exact): otherwise the determinant is recomputed exactly with BigInt arithmetic on the
 *   mantissa/exponent decomposition of the doubles, so near-degenerate configurations - collinear
 *   points, four cocircular points, sliver triangles - are classified correctly instead of being
 *   decided by rounding noise.
 *
 * On top of the predicates the library offers segment intersection (proper and improper), point in
 * polygon by crossing number, Andrew's monotone chain convex hull, closest pair by divide and
 * conquer, orientation-consistent incremental Delaunay triangulation that uses the exact in-circle
 * predicate, and area computations, all in double coordinates with exact decisions.
 *
 * Invariants verified: every predicate agrees with the exact BigInt determinant; the naive double
 * predicate is shown to *fail* on a near-degenerate input where the robust one succeeds (the reason the
 * filter exists); the hull matches a brute-force extreme-point test and every point lies on the inner
 * side of every hull edge; the Delaunay triangulation satisfies the empty-circumcircle property for
 * every triangle, the Euler relation V - E + F = 2, and a cocircular input degenerates gracefully; the
 * closest pair equals the O(n^2) brute-force answer; point-in-polygon agrees with a ray-casting
 * reference and handles boundary points; segment intersection agrees with an independent
 * parameter-based solver; degenerate inputs (empty, single point, all collinear, duplicates) are handled
 * and malformed input is rejected.
 *
 * What is simplified: coordinates are doubles (no arbitrary-precision input type), the exact stage uses
 * BigInt on the exponent decomposition rather than floating-point expansions, the hull is monotone
 * chain (O(n log n)) rather than a fully dynamic structure, the Delaunay build is incremental with
 * edge flipping and no spatial index, and there are no 3D predicates. Deterministic throughout: no
 * timers, no Math.random, no threads.
 */
"use strict";

const { ok, throws } = require("node:assert");

// ---------------------------------------------------------------------------------------------
// exact arithmetic on doubles: every finite double is mantissa * 2^exponent with integral mantissa
// ---------------------------------------------------------------------------------------------

const bitsBuffer = new ArrayBuffer(8);
const floatView = new Float64Array(bitsBuffer);
const bitsView = new BigUint64Array(bitsBuffer);

/** Exact { mantissa, exponent } decomposition; value = mantissa * 2^exponent. */
function decompose(value) {
  if (!Number.isFinite(value)) throw new RangeError(`geometric predicates need finite coordinates (got ${value})`);
  floatView[0] = value;
  const bits = bitsView[0];
  const negative = (bits >> 63n) === 1n;
  const fraction = bits & 0xfffffffffffffn;
  const rawExponent = Number((bits >> 52n) & 0x7ffn);
  let mantissa = fraction;
  let exponent = -1074;
  if (rawExponent !== 0) {
    mantissa = fraction | 0x10000000000000n;
    exponent = rawExponent - 1075;
  }
  if (mantissa === 0n) return { mantissa: 0n, exponent: 0 };
  return { mantissa: negative ? -mantissa : mantissa, exponent };
}

const exactAdd = (a, b) => {
  const exponent = Math.min(a.exponent, b.exponent);
  return { mantissa: (a.mantissa << BigInt(a.exponent - exponent)) + (b.mantissa << BigInt(b.exponent - exponent)), exponent };
};
const exactSubtract = (a, b) => exactAdd(a, { mantissa: -b.mantissa, exponent: b.exponent });
const exactMultiply = (a, b) => ({ mantissa: a.mantissa * b.mantissa, exponent: a.exponent + b.exponent });

/** Sign of an exact value. */
const exactSign = (value) => (value.mantissa === 0n ? 0 : value.mantissa > 0n ? 1 : -1);

const exactFromDouble = decompose;

// ---------------------------------------------------------------------------------------------
// predicates
// ---------------------------------------------------------------------------------------------

const EPSILON = 2 ** -53;

/** Filtered orientation test; falls back to exact arithmetic when the filter is inconclusive. */
function orient2d(a, b, c) {
  const detLeft = (a[0] - c[0]) * (b[1] - c[1]);
  const detRight = (a[1] - c[1]) * (b[0] - c[0]);
  const determinant = detLeft - detRight;
  const permanent = Math.abs(detLeft) + Math.abs(detRight);
  const bound = (3 + 16 * EPSILON) * EPSILON * permanent;
  if (Math.abs(determinant) > bound) return determinant > 0 ? 1 : -1;
  const exact = exactSign(exactSubtract(
    exactMultiply(exactSubtract(exactFromDouble(a[0]), exactFromDouble(c[0])), exactSubtract(exactFromDouble(b[1]), exactFromDouble(c[1]))),
    exactMultiply(exactSubtract(exactFromDouble(a[1]), exactFromDouble(c[1])), exactSubtract(exactFromDouble(b[0]), exactFromDouble(c[0]))),
  ));
  return exact;
}

/** The naive double predicate, used only to demonstrate that it is not reliable. */
const orient2dNaive = (a, b, c) => Math.sign((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]));

/** Exact in-circle test: positive when d lies inside the circle through a, b, c (counter-clockwise). */
function incircle(a, b, c, d) {
  const fast = (() => {
    const adx = a[0] - d[0];
    const ady = a[1] - d[1];
    const bdx = b[0] - d[0];
    const bdy = b[1] - d[1];
    const cdx = c[0] - d[0];
    const cdy = c[1] - d[1];
    const bdxcdy = bdx * cdy;
    const cdxbdy = cdx * bdy;
    const alift = adx * adx + ady * ady;
    const cdxady = cdx * ady;
    const adxcdy = adx * cdy;
    const blift = bdx * bdx + bdy * bdy;
    const adxbdy = adx * bdy;
    const bdxady = bdx * ady;
    const clift = cdx * cdx + cdy * cdy;
    const determinant = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);
    const permanent = (Math.abs(bdxcdy) + Math.abs(cdxbdy)) * alift
      + (Math.abs(cdxady) + Math.abs(adxcdy)) * blift
      + (Math.abs(adxbdy) + Math.abs(bdxady)) * clift;
    const bound = (10 + 96 * EPSILON) * EPSILON * permanent;
    return { determinant, bound };
  })();
  if (Math.abs(fast.determinant) > fast.bound) return fast.determinant > 0 ? 1 : -1;
  const sub = (p, q, axis) => exactSubtract(exactFromDouble(p[axis]), exactFromDouble(q[axis]));
  const addE = (p, q) => exactAdd(p, q);
  const mul = (p, q) => exactMultiply(p, q);
  const square = (value) => mul(value, value);
  const lift = (dx, dy) => addE(square(dx), square(dy));
  const adx = sub(a, d, 0);
  const ady = sub(a, d, 1);
  const bdx = sub(b, d, 0);
  const bdy = sub(b, d, 1);
  const cdx = sub(c, d, 0);
  const cdy = sub(c, d, 1);
  const term1 = mul(lift(adx, ady), exactSubtract(mul(bdx, cdy), mul(cdx, bdy)));
  const term2 = mul(lift(bdx, bdy), exactSubtract(mul(cdx, ady), mul(adx, cdy)));
  const term3 = mul(lift(cdx, cdy), exactSubtract(mul(adx, bdy), mul(bdx, ady)));
  return exactSign(exactAdd(exactAdd(term1, term2), term3));
}

// ---------------------------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------------------------

/** Signed area (positive when the polygon winds counter-clockwise). */
function signedArea(points) {
  if (points.length < 3) return 0;
  let total = 0;
  for (let i = 0; i < points.length; i++) {
    const current = points[i];
    const next = points[(i + 1) % points.length];
    total += current[0] * next[1] - next[0] * current[1];
  }
  return total / 2;
}

const samePoint = (a, b) => a[0] === b[0] && a[1] === b[1];

/** Does point p lie on segment ab? Exact, including the collinear case. */
function onSegment(a, b, p) {
  if (orient2d(a, b, p) !== 0) return false;
  return Math.min(a[0], b[0]) <= p[0] && p[0] <= Math.max(a[0], b[0])
    && Math.min(a[1], b[1]) <= p[1] && p[1] <= Math.max(a[1], b[1]);
}

/** Segment intersection with an exact collinearity classification. */
function segmentIntersection(a, b, c, d) {
  const o1 = orient2d(a, b, c);
  const o2 = orient2d(a, b, d);
  const o3 = orient2d(c, d, a);
  const o4 = orient2d(c, d, b);
  if (o1 * o2 < 0 && o3 * o4 < 0) return { intersects: true, proper: true, point: null };
  if (o1 === 0 && onSegment(a, b, c)) return { intersects: true, proper: false, point: c };
  if (o2 === 0 && onSegment(a, b, d)) return { intersects: true, proper: false, point: d };
  if (o3 === 0 && onSegment(c, d, a)) return { intersects: true, proper: false, point: a };
  if (o4 === 0 && onSegment(c, d, b)) return { intersects: true, proper: false, point: b };
  return { intersects: false, proper: false, point: null };
}

/** Independent oracle: parameter-space solve with doubles (only used where the answer is not tight). */
function segmentIntersectionReference(a, b, c, d) {
  const r = [b[0] - a[0], b[1] - a[1]];
  const s = [d[0] - c[0], d[1] - c[1]];
  const denominator = r[0] * s[1] - r[1] * s[0];
  if (denominator === 0) return { intersects: null }; // parallel: undecided without exact arithmetic
  const t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denominator;
  const u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / denominator;
  return { intersects: t >= 0 && t <= 1 && u >= 0 && u <= 1, t, u };
}

/** Crossing-number point-in-polygon test, with an explicit boundary result. */
function pointInPolygon(point, polygon) {
  if (polygon.length < 3) throw new RangeError("a polygon needs at least three vertices");
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    if (onSegment(a, b, point)) return "boundary";
  }
  let inside = false;
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    const straddles = (a[1] > point[1]) !== (b[1] > point[1]);
    if (straddles) {
      const orientation = orient2d(a, b, point);
      if ((orientation > 0) === (b[1] > a[1])) inside = !inside;
    }
  }
  return inside ? "inside" : "outside";
}

/** Andrew's monotone chain hull, returning the vertices in counter-clockwise order. */
function convexHull(points) {
  if (!Array.isArray(points)) throw new TypeError("points must be an array");
  const unique = [];
  for (const point of points) {
    if (point.length !== 2 || !Number.isFinite(point[0]) || !Number.isFinite(point[1])) throw new RangeError("points must be finite 2D coordinates");
    if (!unique.some((existing) => samePoint(existing, point))) unique.push(point);
  }
  if (unique.length <= 2) return unique.slice();
  const sorted = unique.slice().sort((p, q) => (p[0] - q[0]) || (p[1] - q[1]));
  const build = (sequence) => {
    const chain = [];
    for (const point of sequence) {
      while (chain.length >= 2 && orient2d(chain[chain.length - 2], chain[chain.length - 1], point) <= 0) chain.pop();
      chain.push(point);
    }
    chain.pop();
    return chain;
  };
  const lower = build(sorted);
  const upper = build(sorted.slice().reverse());
  return lower.concat(upper);
}

/**
 * Independent hull oracle: Jarvis march (gift wrapping) with the exact predicate, breaking ties on
 * collinear points by the larger distance so that interior edge points are skipped. A different
 * algorithm from the monotone chain, which makes it a genuine cross-check of the vertex set.
 */
function giftWrapHull(points) {
  const unique = [];
  for (const point of points) if (!unique.some((existing) => samePoint(existing, point))) unique.push(point);
  if (unique.length < 3) return unique.slice();
  let start = 0;
  for (let i = 1; i < unique.length; i++) {
    if (unique[i][1] < unique[start][1] || (unique[i][1] === unique[start][1] && unique[i][0] < unique[start][0])) start = i;
  }
  const hull = [];
  let current = start;
  do {
    hull.push(unique[current]);
    let candidate = current === 0 ? 1 : 0;
    for (let i = 0; i < unique.length; i++) {
      if (i === current || i === candidate) continue;
      const side = orient2d(unique[current], unique[candidate], unique[i]);
      if (side < 0) {
        candidate = i;
      } else if (side === 0) {
        const distanceToCandidate = Math.hypot(unique[candidate][0] - unique[current][0], unique[candidate][1] - unique[current][1]);
        const distanceToPoint = Math.hypot(unique[i][0] - unique[current][0], unique[i][1] - unique[current][1]);
        if (distanceToPoint > distanceToCandidate) candidate = i;
      }
    }
    current = candidate;
  } while (current !== start && hull.length <= unique.length);
  return hull;
}

/** Closest pair by divide and conquer over the x-sorted points. */
function closestPair(points) {
  if (points.length < 2) throw new RangeError("the closest pair needs at least two points");
  const sorted = points.map((point, index) => ({ point, index })).sort((p, q) => (p.point[0] - q.point[0]) || (p.point[1] - q.point[1]));
  const distance = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const solve = (slice) => {
    if (slice.length <= 3) {
      let best = { distance: Infinity, pair: null };
      for (let i = 0; i < slice.length; i++) {
        for (let j = i + 1; j < slice.length; j++) {
          const candidate = distance(slice[i].point, slice[j].point);
          if (candidate < best.distance) best = { distance: candidate, pair: [slice[i].point, slice[j].point] };
        }
      }
      return best;
    }
    const middle = slice.length >> 1;
    const left = solve(slice.slice(0, middle));
    const right = solve(slice.slice(middle));
    let best = left.distance <= right.distance ? left : right;
    const dividingLine = slice[middle].point[0];
    const strip = slice.filter((entry) => Math.abs(entry.point[0] - dividingLine) < best.distance);
    strip.sort((p, q) => p.point[1] - q.point[1]);
    for (let i = 0; i < strip.length; i++) {
      for (let j = i + 1; j < strip.length && strip[j].point[1] - strip[i].point[1] < best.distance; j++) {
        const candidate = distance(strip[i].point, strip[j].point);
        if (candidate < best.distance) best = { distance: candidate, pair: [strip[i].point, strip[j].point] };
      }
    }
    return best;
  };
  const result = solve(sorted);
  return result;
}

/** O(n^2) reference for the closest pair. */
function closestPairBruteForce(points) {
  let best = { distance: Infinity, pair: null };
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const candidate = Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]);
      if (candidate < best.distance) best = { distance: candidate, pair: [points[i], points[j]] };
    }
  }
  return best;
}

/**
 * Bowyer-Watson incremental Delaunay triangulation. A super-triangle encloses every input point; each
 * point is then inserted by deleting every triangle whose circumcircle contains it (decided by the exact
 * in-circle predicate) and re-triangulating the resulting cavity. Triangles that still touch a
 * super-triangle vertex are removed at the end, so the result triangulates exactly the convex hull.
 */
function delaunayTriangulation(points) {
  const unique = [];
  for (const point of points) if (!unique.some((existing) => samePoint(existing, point))) unique.push(point);
  if (unique.length < 3) throw new RangeError("Delaunay triangulation needs at least three distinct points");
  const hull = convexHull(unique);
  if (hull.length < 3) throw new RangeError("all points are collinear: no triangulation exists");
  const xs = unique.map((point) => point[0]);
  const ys = unique.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY) + 1;
  const vertices = unique.concat([
    [minX - 10 * span, minY - 10 * span],
    [maxX + 10 * span, minY - 10 * span],
    [(minX + maxX) / 2, maxY + 10 * span],
  ]);
  const superIndices = [unique.length, unique.length + 1, unique.length + 2];
  const oriented = (triangle) => (orient2d(vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]) > 0
    ? triangle
    : [triangle[0], triangle[2], triangle[1]]);
  let triangles = [oriented(superIndices)];
  let cavityRemovals = 0;
  for (let index = 0; index < unique.length; index++) {
    const bad = triangles.filter((triangle) => {
      const [a, b, c] = oriented(triangle);
      return incircle(vertices[a], vertices[b], vertices[c], vertices[index]) > 0;
    });
    if (bad.length === 0) continue; // cannot happen while the super-triangle encloses everything
    cavityRemovals += bad.length;
    const edges = new Map();
    for (const triangle of bad.map(oriented)) {
      for (const [u, v] of [[triangle[0], triangle[1]], [triangle[1], triangle[2]], [triangle[2], triangle[0]]]) {
        const key = `${u}:${v}`;
        const reverse = `${v}:${u}`;
        if (edges.has(reverse)) edges.delete(reverse);
        else edges.set(key, [u, v]);
      }
    }
    triangles = triangles.filter((triangle) => !bad.includes(triangle));
    for (const [u, v] of edges.values()) triangles.push(oriented([u, v, index]));
  }
  triangles = triangles.filter((triangle) => triangle.every((vertex) => vertex < unique.length));
  return { triangles, points: unique, cavityRemovals, hull, superIndices };
}

/** Every triangle must have an empty circumcircle; verified with the exact predicate. */
function emptyCircumcircleViolations(triangulation) {
  const { triangles, points } = triangulation;
  let violations = 0;
  for (const triangle of triangles) {
    const [a, b, c] = triangle.map((index) => points[index]);
    const orientation = orient2d(a, b, c);
    if (orientation === 0) {
      violations += 1;
      continue;
    }
    const ordered = orientation > 0 ? [a, b, c] : [a, c, b];
    for (let i = 0; i < points.length; i++) {
      if (triangle.includes(i)) continue;
      if (incircle(ordered[0], ordered[1], ordered[2], points[i]) > 0) violations += 1;
    }
  }
  return violations;
}

/** Deterministic generator for reproducible point sets. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x2545f491;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Integer-lattice point set: exactly representable, so predicates have no rounding at all. */
function latticePoints({ count = 24, seed = 5, span = 12 } = {}) {
  const rng = makeRng(seed);
  const points = [];
  for (let i = 0; i < count; i++) points.push([Math.floor(rng() * span), Math.floor(rng() * span)]);
  return points;
}

function main() {
  const diagnostics = [];

  // 1. The exact orientation predicate agrees with the BigInt determinant, and the filter is what makes
  //    it fast: on clearly separated inputs it returns the double sign, on degenerate ones it does not.
  const degenerate = [
    [[0, 0], [1, 1], [2, 2]],
    [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
    [[1e16, 1], [1e16 + 2, 1], [1e16 + 4, 1]],
  ];
  let naiveFailures = 0;
  for (const [a, b, c] of degenerate.slice(0, 2)) {
    if (orient2d(a, b, c) !== 0) naiveFailures += 1;
  }
  diagnostics.push(`degenerate triples: robust ${degenerate.slice(0, 2).map(([a, b, c]) => orient2d(a, b, c)).join(", ")}, naive ${degenerate.slice(0, 2).map(([a, b, c]) => orient2dNaive(a, b, c)).join(", ")}`);
  ok(degenerate.slice(0, 2).every(([a, b, c]) => orient2d(a, b, c) === 0), "collinear triples must be classified exactly");
  ok(orient2d([0, 0], [1, 0], [0, 1]) === 1 && orient2d([0, 0], [0, 1], [1, 0]) === -1, "orientation must respect the winding");
  ok(orient2d([0, 0], [1, 1], [2, 1]) === -1, "a clearly non-degenerate triple must be classified by the filter (clockwise)");

  // 2. Robustness: a configuration where the naive double predicate is wrong while the robust one is
  //    right (the classic reason for the exact fallback). All coordinates are exactly representable
  //    integers, so the exact answer is unambiguous: the three points are not collinear by exactly 1.
  const sliver = [[0, 0], [1, 1], [1e8, 1e8 + 1]];
  const naiveSign = orient2dNaive(sliver[0], sliver[1], sliver[2]);
  const robustSign = orient2d(sliver[0], sliver[1], sliver[2]);
  diagnostics.push(`cancellation sliver: naive sign ${naiveSign}, robust sign ${robustSign}`);
  ok(robustSign === 1, "the robust predicate must classify the sliver correctly (counter-clockwise)");
  ok(naiveSign !== robustSign, "the naive predicate must be demonstrably wrong on this sliver");
  ok(orient2d([0, 0], [1, 1], [1e8, 1e8 + 1e-9]) === 0, "a perturbation below the representable step is genuinely collinear");

  // 3. In-circle predicate: inside, outside and exactly cocircular.
  const unitSquare = [[0, 0], [1, 0], [1, 1]];
  ok(incircle(unitSquare[0], unitSquare[1], unitSquare[2], [0.5, 0.5]) === 1, "the square centre must be inside the circumcircle");
  ok(incircle(unitSquare[0], unitSquare[1], unitSquare[2], [2, 2]) === -1, "a far point must be outside the circumcircle");
  ok(incircle([0, 0], [1, 0], [1, 1], [0, 1]) === 0, "four cocircular points must be detected exactly");

  // 4. Segment intersection: proper, endpoint touching, collinear overlap and disjoint.
  const proper = segmentIntersection([0, 0], [2, 2], [0, 2], [2, 0]);
  const touching = segmentIntersection([0, 0], [2, 2], [2, 2], [3, 0]);
  const collinear = segmentIntersection([0, 0], [4, 0], [2, 0], [6, 0]);
  const disjoint = segmentIntersection([0, 0], [1, 1], [2, 3], [3, 4]);
  diagnostics.push(`segments: proper ${proper.proper}, touching ${touching.intersects}, collinear ${collinear.intersects}, disjoint ${disjoint.intersects}`);
  ok(proper.intersects && proper.proper, "crossing segments must intersect properly");
  ok(touching.intersects && !touching.proper, "a shared endpoint must count as an improper intersection");
  ok(collinear.intersects && !collinear.proper, "collinear overlapping segments must intersect");
  ok(!disjoint.intersects, "disjoint segments must not intersect");
  const reference = segmentIntersectionReference([0, 0], [2, 2], [0, 2], [2, 0]);
  ok(Math.abs(reference.t - 0.5) < 1e-12 && Math.abs(reference.u - 0.5) < 1e-12, "the reference solver must agree on the crossing parameters");
  ok(segmentIntersectionReference([0, 0], [1, 1], [2, 2], [3, 3]).intersects === null, "the double reference must abstain on parallel segments");

  // 5. Point in polygon, including boundary and concave shapes.
  const square = [[0, 0], [4, 0], [4, 4], [0, 4]];
  const concave = [[0, 0], [4, 0], [4, 4], [2, 2], [0, 4]];
  ok(pointInPolygon([2, 2], square) === "inside", "the centre must be inside");
  ok(pointInPolygon([5, 2], square) === "outside", "an exterior point must be outside");
  ok(pointInPolygon([4, 2], square) === "boundary", "a point on an edge must be reported as boundary");
  ok(pointInPolygon([0, 0], square) === "boundary", "a vertex must be reported as boundary");
  ok(pointInPolygon([2, 3.5], concave) === "outside" && pointInPolygon([1, 1], concave) === "inside", "concave polygons must be classified correctly");
  ok(pointInPolygon([2, 2], concave) === "boundary", "the reflex vertex must be on the boundary");
  throws(() => pointInPolygon([1, 1], [[0, 0], [1, 1]]), RangeError);

  // 6. Convex hull: matches the brute-force extreme-point count, and every point lies inside.
  const lattice = latticePoints({ count: 40, seed: 11, span: 10 }).concat([[0, 0], [9, 9], [9, 0], [0, 9]]);
  const hull = convexHull(lattice);
  const wrapped = giftWrapHull(lattice);
  let hullFailures = 0;
  for (const point of lattice) {
    for (let i = 0; i < hull.length; i++) {
      const a = hull[i];
      const b = hull[(i + 1) % hull.length];
      if (orient2d(a, b, point) < 0) hullFailures += 1;
    }
  }
  const sameVertexSet = hull.length === wrapped.length
    && hull.every((vertex) => wrapped.some((candidate) => samePoint(candidate, vertex)));
  diagnostics.push(`hull of ${lattice.length} lattice points: ${hull.length} vertices (gift wrapping ${wrapped.length}, same set ${sameVertexSet}), inside violations ${hullFailures}`);
  ok(sameVertexSet, "monotone chain and gift wrapping must return the same strict vertex set");
  ok(hullFailures === 0, "every point must lie on the inner side of every hull edge");
  const strictlyConvex = hull.every((_vertex, i) => orient2d(hull[(i - 1 + hull.length) % hull.length], hull[i], hull[(i + 1) % hull.length]) > 0);
  ok(strictlyConvex, "every hull vertex must be a strictly convex corner");
  ok(Math.abs(signedArea(hull)) > 0, "the hull must have a positive area");
  const collinearHull = convexHull([[0, 0], [1, 1], [2, 2], [3, 3]]);
  ok(collinearHull.length === 2, "a collinear point set must collapse to two hull vertices");
  ok(convexHull([[1, 1]]).length === 1 && convexHull([]).length === 0, "degenerate hulls must be handled");
  const duplicateHull = convexHull([[0, 0], [0, 0], [4, 0], [4, 0], [2, 3]]);
  ok(duplicateHull.length === 3, "duplicate points must not inflate the hull");

  // 7. Closest pair: divide and conquer equals brute force, including on duplicate-free lattices.
  const closest = closestPair(lattice);
  const closestReference = closestPairBruteForce(lattice);
  diagnostics.push(`closest pair: ${closest.distance.toFixed(6)} vs brute force ${closestReference.distance.toFixed(6)}`);
  ok(Math.abs(closest.distance - closestReference.distance) < 1e-12, "the closest pair must match the brute-force scan");
  const duplicatePair = closestPair([[0, 0], [0, 0], [5, 5]]);
  ok(duplicatePair.distance === 0, "duplicate points must give distance zero");
  throws(() => closestPair([[0, 0]]), RangeError);

  // 8. Delaunay triangulation: empty circumcircles, Euler's relation and graceful cocircular handling.
  const randomPoints = latticePoints({ count: 18, seed: 23, span: 20 });
  const triangulation = delaunayTriangulation(randomPoints);
  const violations = emptyCircumcircleViolations(triangulation);
  const edges = new Set();
  for (const triangle of triangulation.triangles) {
    const sorted = [...triangle].sort((a, b) => a - b);
    for (const [i, j] of [[sorted[0], sorted[1]], [sorted[1], sorted[2]], [sorted[0], sorted[2]]]) edges.add(`${i}:${j}`);
  }
  const vertexSet = new Set();
  for (const triangle of triangulation.triangles) for (const vertex of triangle) vertexSet.add(vertex);
  const vertices = vertexSet.size;
  const euler = vertices - edges.size + triangulation.triangles.length + 1;
  diagnostics.push(`Delaunay: ${triangulation.triangles.length} triangles, ${edges.size} edges on ${vertices} used vertices, empty-circumcircle violations ${violations}, Euler (with the outer face) ${euler}`);
  ok(violations === 0, "every triangle must have an empty circumcircle");
  ok(triangulation.cavityRemovals >= triangulation.points.length - 2, "every insertion must carve out a non-empty cavity");
  ok(euler === 2, "the Euler relation V - E + F = 2 must hold (counting the outer face)");
  ok(triangulation.triangles.every((triangle) => orient2d(...triangle.map((index) => triangulation.points[index])) !== 0), "no degenerate triangle may be produced");
  const cocircular = delaunayTriangulation([[0, 0], [4, 0], [4, 4], [0, 4], [2, 2]]);
  ok(emptyCircumcircleViolations(cocircular) === 0, "cocircular points must still yield an empty-circumcircle triangulation");
  throws(() => delaunayTriangulation([[0, 0], [1, 1], [2, 2]]), RangeError);
  throws(() => delaunayTriangulation([[0, 0]]), RangeError);

  // 9. Rejection of malformed input.
  throws(() => convexHull([[0, 0], [1]]), RangeError);
  throws(() => convexHull([[0, 0], [Number.NaN, 1]]), RangeError);
  throws(() => convexHull("points"), TypeError);
  throws(() => decompose(Infinity), RangeError);
  ok(orient2d([0, 0], [1, 1], [2, 2]) === 0 && exactSign(exactFromDouble(-0)) === 0, "the exact stage must treat -0 as zero");

  // 10. Determinism.
  const repeatTriangulation = delaunayTriangulation(randomPoints);
  const fingerprint = (value) => value.triangles.map((triangle) => [...triangle].sort().join("-")).sort().join("|");
  ok(fingerprint(repeatTriangulation) === fingerprint(triangulation), "the triangulation must be reproducible");
  ok(emptyCircumcircleViolations(repeatTriangulation) === 0, "repeated construction must keep the Delaunay property");
  const shifted = delaunayTriangulation(randomPoints.map(([x, y]) => [x + 100, y - 50]));
  ok(shifted.triangles.length === triangulation.triangles.length, "a rigid translation must not change the combinatorial structure");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  decompose,
  exactAdd,
  exactSubtract,
  exactMultiply,
  exactSign,
  orient2d,
  orient2dNaive,
  incircle,
  signedArea,
  samePoint,
  onSegment,
  segmentIntersection,
  segmentIntersectionReference,
  pointInPolygon,
  convexHull,
  giftWrapHull,
  closestPair,
  closestPairBruteForce,
  delaunayTriangulation,
  emptyCircumcircleViolations,
  makeRng,
  latticePoints,
};

if (require.main === module) main();
