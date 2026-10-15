/**
 * Task 373 - Diameter of a convex polygon by rotating calipers. (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The diameter of a point set is its farthest pair; the farthest pair of a set always
 * lies on the convex hull, so the problem reduces to the diameter of a convex polygon.
 * Rotating calipers walk an antipodal vertex j along the hull for each directed edge
 * (i, i+1): j advances while the triangle area (i, i+1, j+1) exceeds (i, i+1, j).
 * Every diameter pair is antipodal to some edge, so checking the distance from both
 * edge endpoints to j for all i finds the exact maximum.  All arithmetic is integer:
 * the result is the exact SQUARED distance; a float square root is taken only to
 * compare against the classic square = side*sqrt(2) expectation (relative tolerance
 * 1e-12, the standard double-sqrt tolerance).
 *
 * The hull is built by Andrew's monotone chain (duplicate points removed, collinear
 * points dropped).  Degenerate cases are documented: empty/one point -> diameter 0;
 * two distinct points or a fully collinear set -> hull is the two extreme points.
 * Cross-checked against the brute-force O(h^2) scan over the hull.
 *
 * Independent oracle: brute force over ALL input points (not just the hull) - the
 * farthest pair of any point set is a hull pair, so all three numbers must agree.
 */
"use strict";

class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/** Andrew's monotone chain: counter-clockwise hull without collinear points. */
function convexHull(points) {
  const key = (p) => `${p[0]},${p[1]}`;
  const unique = [...new Map(points.map((p) => [key(p), p])).values()]
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (unique.length <= 2) return unique;

  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of unique) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = unique.length - 1; i >= 0; i--) {
    const p = unique[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function distSquared(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

function area2(a, b, c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

/** Rotating calipers over a CCW convex hull; exact integer result. */
function hullDiameterSquared(hull) {
  const h = hull.length;
  if (h <= 1) return 0;
  if (h === 2) return distSquared(hull[0], hull[1]);
  let best = 0;
  let j = 1;
  for (let i = 0; i < h; i++) {
    const ni = (i + 1) % h;
    let steps = 0;
    while (steps <= h) {                          // guard against degenerate ties
      const nj = (j + 1) % h;
      if (area2(hull[i], hull[ni], hull[nj]) > area2(hull[i], hull[ni], hull[j])) {
        j = nj;
        steps++;
      } else {
        break;
      }
    }
    best = Math.max(best, distSquared(hull[i], hull[j]), distSquared(hull[ni], hull[j]));
  }
  return best;
}

function diameterSquared(points) {
  const hull = convexHull(points);
  return { diameter2: hullDiameterSquared(hull), hull };
}

/** O(h^2) oracle over the hull vertices. */
function bruteHullDiameterSquared(hull) {
  let best = 0;
  for (let i = 0; i < hull.length; i++) {
    for (let k = i + 1; k < hull.length; k++) best = Math.max(best, distSquared(hull[i], hull[k]));
  }
  return best;
}

/** O(n^2) oracle over every input point. */
function bruteAllPointsDiameterSquared(points) {
  let best = 0;
  for (let i = 0; i < points.length; i++) {
    for (let k = i + 1; k < points.length; k++) best = Math.max(best, distSquared(points[i], points[k]));
  }
  return best;
}

function lcgPoints(count, lcg, extent) {
  const points = [];
  for (let i = 0; i < count; i++) points.push([lcg.next(extent), lcg.next(extent)]);
  return points;
}

module.exports = {
  convexHull, hullDiameterSquared, diameterSquared, bruteHullDiameterSquared, bruteAllPointsDiameterSquared,
};

if (require.main === module) {
  const assert = require("node:assert");

  // classic square: side 3, diagonal 3*sqrt(2)
  const side = 3;
  const square = [[0, 0], [side, 0], [side, side], [0, side]];
  let { diameter2, hull } = diameterSquared(square);
  assert.strictEqual(diameter2, 2 * side * side, "square diameter^2 = 18");
  assert.strictEqual(diameter2, 18, "square diameter^2 literal");
  assert.strictEqual(hull.length, 4, "square hull has 4 vertices");
  assert.ok(Math.abs(Math.sqrt(diameter2) - side * Math.sqrt(2)) <= 1e-12 * side * Math.sqrt(2),
    "square diagonal is side*sqrt(2)");

  // hand-checked small cases
  assert.strictEqual(diameterSquared([[7, 7]]).diameter2, 0, "single point");
  assert.strictEqual(diameterSquared([[1, 1], [4, 5]]).diameter2, 25, "two points");
  ({ diameter2, hull } = diameterSquared([[0, 0], [4, 0], [0, 3]]));
  assert.strictEqual(hull.length, 3, "triangle hull size");
  assert.strictEqual(diameter2, 25, "3-4-5 triangle");
  assert.strictEqual(diameter2, bruteHullDiameterSquared(hull), "triangle vs oracle");
  assert.strictEqual(diameterSquared([[0, 0], [2, 0], [2, 5], [0, 5]]).diameter2, 29, "rectangle diagonal");
  ({ diameter2, hull } = diameterSquared([[0, 0], [1, 1], [3, 3]]));
  assert.strictEqual(hull.length, 2, "collinear hull degenerates");
  assert.deepStrictEqual(hull, [[0, 0], [3, 3]], "collinear hull points");
  assert.strictEqual(diameter2, 18, "collinear diameter");
  ({ diameter2, hull } = diameterSquared([[0, 0], [3, 0], [3, 3], [0, 3], [1, 1]]));
  assert.strictEqual(hull.length, 4, "interior point ignored");
  assert.strictEqual(diameter2, 18, "interior ignored diameter");

  // LCG cross-checks: calipers == brute force on the hull == brute force on all points
  const lcg = new Lcg(37313);
  for (let trial = 0; trial < 60; trial++) {
    const count = 3 + lcg.next(10);
    const points = lcgPoints(count, lcg, 50);
    hull = convexHull(points);
    const calipers = hullDiameterSquared(hull);
    assert.strictEqual(calipers, bruteHullDiameterSquared(hull), `calipers vs hull oracle ${trial}`);
    assert.strictEqual(calipers, bruteAllPointsDiameterSquared(points), `calipers vs all-points oracle ${trial}`);
  }

  // larger deterministic set for a wider sweep
  const bigPoints = lcgPoints(120, lcg, 100);
  hull = convexHull(bigPoints);
  assert.ok(hull.length >= 3, "large hull size");
  assert.strictEqual(hullDiameterSquared(hull), bruteAllPointsDiameterSquared(bigPoints), "large set diameter");
  console.log("PASS");
}
