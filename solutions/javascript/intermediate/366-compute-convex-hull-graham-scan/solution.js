/** Task 366 - Compute convex hull (Graham scan). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
"use strict";

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/** Cross product (a - o) x (b - o); > 0 means o -> a -> b turns left. */
function cross(o, a, b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

function dist2(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

function uniqueSortedPoints(points) {
  const seen = new Map();
  for (const [x, y] of points) seen.set(`${x},${y}`, [x, y]);
  return [...seen.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

/** Convex hull, counter-clockwise, starting at the lexicographically smallest point. */
function grahamScan(points) {
  const pts = uniqueSortedPoints(points);
  if (pts.length <= 2) return pts.map((p) => p.slice());
  const pivot = pts[0];
  const rest = pts.slice(1).sort((a, b) => {
    // Equal angles keep the nearest point first; collinear points on hull edges are
    // later removed by the cross <= 0 pops.
    const turn = cross(pivot, a, b);
    if (turn > 0) return -1;
    if (turn < 0) return 1;
    return dist2(pivot, a) - dist2(pivot, b);
  });
  const hull = [pivot];
  for (const point of rest) {
    while (hull.length >= 2 &&
           cross(hull[hull.length - 2], hull[hull.length - 1], point) <= 0) {
      hull.pop();
    }
    hull.push(point);
  }
  return hull.map((p) => p.slice());
}

/** Independent gift wrapping oracle with the same conventions as grahamScan. */
function jarvisMarch(points) {
  const pts = uniqueSortedPoints(points);
  if (pts.length <= 2) return pts.map((p) => p.slice());
  const start = pts[0];
  const hull = [start];
  let current = start;
  for (;;) {
    let candidate = null;
    for (const p of pts) {
      if (p === current) continue;
      if (candidate === null) {
        candidate = p;
        continue;
      }
      const turn = cross(current, candidate, p);
      if (turn < 0) {
        candidate = p;
      } else if (turn === 0) {
        const sameDirection = (p[0] - current[0]) * (candidate[0] - current[0]) +
          (p[1] - current[1]) * (candidate[1] - current[1]) > 0;
        if (sameDirection && dist2(current, p) > dist2(current, candidate)) {
          candidate = p;                               // keep the farthest collinear point
        }
      }
    }
    if (candidate === start) break;
    hull.push(candidate);
    current = candidate;
    if (hull.length > pts.length) throw new Error("degenerate input");
  }
  return hull.map((p) => p.slice());
}

/** Twice the signed area (shoelace); positive for counter-clockwise order. */
function hullArea2(hull) {
  let total = 0;
  for (let i = 0; i < hull.length; i++) {
    const [x1, y1] = hull[i];
    const [x2, y2] = hull[(i + 1) % hull.length];
    total += x1 * y2 - x2 * y1;
  }
  return total === 0 ? 0 : total;
}

module.exports = { grahamScan, jarvisMarch, hullArea2, cross, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  const square = [[0, 0], [2, 0], [2, 2], [0, 2]];
  const squareHull = [[0, 0], [2, 0], [2, 2], [0, 2]];
  assert.deepStrictEqual(grahamScan(square), squareHull);
  assert.deepStrictEqual(jarvisMarch(square), squareHull);
  assert.strictEqual(hullArea2(grahamScan(square)), 8);

  const inside = [[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]];
  assert.deepStrictEqual(grahamScan(inside), squareHull);
  assert.deepStrictEqual(jarvisMarch(inside), squareHull);

  const collinear = [[0, 0], [1, 0], [2, 0], [3, 0]];
  assert.deepStrictEqual(grahamScan(collinear), [[0, 0], [3, 0]]);
  assert.deepStrictEqual(jarvisMarch(collinear), [[0, 0], [3, 0]]);
  const mixed = [[0, 0], [1, 1], [2, 2], [3, 3], [0, 3]];
  assert.deepStrictEqual(grahamScan(mixed), [[0, 0], [3, 3], [0, 3]]);
  assert.deepStrictEqual(jarvisMarch(mixed), grahamScan(mixed));

  const duplicates = [[0, 0], [0, 0], [2, 0], [2, 0], [2, 2], [0, 2], [2, 2]];
  assert.deepStrictEqual(grahamScan(duplicates), squareHull);
  assert.deepStrictEqual(jarvisMarch(duplicates), squareHull);

  assert.deepStrictEqual(grahamScan([[0, 0], [4, 0], [2, 3]]), [[0, 0], [4, 0], [2, 3]]);
  assert.deepStrictEqual(grahamScan([[5, 7]]), [[5, 7]]);
  assert.deepStrictEqual(grahamScan([[0, 0], [1, 1]]), [[0, 0], [1, 1]]);
  assert.deepStrictEqual(grahamScan([]), []);

  // LCG cross-check: hull size, doubled area and the exact point list vs Jarvis.
  const lcg = new Lcg(15731);
  for (let trial = 0; trial < 60; trial++) {
    const count = 1 + lcg.next(40);
    const modulus = 6 + lcg.next(20);                     // small ranges force collinearity
    const points = [];
    for (let i = 0; i < count; i++) points.push([lcg.next(modulus), lcg.next(modulus)]);
    const scan = grahamScan(points);
    const march = jarvisMarch(points);
    assert.strictEqual(scan.length, march.length, `size trial ${trial}`);
    assert.strictEqual(hullArea2(scan), hullArea2(march), `area trial ${trial}`);
    assert.deepStrictEqual(scan, march, `hull trial ${trial}`);
    if (scan.length > 2) {
      for (let i = 0; i < scan.length; i++) {
        const a = scan[(i + scan.length - 1) % scan.length];
        const b = scan[i];
        const c = scan[(i + 1) % scan.length];
        assert.ok(cross(a, b, c) > 0, `convex trial ${trial}`);
      }
    }
  }
  console.log("PASS");
}
