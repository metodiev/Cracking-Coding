/** Task 368 - Point-in-polygon test (ray casting). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
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

function crossAt(o, a, b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

/** Exact test for p lying on the closed segment a-b. */
function onSegment(a, b, p) {
  if (crossAt(a, b, p) !== 0) return false;
  return Math.min(a[0], b[0]) <= p[0] && p[0] <= Math.max(a[0], b[0]) &&
    Math.min(a[1], b[1]) <= p[1] && p[1] <= Math.max(a[1], b[1]);
}

function onBoundary(polygon, p) {
  for (let i = 0; i < polygon.length; i++) {
    if (onSegment(polygon[i], polygon[(i + 1) % polygon.length], p)) return true;
  }
  return false;
}

/**
 * Even-odd ray casting with the half-open edge rule: an edge counts only when exactly
 * one endpoint has y > p.y, so horizontal edges never cross and a vertex shared by two
 * edges is counted once.  The intersection test is exact integer arithmetic: instead of
 * computing x_intersection, it compares (x1-px)*dy + (py-y1)*(x2-x1) against zero with
 * the sign of dy.
 */
function rayCastInside(polygon, p) {
  const px = p[0];
  const py = p[1];
  let inside = false;
  for (let i = 0; i < polygon.length; i++) {
    const [x1, y1] = polygon[i];
    const [x2, y2] = polygon[(i + 1) % polygon.length];
    if ((y1 > py) === (y2 > py)) continue;               // ignore horizontal edges
    const dy = y2 - y1;
    const lhs = (x1 - px) * dy + (py - y1) * (x2 - x1);
    if ((lhs > 0) === (dy > 0)) inside = !inside;
  }
  return inside;
}

/** Winding number alternative: +1 per CCW crossing above p, -1 per clockwise one. */
function windingNumber(polygon, p) {
  const px = p[0];
  const py = p[1];
  let winding = 0;
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    if (a[1] <= py) {
      if (b[1] > py && crossAt(a, b, p) > 0) winding += 1;
    } else if (b[1] <= py && crossAt(a, b, p) < 0) {
      winding -= 1;
    }
  }
  return winding;
}

/** Convex hull (CCW) used to build test polygons from LCG points. */
function monotoneChain(points) {
  const seen = new Map();
  for (const [x, y] of points) seen.set(`${x},${y}`, [x, y]);
  const pts = [...seen.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length <= 2) return pts;
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 &&
           crossAt(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 &&
           crossAt(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

module.exports = { rayCastInside, windingNumber, onBoundary, onSegment, monotoneChain, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  const square = [[0, 0], [4, 0], [4, 4], [0, 4]];
  assert.ok(!onBoundary(square, [2, 2]) && rayCastInside(square, [2, 2]), "square inside");
  assert.ok(!rayCastInside(square, [5, 2]), "square right outside");
  assert.ok(!rayCastInside(square, [-1, 2]), "square left outside");
  assert.ok(!rayCastInside(square, [2, 5]), "square above outside");
  assert.ok(onBoundary(square, [4, 4]), "square vertex boundary");
  assert.ok(onBoundary(square, [0, 2]), "square left edge boundary");
  assert.ok(onBoundary(square, [2, 0]), "square bottom edge boundary");
  assert.ok(onBoundary(square, [0, 0]), "square corner boundary");
  // (-1, 0): the horizontal ray passes straight through vertices (0,0) and (4,0);
  // the half-open rule counts both vertical edges exactly once -> outside.
  assert.ok(!rayCastInside(square, [-1, 0]) && windingNumber(square, [-1, 0]) === 0,
    "ray through two vertices");

  const triangle = [[0, 0], [6, 0], [0, 6]];
  assert.ok(!onBoundary(triangle, [1, 1]) && rayCastInside(triangle, [1, 1]), "triangle inside");
  assert.ok(onBoundary(triangle, [3, 3]), "triangle hypotenuse boundary");
  assert.ok(!rayCastInside(triangle, [5, 2]), "triangle outside");
  assert.ok(!rayCastInside(triangle, [7, 0]), "triangle right of x = 6");
  assert.ok(onBoundary(triangle, [0, 3]), "triangle left edge boundary");

  // Concave "arrow" polygon: a square with a triangular notch cut from the top.
  const arrow = [[0, 0], [6, 0], [6, 6], [3, 3], [0, 6]];
  assert.ok(!onBoundary(arrow, [3, 1]) && rayCastInside(arrow, [3, 1]), "arrow inside bottom");
  assert.ok(!rayCastInside(arrow, [3, 4]), "arrow notch outside");
  assert.ok(!onBoundary(arrow, [1, 4]) && rayCastInside(arrow, [1, 4]), "arrow inside left");
  assert.ok(!onBoundary(arrow, [5, 4]) && rayCastInside(arrow, [5, 4]), "arrow inside right");
  assert.ok(onBoundary(arrow, [3, 3]), "arrow notch vertex boundary");
  assert.ok(onBoundary(arrow, [5, 5]), "arrow on edge y = x");
  assert.ok(!rayCastInside(arrow, [2, 6]), "arrow above notch outside");
  assert.ok(onBoundary(arrow, [0, 3]), "arrow left edge boundary");
  // (1, 3): the ray y = 3 passes through the notch vertex (3, 3) and exits at x = 6;
  // the vertex contributes two crossings that cancel, then one real crossing -> inside.
  assert.ok(!onBoundary(arrow, [1, 3]), "point (1,3) not on boundary");
  assert.ok(rayCastInside(arrow, [1, 3]), "ray through notch vertex");
  assert.strictEqual(windingNumber(arrow, [1, 3]), 1);

  // Both algorithms must agree everywhere except on the boundary.
  for (const polygon of [square, triangle, arrow]) {
    for (let x = -2; x <= 9; x++) {
      for (let y = -2; y <= 9; y++) {
        const p = [x, y];
        if (onBoundary(polygon, p)) continue;            // boundary is documented as undefined
        assert.strictEqual(rayCastInside(polygon, p), windingNumber(polygon, p) !== 0,
          `agreement at ${x},${y}`);
      }
    }
  }

  // LCG convex polygons from hulls: ray casting and winding number must agree.
  const lcg = new Lcg(31415);
  for (let trial = 0; trial < 30; trial++) {
    const count = 3 + lcg.next(20);
    const points = [];
    for (let i = 0; i < count; i++) points.push([lcg.next(12), lcg.next(12)]);
    const polygon = monotoneChain(points);
    if (polygon.length < 3) continue;
    for (let x = -2; x <= 14; x++) {
      for (let y = -2; y <= 14; y++) {
        const p = [x, y];
        if (onBoundary(polygon, p)) continue;
        assert.strictEqual(rayCastInside(polygon, p), windingNumber(polygon, p) !== 0,
          `LCG agreement trial ${trial}`);
      }
    }
  }
  console.log("PASS");
}
