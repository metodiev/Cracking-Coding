/** Task 367 - Compute convex hull (Andrew monotone chain). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
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

/** Convex hull (CCW, starting at the lexicographically smallest point); collinear dropped. */
function monotoneChain(points) {
  const pts = uniqueSortedPoints(points);
  if (pts.length <= 2) return pts.map((p) => p.slice());
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 &&
           cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 &&
           cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1)).map((p) => p.slice());
}

/** The task 366 Graham scan, kept here so both hulls can be compared directly. */
function grahamScan(points) {
  const pts = uniqueSortedPoints(points);
  if (pts.length <= 2) return pts.map((p) => p.slice());
  const pivot = pts[0];
  const rest = pts.slice(1).sort((a, b) => {
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

function signedArea2(hull) {
  let total = 0;
  for (let i = 0; i < hull.length; i++) {
    const [x1, y1] = hull[i];
    const [x2, y2] = hull[(i + 1) % hull.length];
    total += x1 * y2 - x2 * y1;
  }
  return total === 0 ? 0 : total;
}

/** Counter-clockwise order starting at the lexicographically smallest point. */
function normalize(hull) {
  let pts = hull.map((p) => p.slice());
  if (pts.length <= 1) return pts;
  if (signedArea2(pts) < 0) pts = pts.reverse();
  let start = 0;
  for (let i = 1; i < pts.length; i++) {
    if (pts[i][0] < pts[start][0] || (pts[i][0] === pts[start][0] && pts[i][1] < pts[start][1])) {
      start = i;
    }
  }
  return pts.slice(start).concat(pts.slice(0, start));
}

module.exports = { monotoneChain, grahamScan, normalize, signedArea2, cross, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  // Same hand-checked sets as task 366; both algorithms must return the same hull.
  const cases = [
    ["square", [[0, 0], [2, 0], [2, 2], [0, 2]], [[0, 0], [2, 0], [2, 2], [0, 2]]],
    ["inside", [[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]], [[0, 0], [2, 0], [2, 2], [0, 2]]],
    ["collinear", [[0, 0], [1, 0], [2, 0], [3, 0]], [[0, 0], [3, 0]]],
    ["mixed", [[0, 0], [1, 1], [2, 2], [3, 3], [0, 3]], [[0, 0], [3, 3], [0, 3]]],
    ["duplicates", [[0, 0], [0, 0], [2, 0], [2, 0], [2, 2], [0, 2], [2, 2]],
      [[0, 0], [2, 0], [2, 2], [0, 2]]],
    ["triangle", [[0, 0], [4, 0], [2, 3]], [[0, 0], [4, 0], [2, 3]]],
  ];
  for (const [name, points, expected] of cases) {
    assert.deepStrictEqual(monotoneChain(points), expected, `${name} monotone chain`);
    assert.deepStrictEqual(grahamScan(points), expected, `${name} graham scan`);
    assert.deepStrictEqual(normalize(monotoneChain(points)), normalize(grahamScan(points)), name);
  }
  assert.deepStrictEqual(normalize(monotoneChain([[5, 7]])), normalize(grahamScan([[5, 7]])));
  assert.deepStrictEqual(normalize(monotoneChain([[0, 0], [1, 1]])),
    normalize(grahamScan([[0, 0], [1, 1]])));
  assert.deepStrictEqual(monotoneChain([]), []);
  assert.strictEqual(signedArea2(monotoneChain([[0, 0], [2, 0], [2, 2], [0, 2]])), 8);

  // Normalization really canonicalizes: reversed clockwise input flips back.
  const clockwise = [[0, 0], [0, 2], [2, 2], [2, 0]];
  const canonical = [[0, 0], [2, 0], [2, 2], [0, 2]];
  assert.ok(signedArea2(clockwise) < 0);
  assert.deepStrictEqual(normalize(clockwise), canonical);
  assert.deepStrictEqual(normalize([[2, 2], [0, 2], [0, 0], [2, 0]]), canonical);

  // LCG cross-check: monotone chain and Graham scan agree after normalization.
  const lcg = new Lcg(90210);
  for (let trial = 0; trial < 80; trial++) {
    const count = 1 + lcg.next(45);
    const modulus = 5 + lcg.next(25);                     // small ranges force collinearity
    const points = [];
    for (let i = 0; i < count; i++) points.push([lcg.next(modulus), lcg.next(modulus)]);
    const chain = normalize(monotoneChain(points));
    const scan = normalize(grahamScan(points));
    assert.deepStrictEqual(chain, scan, `trial ${trial}`);
    assert.ok(signedArea2(chain) >= 0, `CCW trial ${trial}`);
    if (chain.length > 2) {
      for (let i = 0; i < chain.length; i++) {
        const a = chain[(i + chain.length - 1) % chain.length];
        const b = chain[i];
        const c = chain[(i + 1) % chain.length];
        assert.ok(cross(a, b, c) > 0, `convex trial ${trial}`);
      }
    }
  }
  console.log("PASS");
}
