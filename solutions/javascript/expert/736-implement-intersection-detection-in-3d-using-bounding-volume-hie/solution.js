/** Task 736 - Implement intersection detection in 3D using bounding volume hierarchies.
 *  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Layer 1, the hierarchy: a static AABB tree. A node stores the union box of its triangles and
 * is split at the median centroid along the longest axis of the centroid bounds, so it is
 * balanced by construction and every children box is a subset of its parent box. Self-collision
 * is the paired descent: each internal node is visited against itself through the three
 * disjoint sub-pairs (left,left), (left,right), (right,right); a node pair is abandoned as soon
 * as the boxes are disjoint, so only pairs of boxes that really touch reach the primitive.
 *
 * Layer 2, the primitive: Moeller's interval test. Reject when one triangle is strictly on one
 * side of the other's plane; otherwise the two crossing segments both lie on the line
 * plane1 ^ plane2, whose dominant coordinate direction makes the projection injective, so the
 * two projected intervals decide the answer. Triangles that meet exactly on a boundary make the
 * floating-point intervals ambiguous; those pairs (|gap| below a relative tolerance) are
 * re-decided with exact BigInt rational comparisons, and zero-area triangles are reduced to
 * their longest edge before the segment routines see them.
 *
 * Verified: an independent exact oracle - the triangles meet iff the origin lies in the convex
 * hull of the nine pairwise differences, decided by Caratheodory over affinely independent
 * subsets with BigInt determinants - agrees with the float primitive on hundreds of random
 * small-integer triangles; the hierarchy's pair set equals brute force on random clouds and on
 * separated clusters, node box/leaf invariants hold, and the primitive-call counters show the
 * pruning.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

// ---- float vector helpers --------------------------------------------------------------
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const isZeroVec = (v) => v[0] === 0 && v[1] === 0 && v[2] === 0;

/** Coordinate axis to drop when projecting: the largest |component| is injective on the line. */
function dominantAxis(v) {
  const abs = v.map(Math.abs);
  if (abs[0] >= abs[1] && abs[0] >= abs[2]) return 0;
  return abs[1] >= abs[2] ? 1 : 2;
}

function project2(point, drop) {
  const out = [];
  for (let axis = 0; axis < 3; axis++) if (axis !== drop) out.push(point[axis]);
  return out;
}

// ---- axis aligned boxes ----------------------------------------------------------------
function triBounds(tri) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const p of tri) {
    for (let a = 0; a < 3; a++) {
      if (p[a] < lo[a]) lo[a] = p[a];
      if (p[a] > hi[a]) hi[a] = p[a];
    }
  }
  return { lo, hi };
}

function boxUnion(a, b) {
  return {
    lo: [Math.min(a.lo[0], b.lo[0]), Math.min(a.lo[1], b.lo[1]), Math.min(a.lo[2], b.lo[2])],
    hi: [Math.max(a.hi[0], b.hi[0]), Math.max(a.hi[1], b.hi[1]), Math.max(a.hi[2], b.hi[2])],
  };
}

function boxOverlap(a, b) {
  for (let i = 0; i < 3; i++) if (a.hi[i] < b.lo[i] || b.hi[i] < a.lo[i]) return false;
  return true;
}

// ---- primitive: exact-ish triangle/triangle test ---------------------------------------
let exactFallbacks = 0;

/** Crossing points of `tri` with a plane, projected on one axis (as float positions). */
function crossingInterval(tri, distances, axis) {
  const hits = [];
  for (let i = 0; i < 3; i++) {
    const j = (i + 1) % 3;
    const di = distances[i];
    const dj = distances[j];
    if (di === 0) hits.push(tri[i][axis]);
    if ((di > 0 && dj < 0) || (di < 0 && dj > 0)) {
      const t = di / (di - dj);
      hits.push(tri[i][axis] + t * (tri[j][axis] - tri[i][axis]));
    }
  }
  if (hits.length === 0) return null;
  return [Math.min(...hits), Math.max(...hits)];
}

/** Crossing points as exact rationals [numerator, denominator] for the ambiguous comparisons. */
function rationalCrossings(tri, distances, axis) {
  const out = [];
  for (let i = 0; i < 3; i++) {
    const j = (i + 1) % 3;
    const di = distances[i];
    const dj = distances[j];
    if (di === 0) out.push([tri[i][axis], 1]);
    if ((di > 0 && dj < 0) || (di < 0 && dj > 0)) {
      const denominator = di - dj;
      out.push([tri[i][axis] * denominator + di * (tri[j][axis] - tri[i][axis]), denominator]);
    }
  }
  return out;
}

function rationalLess(a, b) {
  let [an, ad] = a;
  let [bn, bd] = b;
  if (ad < 0) {
    an = -an;
    ad = -ad;
  }
  if (bd < 0) {
    bn = -bn;
    bd = -bd;
  }
  return an * bd < bn * ad;
}

function exactIntervalOverlap(t2, dv2, t1, dv1, axis) {
  exactFallbacks++;
  const span = (points) => {
    let lo = points[0];
    let hi = points[0];
    for (const p of points) {
      if (rationalLess(p, lo)) lo = p;
      if (rationalLess(hi, p)) hi = p;
    }
    return [lo, hi];
  };
  const first = span(rationalCrossings(t2, dv2, axis));
  const second = span(rationalCrossings(t1, dv1, axis));
  return !rationalLess(first[1], second[0]) && !rationalLess(second[1], first[0]);
}

/** 2D segment intersection; orientations of integer coordinates are exact. */
function segments2(p1, p2, p3, p4) {
  const orient = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const inBox = (a, b, c) =>
    Math.min(a[0], b[0]) <= c[0] && c[0] <= Math.max(a[0], b[0]) &&
    Math.min(a[1], b[1]) <= c[1] && c[1] <= Math.max(a[1], b[1]);
  const opposite = (u, v) => (u > 0 && v < 0) || (u < 0 && v > 0);
  const d1 = orient(p3, p4, p1);
  const d2 = orient(p3, p4, p2);
  const d3 = orient(p1, p2, p3);
  const d4 = orient(p1, p2, p4);
  if (opposite(d1, d2) && opposite(d3, d4)) return true;
  return (d1 === 0 && inBox(p3, p4, p1)) || (d2 === 0 && inBox(p3, p4, p2)) ||
    (d3 === 0 && inBox(p1, p2, p3)) || (d4 === 0 && inBox(p1, p2, p4));
}

function pointInTriangle2(p, tri) {
  const orient = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const s = [orient(tri[0], tri[1], p), orient(tri[1], tri[2], p), orient(tri[2], tri[0], p)];
  return !((s[0] < 0 || s[1] < 0 || s[2] < 0) && (s[0] > 0 || s[1] > 0 || s[2] > 0));
}

/** Coplanar pair: 2D separating-axis test on the in-plane edge normals. */
function coplanarOverlap(t1, t2, normal) {
  const drop = dominantAxis(normal);
  const flat = (tri) => tri.map((p) => project2(p, drop));
  const a = flat(t1);
  const b = flat(t2);
  const axes = [];
  for (const poly of [a, b]) {
    for (let i = 0; i < 3; i++) {
      const p = poly[i];
      const q = poly[(i + 1) % 3];
      axes.push([q[1] - p[1], p[0] - q[0]]);
    }
  }
  for (const axis of axes) {
    if (axis[0] === 0 && axis[1] === 0) continue;
    const span = (poly) => {
      const values = poly.map((p) => p[0] * axis[0] + p[1] * axis[1]);
      return [Math.min(...values), Math.max(...values)];
    };
    const sa = span(a);
    const sb = span(b);
    if (sa[0] > sb[1] || sb[0] > sa[1]) return false;
  }
  return true;
}

/** Longest edge of a zero-area triangle: it spans the collinear vertices. */
function longestEdge(tri) {
  const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
  let best = [tri[0], tri[1]];
  let bestLen = dist2(tri[0], tri[1]);
  for (const [i, j] of [[1, 2], [2, 0]]) {
    const len = dist2(tri[i], tri[j]);
    if (len > bestLen) {
      bestLen = len;
      best = [tri[i], tri[j]];
    }
  }
  return best;
}

function pointInTriangle3(x, tri, n) {
  for (let i = 0; i < 3; i++) {
    if (dot(cross(sub(tri[(i + 1) % 3], tri[i]), sub(x, tri[i])), n) < 0) return false;
  }
  return true;
}

/** Segment against a non-degenerate triangle: the plane is crossed, so check containment. */
function segmentTriangleIntersect(p, q, tri) {
  const n = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]));
  const dp = dot(n, sub(p, tri[0]));
  const dq = dot(n, sub(q, tri[0]));
  if ((dp > 0 && dq > 0) || (dp < 0 && dq < 0)) return false;
  if (dp === 0 && dq === 0) { // the segment lies in the triangle's plane
    const drop = dominantAxis(n);
    const fa = project2(p, drop);
    const fb = project2(q, drop);
    const flat = tri.map((v) => project2(v, drop));
    if (segments2(fa, fb, flat[0], flat[1]) || segments2(fa, fb, flat[1], flat[2]) ||
      segments2(fa, fb, flat[2], flat[0])) return true;
    return pointInTriangle2(fa, flat) || pointInTriangle2(fb, flat);
  }
  const t = dp / (dp - dq);
  const x = [0, 1, 2].map((axis) => p[axis] + t * (q[axis] - p[axis]));
  return pointInTriangle3(x, tri, n);
}

/** Segment against segment: skew segments cannot meet, coplanar ones are compared in 2D. */
function segmentsIntersect3(a1, a2, b1, b2) {
  const d1 = sub(a2, a1);
  const d2 = sub(b2, b1);
  const r = sub(b1, a1);
  const n = cross(d1, d2);
  let drop;
  if (!isZeroVec(n)) {
    if (dot(r, n) !== 0) return false;
    drop = dominantAxis(n);
  } else {
    if (!isZeroVec(cross(r, d1))) return false; // parallel but on different lines
    drop = dominantAxis(d1);
  }
  return segments2(project2(a1, drop), project2(a2, drop), project2(b1, drop), project2(b2, drop));
}

/** Moeller's triangle-triangle test for closed triangles (touching counts as intersecting). */
function triTriIntersect(t1, t2) {
  const n1 = cross(sub(t1[1], t1[0]), sub(t1[2], t1[0]));
  const n2 = cross(sub(t2[1], t2[0]), sub(t2[2], t2[0]));
  if (isZeroVec(n1) || isZeroVec(n2)) {
    const s1 = longestEdge(t1);
    const s2 = longestEdge(t2);
    if (isZeroVec(n1) && isZeroVec(n2)) return segmentsIntersect3(s1[0], s1[1], s2[0], s2[1]);
    if (isZeroVec(n1)) return segmentTriangleIntersect(s1[0], s1[1], t2);
    return segmentTriangleIntersect(s2[0], s2[1], t1);
  }
  const d1 = -dot(n1, t1[0]);
  const dv2 = t2.map((p) => dot(n1, p) + d1);
  if (dv2.every((v) => v > 0) || dv2.every((v) => v < 0)) return false;
  const d2 = -dot(n2, t2[0]);
  const dv1 = t1.map((p) => dot(n2, p) + d2);
  if (dv1.every((v) => v > 0) || dv1.every((v) => v < 0)) return false;
  if (dv2.every((v) => v === 0)) return coplanarOverlap(t1, t2, n1);
  const axis = dominantAxis(cross(n1, n2)); // direction of plane1 ^ plane2
  const onPlane1 = crossingInterval(t2, dv2, axis);
  const onPlane2 = crossingInterval(t1, dv1, axis);
  if (onPlane1 === null || onPlane2 === null) return false;
  const gap = Math.max(onPlane1[0] - onPlane2[1], onPlane2[0] - onPlane1[1]);
  const scale = Math.max(1, ...t1.flat().map(Math.abs), ...t2.flat().map(Math.abs));
  if (Math.abs(gap) > 1e-9 * scale) return gap < 0;
  return exactIntervalOverlap(t2, dv2, t1, dv1, axis);
}

// ---- exact oracle: the origin must lie in the Minkowski difference ----------------------
const big = (v) => v.map((c) => BigInt(c));
const bsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const bcross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const bdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const bIsZero = (v) => v[0] === 0n && v[1] === 0n && v[2] === 0n;

function det3(rows) {
  const [[a, b, c], [d, e, f], [g, h, i]] = rows;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

/** 4x4 determinant, expanded along the first row. */
function det4(rows) {
  let total = 0n;
  for (let column = 0; column < 4; column++) {
    const minor = det3(rows.slice(1).map((row) => row.filter((_, c) => c !== column)));
    const term = rows[0][column] * minor;
    total += column % 2 === 0 ? term : -term;
  }
  return total;
}

/** Sign of the origin's side of the directed edge (a -> b) inside the plane with normal n. */
function originSide(n, a, b) {
  return bdot(n, bcross(bsub(b, a), [-a[0], -a[1], -a[2]]));
}

function originInTriangle(p, q, r) {
  const n = bcross(bsub(q, p), bsub(r, p));
  if (bIsZero(n)) return false; // collinear triples are decided by their own pairs
  if (bdot(n, p) !== 0n) return false; // the origin is not even in the triangle's plane
  const signs = [originSide(n, p, q), originSide(n, q, r), originSide(n, r, p)];
  return !(signs.some((s) => s < 0n) && signs.some((s) => s > 0n));
}

function originInTetrahedron(p, q, r, s) {
  const rows = [p, q, r, s].map((v) => [v[0], v[1], v[2], 1n]);
  const determinant = det4(rows);
  if (determinant === 0n) return false; // affinely dependent: a subset of size <= 3 decides
  const rhs = [0n, 0n, 0n, 1n];
  let anyNegative = false;
  let anyPositive = false;
  for (let column = 0; column < 4; column++) {
    // Cramer: replace the point's column of the barycentric system by the right-hand side.
    const replaced = rows.map((row, k) => (k === column ? rhs : row));
    const numerator = det4(replaced);
    if (numerator < 0n) anyNegative = true;
    if (numerator > 0n) anyPositive = true;
  }
  return determinant > 0n ? !anyNegative : !anyPositive;
}

/**
 * Exact: 0 in conv{points} iff an affinely independent witness subset of size <= 4 contains it
 * (Caratheodory), so enumerating points, pairs, triples and quadruples is complete.
 */
function originInConvexHull(points) {
  const n = points.length;
  for (const p of points) if (bIsZero(p)) return true;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const p = points[i];
      const q = points[j];
      if (bIsZero(bcross(p, q)) && bdot(p, q) <= 0n) return true; // opposite ends of one line
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        if (originInTriangle(points[i], points[j], points[k])) return true;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        for (let l = k + 1; l < n; l++) {
          if (originInTetrahedron(points[i], points[j], points[k], points[l])) return true;
        }
      }
    }
  }
  return false;
}

/** Exact independent oracle: t1 and t2 meet iff 0 lies in conv{b - a} over all vertex pairs. */
function exactTriTriIntersect(t1, t2) {
  for (const pt of [...t1, ...t2]) {
    if (pt.some((c) => !Number.isInteger(c))) {
      throw new TypeError("the exact oracle needs integer coordinates");
    }
  }
  const differences = [];
  for (const b of t2) for (const a of t1) differences.push(bsub(big(b), big(a)));
  return originInConvexHull(differences);
}

// ---- the hierarchy ---------------------------------------------------------------------
class BoundingVolumeHierarchy {
  constructor(triangles, leafSize = 4) {
    this.triangles = triangles;
    this.leafSize = leafSize;
    this.nodes = [];
    const indices = triangles.map((_, i) => i);
    this.root = indices.length === 0 ? null : this.#build(indices);
  }

  #build(items) {
    let box = triBounds(this.triangles[items[0]]);
    for (const i of items.slice(1)) box = boxUnion(box, triBounds(this.triangles[i]));
    const node = { items: null, left: null, right: null, box, size: items.length };
    this.nodes.push(node);
    if (items.length <= this.leafSize) {
      node.items = items;
      return node;
    }
    const low = [Infinity, Infinity, Infinity];
    const high = [-Infinity, -Infinity, -Infinity];
    for (const i of items) {
      const b = triBounds(this.triangles[i]);
      for (let a = 0; a < 3; a++) {
        const center = (b.lo[a] + b.hi[a]) / 2;
        low[a] = Math.min(low[a], center);
        high[a] = Math.max(high[a], center);
      }
    }
    const extent = [high[0] - low[0], high[1] - low[1], high[2] - low[2]];
    const axis = extent[0] >= extent[1] && extent[0] >= extent[2] ? 0 : extent[1] >= extent[2] ? 1 : 2;
    const sorted = items.slice().sort((i, j) => {
      const bi = triBounds(this.triangles[i]);
      const bj = triBounds(this.triangles[j]);
      return bi.lo[axis] + bi.hi[axis] - (bj.lo[axis] + bj.hi[axis]) || i - j;
    });
    const half = sorted.length >> 1;
    node.left = this.#build(sorted.slice(0, half));
    node.right = this.#build(sorted.slice(half));
    return node;
  }

  /** All intersecting index pairs with i < j, plus how many primitive calls it took. */
  selfIntersections() {
    const pairs = [];
    let primitiveCalls = 0;
    if (this.root === null) return { pairs, primitiveCalls };
    const descend = (a, b) => {
      if (!boxOverlap(a.box, b.box)) return;
      if (a.items !== null && b.items !== null) {
        for (const i of a.items) {
          for (const j of b.items) {
            if (a === b && j <= i) continue;
            primitiveCalls++;
            if (triTriIntersect(this.triangles[i], this.triangles[j])) pairs.push(i < j ? [i, j] : [j, i]);
          }
        }
        return;
      }
      if (a === b) {
        descend(a.left, a.left);
        descend(a.left, a.right);
        descend(a.right, a.right);
        return;
      }
      if (a.items !== null) {
        descend(a, b.left);
        descend(a, b.right);
        return;
      }
      if (b.items !== null || a.size >= b.size) {
        descend(a.left, b);
        descend(a.right, b);
        return;
      }
      descend(a, b.left);
      descend(a, b.right);
    };
    descend(this.root, this.root);
    pairs.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    return { pairs, primitiveCalls };
  }
}

function brutePairs(triangles) {
  const pairs = [];
  for (let i = 0; i < triangles.length; i++) {
    for (let j = i + 1; j < triangles.length; j++) {
      if (triTriIntersect(triangles[i], triangles[j])) pairs.push([i, j]);
    }
  }
  return pairs;
}

/** Every triangle covered by its node box, children nested, leaves small enough. */
function checkNodeInvariants(bvh, node) {
  const covered = [];
  const walk = (n) => {
    if (n.items !== null) {
      ok(n.items.length <= bvh.leafSize, "leaf size bound");
      for (const i of n.items) {
        covered.push(i);
        const b = triBounds(bvh.triangles[i]);
        for (let a = 0; a < 3; a++) {
          ok(b.lo[a] >= n.box.lo[a] && b.hi[a] <= n.box.hi[a], "node box must cover its triangles");
        }
      }
      return;
    }
    walk(n.left);
    walk(n.right);
    strictEqual(n.size, n.left.size + n.right.size, "subtree sizes must add up");
    for (let a = 0; a < 3; a++) {
      ok(n.left.box.lo[a] >= n.box.lo[a] && n.left.box.hi[a] <= n.box.hi[a], "left box nested");
      ok(n.right.box.lo[a] >= n.box.lo[a] && n.right.box.hi[a] <= n.box.hi[a], "right box nested");
    }
  };
  walk(node);
  return covered.sort((x, y) => x - y);
}

module.exports = {
  triTriIntersect,
  exactTriTriIntersect,
  BoundingVolumeHierarchy,
  brutePairs,
  makeRng,
  originInConvexHull,
  exactFallbackCount: () => exactFallbacks,
};

if (require.main === module) {
  const t = (a, b, c) => [a, b, c];
  const X1 = t([-1, 0, -1], [1, 0, -1], [0, 0, 1]);
  const X2 = t([0, -1, -1], [0, 1, -1], [0, 0, 1]);
  const far = t([-1, 5, -1], [1, 5, -1], [0, 5, 1]);
  const plane = t([-1, -1, 0], [2, -1, 0], [-1, 2, 0]);
  const onPlane = t([1, -1, 0], [2, 1, 0], [1, 2, 0]);
  const offPlane = t([5, 5, 0], [6, 5, 0], [5, 6, 0]);

  const cases = [
    ["crossing", X1, X2, true],
    ["separated planes", X1, far, false],
    ["parallel below", X1, t([-1, -3, -1], [1, -3, -1], [0, -3, 1]), false],
    ["coplanar overlap", plane, onPlane, true],
    ["coplanar disjoint", plane, offPlane, false],
    ["coplanar shared boundary", plane, t([2, -1, 0], [3, 2, 0], [-1, 2, 0]), true],
    ["vertex touch", t([0, 0, 0], [4, 0, 0], [0, 0, 4]), t([0, 0, 0], [0, 4, 0], [2, 2, 2]), true],
    ["apart", t([0, 0, 0], [1, 0, 0], [0, 1, 0]), t([9, 9, 9], [10, 9, 9], [9, 10, 9]), false],
    ["degenerate vs crossing", t([0, 0, 0], [1, 1, 1], [2, 2, 2]), t([0, 2, 0], [3, -1, 1], [3, -1, -1]), false],
    ["degenerate vs hitting", t([0, 0, 0], [1, 1, 1], [2, 2, 2]), t([0, 0, 0], [3, 0, 1], [3, 0, -1]), true],
    ["two degenerate segments", t([0, 0, 0], [1, 1, 1], [2, 2, 2]), t([0, 2, 2], [1, 1, 1], [2, 0, 0]), true],
  ];
  for (const [label, a, b, expected] of cases) {
    strictEqual(triTriIntersect(a, b), expected, `primitive: ${label}`);
    strictEqual(exactTriTriIntersect(a, b), expected, `oracle: ${label}`);
  }
  throws(() => exactTriTriIntersect(plane, t([0.5, 0, 0], [1, 0, 0], [0, 1, 0])), TypeError,
    "the exact oracle must reject non-integer coordinates");

  // -- the float primitive against the exact oracle on random small-integer triangles ----
  const rng = makeRng(736736);
  const randomTri = (spread, center = [0, 0, 0]) => {
    const pts = [];
    for (let k = 0; k < 3; k++) {
      pts.push([
        center[0] + rng(spread) - (spread >> 1),
        center[1] + rng(spread) - (spread >> 1),
        center[2] + rng(spread) - (spread >> 1),
      ]);
    }
    return pts;
  };
  let agreements = 0;
  let intersections = 0;
  for (let trial = 0; trial < 400; trial++) {
    const a = randomTri(9);
    const b = randomTri(9, [rng(7) - 3, rng(7) - 3, rng(7) - 3]);
    const fast = triTriIntersect(a, b);
    const exact = exactTriTriIntersect(a, b);
    strictEqual(fast, exact, `primitive and oracle disagree on trial ${trial}`);
    agreements++;
    if (exact) intersections++;
  }
  ok(intersections > 10, `the corpus must contain real intersections (got ${intersections})`);
  ok(exactFallbacks > 0, "near-touching pairs must reach the exact rational fallback");

  // -- hierarchy equals brute force, with the node invariants checked --------------------
  let hierarchyCalls = 0;
  let bruteCalls = 0;
  for (let trial = 0; trial < 15; trial++) {
    const n = 12 + rng(19);
    const triangles = [];
    for (let i = 0; i < n; i++) triangles.push(randomTri(8, [rng(13) - 6, rng(13) - 6, rng(13) - 6]));
    const bvh = new BoundingVolumeHierarchy(triangles, 4);
    const { pairs, primitiveCalls } = bvh.selfIntersections();
    deepStrictEqual(pairs, brutePairs(triangles), `trial ${trial}: BVH pairs must equal brute force`);
    deepStrictEqual(checkNodeInvariants(bvh, bvh.root), triangles.map((_, i) => i),
      `trial ${trial}: the tree must cover every triangle exactly once`);
    hierarchyCalls += primitiveCalls;
    bruteCalls += (n * (n - 1)) / 2;
  }
  ok(hierarchyCalls < bruteCalls, "the hierarchy must not do more work than brute force");
  const pruning = 100 * (1 - hierarchyCalls / bruteCalls);

  // -- separated clusters are pruned almost completely -----------------------------------
  const clusters = [];
  for (let c = 0; c < 6; c++) for (let i = 0; i < 4; i++) clusters.push(randomTri(2, [c * 100, i * 5, 0]));
  const spread = new BoundingVolumeHierarchy(clusters, 4).selfIntersections();
  deepStrictEqual(spread.pairs, brutePairs(clusters));
  const spreadBrute = (clusters.length * (clusters.length - 1)) / 2;
  ok(spread.primitiveCalls < 0.4 * spreadBrute,
    `far-apart clusters must be pruned (calls=${spread.primitiveCalls}, brute=${spreadBrute})`);

  // -- determinism and degenerate inputs -------------------------------------------------
  const cloud = [];
  for (let i = 0; i < 40; i++) cloud.push(randomTri(10, [rng(15) - 7, rng(15) - 7, rng(15) - 7]));
  const first = new BoundingVolumeHierarchy(cloud, 4);
  const second = new BoundingVolumeHierarchy(cloud, 4);
  strictEqual(first.nodes.length, second.nodes.length, "identical builds give identical trees");
  deepStrictEqual(first.selfIntersections().pairs, second.selfIntersections().pairs);

  strictEqual(new BoundingVolumeHierarchy([], 4).root, null, "an empty cloud builds no tree");
  const single = new BoundingVolumeHierarchy([X1], 4).selfIntersections();
  deepStrictEqual(single.pairs, []);
  strictEqual(single.primitiveCalls, 0);
  const duplicated = new BoundingVolumeHierarchy([X1, X1, X2, X2], 1).selfIntersections();
  deepStrictEqual(duplicated.pairs, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
    "duplicated triangles must pair up exactly once each");

  console.log(`random_pairs=${agreements} intersections=${intersections} exact_fallbacks=${exactFallbacks}`);
  console.log(`bvh_calls=${hierarchyCalls} brute_calls=${bruteCalls} pruning=${pruning.toFixed(1)}%`);
  console.log(`nodes=${first.nodes.length} cluster_calls=${spread.primitiveCalls}/${spreadBrute}`);
  console.log("PASS");
}
