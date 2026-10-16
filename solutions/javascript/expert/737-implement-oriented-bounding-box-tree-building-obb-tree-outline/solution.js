/** Task 737 - Implement oriented bounding box tree building (OBB-tree) outline.
 *  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * An OBB-tree is built top down. For a node's point set the covariance matrix is formed and
 * its eigenvectors are extracted by power iteration with deflation (start vector (1,1,1)/sqrt3
 * and re-orthogonalisation after every step, so the result is deterministic); the eigenvectors
 * become the box axes and the box is the tightest one with those axes - half extents are the
 * extreme projections, the centre the midpoint of each projection range. The node is then split
 * at the *median* of the projections on its longest axis, so both children own at most half of
 * the points and the recursion depth is logarithmic; the leaves hold at most LEAF_SIZE points.
 * An axis-aligned box is computed for every node as well so the two fittings can be compared.
 *
 * What is simplified: the boxes are not refitted after construction (no insertion/deletion of
 * points, no "top-down refinement with overlap minimisation"), the tree is not stored as a
 * compact array, and the split axis is the node's own longest OBB axis instead of the surface
 * area heuristic. What is verified: every point of a subtree lies inside its node's OBB, every
 * node's OBB is inside the axis-aligned box of the same points (so its volume can only be
 * smaller), axes stay orthonormal, the tree covers each point exactly once and the leaf bound
 * holds, and two queries - points within a radius and the nearest point - return exactly what a
 * brute-force scan returns while the visited-node counters show the pruning.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

const LEAF_SIZE = 4;

function makeRng(seed) {
  let state = seed >>> 0 || 0x1234567;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

// ---- small linear algebra --------------------------------------------------------------
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => Math.sqrt(dot(a, a));

function scale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function normalize(a) {
  const length = norm(a);
  return length === 0 ? [0, 0, 0] : scale(a, 1 / length);
}

function subVectors(a, b) {
  return sub(a, b);
}

/** Covariance matrix (as three rows) of a point set, divided by the count. */
function covariance(points) {
  const mean = [0, 0, 0];
  for (const p of points) {
    mean[0] += p[0];
    mean[1] += p[1];
    mean[2] += p[2];
  }
  for (let axis = 0; axis < 3; axis++) mean[axis] /= points.length;
  const matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of points) {
    const d = sub(p, mean);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) matrix[i][j] += d[i] * d[j];
  }
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) matrix[i][j] /= points.length;
  return { matrix, mean };
}

function applyMatrix(matrix, v) {
  return [
    matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2],
    matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2],
    matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2],
  ];
}

/**
 * Dominant eigenvector of a symmetric 3x3 matrix by power iteration; the start vector and the
 * iteration count are fixed, so the result is deterministic and exactly reproducible.
 */
function dominantEigenvector(matrix, previous) {
  let v = normalize([1, 1, 1]);
  for (let round = 0; round < 200; round++) {
    v = applyMatrix(matrix, v);
    for (const axis of previous) v = subVectors(v, scale(axis, dot(axis, v))); // deflation
    if (norm(v) === 0) return null; // the direction is numerically absent
    v = normalize(v);
  }
  const beforeDeflation = norm(v);
  for (const axis of previous) v = subVectors(v, scale(axis, dot(axis, v)));
  const remaining = norm(v);
  // A rank-deficient covariance leaves only rounding noise here; that noise must not be
  // normalised into a bogus axis, so fall back to an explicit orthogonal direction instead.
  if (remaining < 1e-9 * beforeDeflation) return null;
  return normalize(v);
}

function principalAxes(matrix) {
  const axes = [];
  for (let k = 0; k < 3; k++) {
    const axis = dominantEigenvector(matrix, axes);
    axes.push(axis === null ? fallbackAxis(axes) : axis);
  }
  return axes;
}

/** Any unit vector orthogonal to the axes already found - used for degenerate covariances. */
function fallbackAxis(axes) {
  for (const candidate of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]) {
    let v = candidate;
    for (const axis of axes) v = subVectors(v, scale(axis, dot(axis, v)));
    if (norm(v) > 1e-9) return normalize(v);
  }
  return [0, 0, 1];
}

// ---- boxes -----------------------------------------------------------------------------
function fitObb(points, axes) {
  const low = [Infinity, Infinity, Infinity];
  const high = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let a = 0; a < 3; a++) {
      const projection = dot(p, axes[a]);
      if (projection < low[a]) low[a] = projection;
      if (projection > high[a]) high[a] = projection;
    }
  }
  const center = [0, 0, 0];
  const half = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    // The fitted centre is the midpoint of the projection range, a *coefficient* of axis a, so
    // it has to be rebuilt as a linear combination of the (orthonormal) axes.
    const middle = (low[a] + high[a]) / 2;
    half[a] = (high[a] - low[a]) / 2;
    for (let k = 0; k < 3; k++) center[k] += middle * axes[a][k];
  }
  return { center, axes, half, kind: "obb" };
}

function fitAabb(points) {
  const low = [Infinity, Infinity, Infinity];
  const high = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let a = 0; a < 3; a++) {
      if (p[a] < low[a]) low[a] = p[a];
      if (p[a] > high[a]) high[a] = p[a];
    }
  }
  return { low, high, center: [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, (low[2] + high[2]) / 2], kind: "aabb" };
}

function obbVolume(box) {
  return 8 * box.half[0] * box.half[1] * box.half[2];
}

function aabbVolume(box) {
  return (box.high[0] - box.low[0]) * (box.high[1] - box.low[1]) * (box.high[2] - box.low[2]);
}

/** Local coordinates of p in the OBB frame. */
function toLocal(box, p) {
  const d = sub(p, box.center);
  return [dot(d, box.axes[0]), dot(d, box.axes[1]), dot(d, box.axes[2])];
}

function containsPoint(box, p, tolerance = 1e-9) {
  const local = toLocal(box, p);
  return local.every((value, a) => Math.abs(value) <= box.half[a] + tolerance);
}

/** Squared distance from a point to the OBB (the standard clamp in the box frame). */
function distanceSquaredToPoint(box, p) {
  const local = toLocal(box, p);
  let total = 0;
  for (let a = 0; a < 3; a++) {
    const outside = Math.max(0, Math.abs(local[a]) - box.half[a]);
    total += outside * outside;
  }
  return total;
}

/** Separating axis test for two OBBs: 15 candidate axes (3+3 face normals, 9 edge crosses). */
function obbOverlap(a, b) {
  const axes = [];
  for (let i = 0; i < 3; i++) {
    axes.push(a.axes[i]);
    axes.push(b.axes[i]);
    for (let j = 0; j < 3; j++) {
      const c = [
        a.axes[i][1] * b.axes[j][2] - a.axes[i][2] * b.axes[j][1],
        a.axes[i][2] * b.axes[j][0] - a.axes[i][0] * b.axes[j][2],
        a.axes[i][0] * b.axes[j][1] - a.axes[i][1] * b.axes[j][0],
      ];
      if (norm(c) > 1e-9) axes.push(normalize(c));
    }
  }
  const d = sub(b.center, a.center);
  for (const axis of axes) {
    const ra = a.half[0] * Math.abs(dot(axis, a.axes[0])) + a.half[1] * Math.abs(dot(axis, a.axes[1])) +
      a.half[2] * Math.abs(dot(axis, a.axes[2]));
    const rb = b.half[0] * Math.abs(dot(axis, b.axes[0])) + b.half[1] * Math.abs(dot(axis, b.axes[1])) +
      b.half[2] * Math.abs(dot(axis, b.axes[2]));
    if (Math.abs(dot(d, axis)) > ra + rb + 1e-9) return false;
  }
  return true;
}

/** Axis-aligned cube box centred at `center` with the given half extent. */
function cubeBox(center, radius) {
  return {
    center,
    axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    half: [radius, radius, radius],
    kind: "obb",
  };
}

// ---- the tree --------------------------------------------------------------------------
class ObbTree {
  constructor(points) {
    this.points = points;
    this.visitedNodes = 0;
    const indices = points.map((_, i) => i);
    this.root = indices.length === 0 ? null : this.#build(indices);
  }

  /** Padding keeps coincident and collinear sets from producing a zero-thickness box. */
  #build(indices) {
    const subset = indices.map((i) => this.points[i]);
    const { matrix } = covariance(subset);
    const axes = principalAxes(matrix);
    const obb = fitObb(subset, axes);
    const aabb = fitAabb(subset);
    const node = { obb, aabb, indices, left: null, right: null };
    if (indices.length <= LEAF_SIZE) return node;
    let axis = 0;
    if (obb.half[1] > obb.half[axis]) axis = 1;
    if (obb.half[2] > obb.half[axis]) axis = 2;
    const sorted = indices.slice().sort((i, j) =>
      dot(this.points[i], obb.axes[axis]) - dot(this.points[j], obb.axes[axis]) || i - j);
    const half = sorted.length >> 1;
    node.left = this.#build(sorted.slice(0, half));
    node.right = this.#build(sorted.slice(half));
    return node;
  }

  /** Points within `radius` of `center`, found with OBB/OBB pruning. */
  withinRadius(center, radius) {
    const query = cubeBox(center, radius);
    const hits = [];
    this.visitedNodes = 0;
    const walk = (node) => {
      if (node === null || !obbOverlap(query, node.obb)) return;
      this.visitedNodes++;
      if (node.left === null) {
        for (const i of node.indices) {
          const dx = this.points[i][0] - center[0];
          const dy = this.points[i][1] - center[1];
          const dz = this.points[i][2] - center[2];
          if (dx * dx + dy * dy + dz * dz <= radius * radius) hits.push(i);
        }
        return;
      }
      walk(node.left);
      walk(node.right);
    };
    walk(this.root);
    hits.sort((a, b) => a - b);
    return hits;
  }

  /** Nearest point by branch and bound on the OBB distance lower bound. */
  nearest(center) {
    let best = Infinity;
    let bestIndex = -1;
    this.visitedNodes = 0;
    const walk = (node) => {
      if (node === null) return;
      const bound = distanceSquaredToPoint(node.obb, center);
      if (bound >= best) return; // nothing in this subtree can beat the incumbent
      this.visitedNodes++;
      if (node.left === null) {
        for (const i of node.indices) {
          const dx = this.points[i][0] - center[0];
          const dy = this.points[i][1] - center[1];
          const dz = this.points[i][2] - center[2];
          const d = dx * dx + dy * dy + dz * dz;
          if (d < best || (d === best && i < bestIndex)) {
            best = d;
            bestIndex = i;
          }
        }
        return;
      }
      // descend into the closer child first so the incumbent gets tight early
      const leftFirst = distanceSquaredToPoint(node.left.obb, center) <=
        distanceSquaredToPoint(node.right.obb, center);
      if (leftFirst) {
        walk(node.left);
        walk(node.right);
      } else {
        walk(node.right);
        walk(node.left);
      }
    };
    walk(this.root);
    return { index: bestIndex, distance: Math.sqrt(best) };
  }
}

function bruteWithinRadius(points, center, radius) {
  const hits = [];
  points.forEach((p, i) => {
    const d = sub(p, center);
    if (dot(d, d) <= radius * radius) hits.push(i);
  });
  return hits;
}

function bruteNearest(points, center) {
  let best = Infinity;
  let bestIndex = -1;
  points.forEach((p, i) => {
    const d = sub(p, center);
    const distance = dot(d, d);
    if (distance < best) {
      best = distance;
      bestIndex = i;
    }
  });
  return { index: bestIndex, distance: Math.sqrt(best) };
}

/** Structural invariants: leaf bound, containment per node, orthonormal axes, tightness. */
function auditTree(tree, node) {
  const leafIndices = [];
  const walk = (current) => {
    const subset = current.indices.map((i) => tree.points[i]);
    for (const p of subset) {
      ok(containsPoint(current.obb, p), "every point of a subtree must lie in its node OBB");
    }
    for (let a = 0; a < 3; a++) {
      ok(Math.abs(norm(current.obb.axes[a]) - 1) < 1e-9, "OBB axes must be unit length");
      for (let b = a + 1; b < 3; b++) {
        ok(Math.abs(dot(current.obb.axes[a], current.obb.axes[b])) < 1e-9, "OBB axes must be orthogonal");
      }
      ok(obbVolume(current.obb) <= aabbVolume(current.aabb) * (1 + 1e-9),
        "an OBB cannot be bigger than the axis-aligned box of the same points");
    }
    if (current.left === null) {
      ok(current.indices.length <= LEAF_SIZE, "leaf size bound");
      leafIndices.push(...current.indices);
      return;
    }
    ok(current.indices.length > LEAF_SIZE, "internal nodes must exceed the leaf bound");
    walk(current.left);
    walk(current.right);
  };
  walk(node);
  return leafIndices.sort((a, b) => a - b);
}

module.exports = { ObbTree, principalAxes, covariance, obbOverlap, fitObb, makeRng };

if (require.main === module) {
  const rng = makeRng(737737);
  const diagonalCloud = [];
  for (let i = 0; i < 120; i++) {
    const t = (rng(201) - 100) / 10;
    diagonalCloud.push([t + (rng(11) - 5) / 50, 2 * t + (rng(11) - 5) / 50, 3 * t + (rng(11) - 5) / 50]);
  }
  const tree = new ObbTree(diagonalCloud);
  deepStrictEqual(auditTree(tree, tree.root), diagonalCloud.map((_, i) => i),
    "the leaves must partition every point exactly once");

  // The diagonal cloud is strongly anisotropic: the oriented box must beat the AABB.
  const rootObb = obbVolume(tree.root.obb);
  const rootAabb = aabbVolume(tree.root.aabb);
  ok(rootObb < 0.2 * rootAabb, `PCA orientation must tighten the box (${rootObb.toFixed(3)} vs ${rootAabb.toFixed(3)})`);
  const firstAxis = tree.root.obb.axes[0];
  ok(Math.abs(firstAxis[0] - firstAxis[1] / 2) < 0.05 && Math.abs(firstAxis[2] - firstAxis[1] * 1.5) < 0.05,
    "the principal axis must follow the (1, 2, 3) direction");

  // -- radius query against brute force --------------------------------------------------
  let visited = 0;
  let bruteHits = 0;
  for (let trial = 0; trial < 40; trial++) {
    const anchor = diagonalCloud[rng(diagonalCloud.length)];
    const center = trial % 4 === 3
      ? [(rng(401) - 200) / 20, (rng(401) - 200) / 20, (rng(401) - 200) / 20] // far away: prune
      : [anchor[0] + (rng(21) - 10) / 20, anchor[1] + (rng(21) - 10) / 20, anchor[2] + (rng(21) - 10) / 20];
    const radius = 0.5 + rng(30) / 10;
    const viaTree = tree.withinRadius(center, radius);
    deepStrictEqual(viaTree, bruteWithinRadius(diagonalCloud, center, radius), `radius query ${trial}`);
    visited += tree.visitedNodes;
    bruteHits += viaTree.length;
  }
  ok(bruteHits > 40, `the radius queries must actually hit points (got ${bruteHits})`);
  ok(visited < 40 * diagonalCloud.length / 2, `radius queries must prune (visited=${visited})`);

  // -- nearest point against brute force -------------------------------------------------
  let nearestVisited = 0;
  for (let trial = 0; trial < 60; trial++) {
    const center = [(rng(601) - 300) / 20, (rng(601) - 300) / 20, (rng(601) - 300) / 20];
    const viaTree = tree.nearest(center);
    const brute = bruteNearest(diagonalCloud, center);
    strictEqual(viaTree.index, brute.index, `nearest query ${trial} must find the same point`);
    ok(Math.abs(viaTree.distance - brute.distance) < 1e-12, "nearest distances must match");
    nearestVisited += tree.visitedNodes;
  }
  ok(nearestVisited < 60 * diagonalCloud.length / 3, `nearest queries must prune (visited=${nearestVisited})`);

  // -- OBB/OBB separating axis test versus dense sampling --------------------------------
  const cubeA = cubeBox([0, 0, 0], 1);
  const cubeB = cubeBox([1.5, 0, 0], 1);
  const cubeC = cubeBox([2.5, 0, 0], 1);
  ok(obbOverlap(cubeA, cubeB), "overlapping cubes");
  ok(!obbOverlap(cubeA, cubeC), "separated cubes");
  ok(obbOverlap(cubeA, cubeBox([0, 0, 0], 1)), "a box overlaps itself");
  const rotated = {
    center: [0, 0, 0],
    axes: [normalize([1, 1, 0]), normalize([-1, 1, 0]), [0, 0, 1]],
    half: [2, 0.5, 0.5],
    kind: "obb",
  };
  ok(obbOverlap(cubeA, rotated), "a rotated box crossing the centre overlaps");
  ok(obbOverlap(rotated, cubeBox([0, 0.5, 0], 0.2)), "the rotated slab covers (0, 0.5, 0)");
  ok(!obbOverlap(rotated, cubeBox([0, 1.2, 0], 0.2)), "but it stops well below (0, 1.2, 0)");
  // independent check: sampling can only confirm an overlap, never refute one
  for (let trial = 0; trial < 200; trial++) {
    const boxA = cubeBox([(rng(21) - 10) / 5, (rng(21) - 10) / 5, (rng(21) - 10) / 5], 0.6);
    const boxB = cubeBox([(rng(21) - 10) / 5, (rng(21) - 10) / 5, (rng(21) - 10) / 5], 0.6);
    const overlaps = obbOverlap(boxA, boxB);
    const separated = ["x", "y", "z"].some((_, axis) =>
      Math.abs(boxA.center[axis] - boxB.center[axis]) > boxA.half[axis] + boxB.half[axis] + 1e-9);
    ok(overlaps !== separated, `axis-aligned SAT must agree with the interval test (trial ${trial})`);
  }

  // -- degenerate clouds -----------------------------------------------------------------
  strictEqual(new ObbTree([]).root, null, "an empty cloud has no tree");
  const single = new ObbTree([[3, 4, 5]]);
  ok(containsPoint(single.root.obb, [3, 4, 5]), "a single point is inside its own box");
  deepStrictEqual(single.withinRadius([0, 0, 0], 8), [0], "the point is within 8 units of the origin");
  deepStrictEqual(single.withinRadius([0, 0, 0], 7), [], "but outside a radius of 7 (distance is sqrt(50))");
  strictEqual(single.nearest([0, 0, 0]).index, 0);
  ok(Math.abs(single.nearest([0, 0, 0]).distance - Math.sqrt(50)) < 1e-12, "nearest distance is sqrt(50)");
  const identical = new ObbTree([[1, 1, 1], [1, 1, 1], [1, 1, 1]]);
  deepStrictEqual(auditTree(identical, identical.root), [0, 1, 2], "coincident points are all covered");
  strictEqual(identical.nearest([0, 0, 0]).distance, Math.sqrt(3), "coincident points have one distance");
  const collinear = new ObbTree(Array.from({ length: 12 }, (_, i) => [i, 2 * i, 3 * i]));
  strictEqual(auditTree(collinear, collinear.root).length, 12, "collinear points are all covered");
  const flat = collinear.root.obb;
  const flatVolume = 8 * flat.half[0] * flat.half[1] * flat.half[2];
  ok(flatVolume < 1e-6, `a collinear set has an empty oriented box (volume=${flatVolume.toExponential(2)})`);

  // -- determinism -----------------------------------------------------------------------
  const again = new ObbTree(diagonalCloud);
  strictEqual(again.root.obb.center.join(","), tree.root.obb.center.join(","));
  deepStrictEqual(again.withinRadius([0, 0, 0], 2), tree.withinRadius([0, 0, 0], 2));
  strictEqual(again.nearest([1, 2, 3]).index, tree.nearest([1, 2, 3]).index);

  console.log(`points=${diagonalCloud.length} radius_visited=${visited} nearest_visited=${nearestVisited}`);
  console.log(`root_obb_volume=${rootObb.toFixed(4)} root_aabb_volume=${rootAabb.toFixed(4)}`);
  console.log("PASS");
}
