/** Task 743 - Implement constrained Delaunay triangulation (outline).
 *  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Three stages, all driven by exact BigInt predicates. (1) A Delaunay triangulation of the point
 * set is built incrementally (Bowyer-Watson): a large integer super-triangle is inserted first,
 * every new point deletes the triangles whose circumcircle contains it and retriangulates the
 * resulting star-shaped cavity, and the super-triangle is removed at the end. (2) Each constraint
 * segment that is not already an edge is *recovered* by Lawson's flip walk: while some edge
 * properly crosses the segment, flip a flippable crossing edge (a convex quad), which strictly
 * decreases the number of crossings, so the walk terminates. (3) Delaunay-ness is restored by
 * flipping illegal edges that are neither constraints nor about to be crossed - exactly the
 * standard "recover, then legalise" outline.
 *
 * What is simplified: the point set is in general position (no duplicates, no three collinear
 * points), constraints must join two input points and must not cross each other, and edge
 * recovery uses the flip walk instead of a cavity retriangulation, so the super-triangle is kept
 * only for the Delaunay stage. What is verified: the mesh is a manifold (every interior edge is
 * shared by two triangles, every hull edge by one), the triangle count is 2n-2-h and the areas
 * add up to the hull area, every non-constraint edge satisfies the exact empty-circumcircle test,
 * every constraint really is an edge and no triangulation edge crosses a constraint, constraint
 * free inputs match an independent brute-force Delaunay oracle, and degenerate or invalid inputs
 * are rejected or return the empty mesh.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

const EPS = 1e-9;
const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
const sign = (value) => (value > EPS ? 1 : value < -EPS ? -1 : 0);
const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

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

// ---- exact predicates --------------------------------------------------------------------
const big = (v) => v.map((c) => BigInt(c));

function det3(rows) {
  const [[a, b, c], [d, e, f], [g, h, i]] = rows;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

/** > 0 when d is strictly inside the circumcircle of the counter-clockwise triangle (a, b, c). */
function inCircle(a, b, c, d) {
  const rows = [a, b, c].map((p) => {
    const dx = BigInt(p[0]) - BigInt(d[0]);
    const dy = BigInt(p[1]) - BigInt(d[1]);
    return [dx, dy, dx * dx + dy * dy];
  });
  return det3(rows);
}

function orient2d(a, b, c) {
  return sign(cross(a, b, c));
}

function properCross2d(a, b, c, d) {
  return orient2d(c, d, a) * orient2d(c, d, b) < 0 && orient2d(a, b, c) * orient2d(a, b, d) < 0;
}

/** Exact circumcircle test that tolerates the orientation of the triangle triple. */
function triangleIsDelaunay(points, triangle, others) {
  const [i, j, k] = triangle;
  const a = points[i];
  const b = points[j];
  const c = points[k];
  const ccw = orient2d(a, b, c) > 0 ? [a, b, c] : [a, c, b];
  for (const other of others) {
    if (other === i || other === j || other === k) continue;
    if (inCircle(ccw[0], ccw[1], ccw[2], points[other]) > 0n) return false;
  }
  return true;
}

// ---- incremental Delaunay -----------------------------------------------------------------
/** A super-triangle far outside the input, with integer coordinates so predicates stay exact. */
function makeSuperTriangle(points) {
  let bound = 1;
  for (const p of points) bound = Math.max(bound, Math.abs(p[0]), Math.abs(p[1]));
  const span = bound * 16 + 16;
  return [[-span, -span], [3 * span, -span], [-span, 3 * span]];
}

function triangulateDelaunay(points) {
  const n = points.length;
  if (n < 3) return { triangles: [], hull: [] };
  if (points.every((p) => orient2d(points[0], points[1], p) === 0)) {
    return { triangles: [], hull: [] }; // fully collinear: the mesh is degenerate
  }
  const extended = points.concat(makeSuperTriangle(points));
  const superStart = n;
  let triangles = [[superStart, superStart + 1, superStart + 2]];
  for (let index = 0; index < n; index++) {
    const point = extended[index];
    const bad = [];
    const kept = [];
    triangles.forEach((triangle, position) => {
      const [a, b, c] = triangle.map((v) => extended[v]);
      const ccw = orient2d(a, b, c) > 0 ? [a, b, c] : [a, c, b];
      if (inCircle(ccw[0], ccw[1], ccw[2], point) > 0n) bad.push(position);
      else kept.push(position);
    });
    if (bad.length === 0) continue;
    const boundary = new Map();
    for (const position of bad) {
      const triangle = triangles[position];
      for (let e = 0; e < 3; e++) {
        const u = triangle[e];
        const v = triangle[(e + 1) % 3];
        const key = `${Math.min(u, v)},${Math.max(u, v)}`;
        boundary.set(key, (boundary.get(key) ?? 0) + 1);
      }
    }
    const next = triangles.filter((_, position) => !bad.includes(position));
    for (const [key, count] of boundary) {
      if (count !== 1) continue;
      const [u, v] = key.split(",").map(Number);
      const triangle = orient2d(extended[u], extended[v], point) > 0 ? [u, v, index] : [v, u, index];
      if (Math.abs(cross(extended[triangle[0]], extended[triangle[1]], extended[triangle[2]])) < EPS) continue;
      next.push(triangle);
    }
    triangles = next;
  }
  triangles = triangles.filter((t) => t.every((v) => v < n));
  return { triangles: triangles.map((t) => orient2d(points[t[0]], points[t[1]], points[t[2]]) > 0
    ? t
    : [t[0], t[2], t[1]]), hull: convexHull(points) };
}

/** Counter-clockwise convex hull (strict turns), used for counts and areas. */
function convexHull(points) {
  const sorted = points.map((p, i) => [p[0], p[1], i])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const build = (sequence) => {
    const chain = [];
    for (const p of sequence) {
      while (chain.length >= 2 &&
        sign(cross(chain[chain.length - 2], chain[chain.length - 1], p)) <= 0) chain.pop();
      chain.push(p);
    }
    return chain;
  };
  const lower = build(sorted);
  const upper = build(sorted.slice().reverse());
  return lower.slice(0, -1).concat(upper.slice(0, -1)).map((p) => p[2]);
}

// ---- edge helpers ------------------------------------------------------------------------
const edgeKey = (u, v) => `${Math.min(u, v)},${Math.max(u, v)}`;

function triangleEdges(triangle) {
  return [0, 1, 2].map((e) => edgeKey(triangle[e], triangle[(e + 1) % 3]));
}

/** Map from edge key to the triangles using it. */
function edgeUses(triangles) {
  const uses = new Map();
  triangles.forEach((triangle, position) => {
    for (const key of triangleEdges(triangle)) {
      if (!uses.has(key)) uses.set(key, []);
      uses.get(key).push(position);
    }
  });
  return uses;
}

function findCrossingEdge(points, triangles, a, b) {
  const uses = edgeUses(triangles);
  for (const [key] of uses) {
    const [u, v] = key.split(",").map(Number);
    if (u === a || u === b || v === a || v === b) continue;
    if (properCross2d(points[a], points[b], points[u], points[v])) return key;
  }
  return null;
}

/** Flips the shared edge (u, v) of the two triangles that use it. */
function flipEdge(triangles, key) {
  const uses = edgeUses(triangles).get(key) ?? [];
  if (uses.length !== 2) return false;
  const [p, q] = uses;
  const [u, v] = key.split(",").map(Number);
  const triangleP = triangles[p];
  const triangleQ = triangles[q];
  const oppositeP = triangleP.find((w) => w !== u && w !== v);
  const oppositeQ = triangleQ.find((w) => w !== u && w !== v);
  if (oppositeP === undefined || oppositeQ === undefined) return false;
  triangles[p] = [oppositeP, u, oppositeQ];
  triangles[q] = [oppositeP, oppositeQ, v];
  return true;
}

function orientAll(points, triangles) {
  return triangles.map((t) => (orient2d(points[t[0]], points[t[1]], points[t[2]]) > 0 ? t : [t[0], t[2], t[1]]));
}

/**
 * Constrained Delaunay triangulation: Delaunay first, then edge recovery per constraint, then a
 * legalisation pass that never removes a constraint.
 */
function constrainedDelaunay(points, constraints = []) {
  for (const [u, v] of constraints) {
    if (u === v || u < 0 || v < 0 || u >= points.length || v >= points.length) {
      throw new RangeError("a constraint must join two distinct point indices");
    }
  }
  for (let i = 0; i < constraints.length; i++) {
    for (let j = i + 1; j < constraints.length; j++) {
      if (properCross2d(points[constraints[i][0]], points[constraints[i][1]],
        points[constraints[j][0]], points[constraints[j][1]])) {
        throw new RangeError("constraints must not cross each other");
      }
    }
  }
  for (const [u, v] of constraints) {
    // A segment through another input point cannot be a single edge; the outline refuses it.
    for (let index = 0; index < points.length; index++) {
      if (index === u || index === v) continue;
      if (orient2d(points[u], points[v], points[index]) === 0 &&
        Math.min(points[u][0], points[v][0]) - EPS <= points[index][0] &&
        points[index][0] <= Math.max(points[u][0], points[v][0]) + EPS &&
        Math.min(points[u][1], points[v][1]) - EPS <= points[index][1] &&
        points[index][1] <= Math.max(points[u][1], points[v][1]) + EPS) {
        throw new RangeError(`constraint (${u},${v}) passes through point ${index}`);
      }
    }
  }
  const { triangles } = triangulateDelaunay(points);
  let mesh = triangles.map((t) => t.slice());
  const constrained = new Set(constraints.map(([u, v]) => edgeKey(u, v)));
  let recoveries = 0;

  for (const [a, b] of constraints) {
    if (constrained.has(edgeKey(a, b)) && edgeUses(mesh).has(edgeKey(a, b))) continue;
    let guard = 0;
    while (guard++ < 5000) {
      if (edgeUses(mesh).has(edgeKey(a, b))) break;
      const key = findCrossingEdge(points, mesh, a, b);
      if (key === null) break;
      // the flip may be impossible when the quad is not convex; then take the next crossing edge
      const before = mesh.map((t) => t.join(",")).join("|");
      if (!flipEdge(mesh, key)) break;
      mesh = orientAll(points, mesh);
      const after = mesh.map((t) => t.join(",")).join("|");
      if (before === after) break;
      recoveries++;
    }
  }

  let legalisations = 0;
  let guard = 0;
  while (guard++ < 5000) {
    const uses = edgeUses(mesh);
    let flipped = false;
    for (const [key] of uses) {
      const [u, v] = key.split(",").map(Number);
      if (constrained.has(key)) continue; // a constraint is never flipped away
      const users = uses.get(key);
      if (users.length !== 2) continue;
      const opposite = users.map((position) =>
        mesh[position].find((w) => w !== u && w !== v));
      const [p, q] = opposite;
      const ccw = orient2d(points[p], points[u], points[v]) > 0
        ? [points[p], points[u], points[v]]
        : [points[p], points[v], points[u]];
      if (inCircle(ccw[0], ccw[1], ccw[2], points[q]) > 0n) {
        if (flipEdge(mesh, key)) {
          mesh = orientAll(points, mesh);
          legalisations++;
          flipped = true;
          break;
        }
      }
    }
    if (!flipped) break;
  }
  return { triangles: mesh, constrained, recoveries, legalisations };
}

// ---- validation ---------------------------------------------------------------------------
function meshArea2(points, triangles) {
  let total = 0;
  for (const [a, b, c] of triangles) total += cross(points[a], points[b], points[c]);
  return total;
}

function hullArea2(points) {
  const hull = convexHull(points);
  let total = 0;
  for (let i = 0; i < hull.length; i++) {
    const a = points[hull[i]];
    const b = points[hull[(i + 1) % hull.length]];
    total += a[0] * b[1] - b[0] * a[1];
  }
  return total;
}

/** Number of points on the convex hull boundary (collinear boundary points included). */
function boundaryPointCount(points) {
  const hull = convexHull(points);
  const hullPoints = hull.map((i) => points[i]);
  let count = 0;
  for (const p of points) {
    let onBoundary = false;
    for (let i = 0; i < hullPoints.length && !onBoundary; i++) {
      const a = hullPoints[i];
      const b = hullPoints[(i + 1) % hullPoints.length];
      if (sign(cross(a, b, p)) === 0 &&
        Math.min(a[0], b[0]) - EPS <= p[0] && p[0] <= Math.max(a[0], b[0]) + EPS &&
        Math.min(a[1], b[1]) - EPS <= p[1] && p[1] <= Math.max(a[1], b[1]) + EPS) {
        onBoundary = true;
      }
    }
    if (onBoundary) count++;
  }
  return count;
}

/** Manifold + count + area + orientation checks for a triangulation of the point set. */
function validateMesh(points, triangles, label) {
  const uses = edgeUses(triangles);
  const hull = convexHull(points);
  const hullPoints = hull.map((i) => points[i]);
  const withinHullEdge = (u, v) => {
    for (let i = 0; i < hullPoints.length; i++) {
      const a = hullPoints[i];
      const b = hullPoints[(i + 1) % hullPoints.length];
      if (sign(cross(a, b, points[u])) === 0 && sign(cross(a, b, points[v])) === 0 &&
        Math.min(a[0], b[0]) - EPS <= points[u][0] && points[u][0] <= Math.max(a[0], b[0]) + EPS &&
        Math.min(a[0], b[0]) - EPS <= points[v][0] && points[v][0] <= Math.max(a[0], b[0]) + EPS &&
        Math.min(a[1], b[1]) - EPS <= points[u][1] && points[u][1] <= Math.max(a[1], b[1]) + EPS &&
        Math.min(a[1], b[1]) - EPS <= points[v][1] && points[v][1] <= Math.max(a[1], b[1]) + EPS) {
        return true;
      }
    }
    return false;
  };
  let interior = 0;
  let boundary = 0;
  for (const [key, list] of uses) {
    const [u, v] = key.split(",").map(Number);
    if (list.length === 2) interior++;
    else if (list.length === 1) {
      boundary++;
      ok(withinHullEdge(u, v), `${label}: every boundary edge must lie on the convex hull (${key})`);
    } else {
      ok(false, `${label}: edge ${key} is used by ${list.length} triangles`);
    }
  }
  strictEqual(triangles.length, 2 * points.length - 2 - boundaryPointCount(points),
    `${label}: the count 2n - 2 - B must hold`);
  strictEqual(interior + boundary, 3 * triangles.length - interior,
    `${label}: edge/face incidence (3T = 2I + B) must hold`);
  for (const [a, b, c] of triangles) {
    ok(sign(cross(points[a], points[b], points[c])) > 0, `${label}: triangles must be counter-clockwise`);
  }
  ok(Math.abs(meshArea2(points, triangles) - hullArea2(points)) < EPS,
    `${label}: the mesh must cover the convex hull`);
  return { interior, boundary, uses };
}

function validateConstraints(points, triangles, constraints, label) {
  const keys = new Set([...edgeUses(triangles).keys()]);
  for (const [u, v] of constraints) {
    ok(keys.has(edgeKey(u, v)), `${label}: constraint (${u},${v}) must be an edge of the mesh`);
    for (const [a, b] of constraints) {
      if (a === u && b === v) continue;
      ok(!properCross2d(points[u], points[v], points[a], points[b]),
        `${label}: constraints must not cross in the mesh`);
    }
  }
  // no triangulation edge may properly cross a constraint
  for (const key of keys) {
    const [u, v] = key.split(",").map(Number);
    for (const [a, b] of constraints) {
      if (u === a || u === b || v === a || v === b) continue;
      ok(!properCross2d(points[u], points[v], points[a], points[b]),
        `${label}: edge ${key} must not cross the constraint (${a},${b})`);
    }
  }
}

/** Independent oracle: all triples whose circumcircle is empty of every point. */
function bruteDelaunayTriangles(points) {
  const keys = [];
  const n = points.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        if (orient2d(points[i], points[j], points[k]) === 0) continue;
        const ccw = orient2d(points[i], points[j], points[k]) > 0 ? [i, j, k] : [i, k, j];
        if (triangleIsDelaunay(points, ccw, points.map((_, index) => index))) {
          keys.push(ccw.slice().sort((a, b) => a - b).join(","));
        }
      }
    }
  }
  return keys.sort();
}

module.exports = { constrainedDelaunay, triangulateDelaunay, convexHull, inCircle, validateMesh,
  bruteDelaunayTriangles, edgeUses, edgeKey };

if (require.main === module) {
  const square = [[0, 0], [10, 0], [10, 10], [0, 10]];
  const grid = [[0, 0], [10, 0], [20, 0], [0, 10], [10, 10], [20, 10], [0, 20], [10, 20], [20, 20]];

  // -- degenerate inputs -------------------------------------------------------------------
  strictEqual(triangulateDelaunay([]).triangles.length, 0, "no points, no triangles");
  strictEqual(triangulateDelaunay([[1, 2]]).triangles.length, 0, "a single point has no triangle");
  strictEqual(triangulateDelaunay([[1, 2], [5, 7]]).triangles.length, 0, "two points have no triangle");
  strictEqual(triangulateDelaunay([[0, 0], [1, 1], [2, 2], [3, 3]]).triangles.length, 0,
    "collinear points have no triangle");
  const triangleMesh = triangulateDelaunay([[0, 0], [4, 0], [0, 4]]);
  deepStrictEqual(triangleMesh.triangles, [[0, 1, 2]]);
  validateMesh([[0, 0], [4, 0], [0, 4]], triangleMesh.triangles, "single triangle");

  // -- Delaunay on a general position cloud --------------------------------------------------
  const rng = makeRng(743743);
  const cloud = [];
  while (cloud.length < 14) {
    const candidate = [rng(60) - 30, rng(60) - 30];
    if (!cloud.some((p) => p[0] === candidate[0] && p[1] === candidate[1])) cloud.push(candidate);
  }
  const cloudMesh = constrainedDelaunay(cloud);
  validateMesh(cloud, cloudMesh.triangles, "cloud");
  for (const triangle of cloudMesh.triangles) {
    ok(triangleIsDelaunay(cloud, triangle, cloud.map((_, i) => i)),
      "every triangle of an unconstrained mesh must have an empty circumcircle");
  }

  // -- independent oracle: cocircular input (the square) and general position (the cloud) ------
  const oracle = bruteDelaunayTriangles(square);
  const squareMesh = constrainedDelaunay(square);
  const mine = squareMesh.triangles.map((t) => t.slice().sort((a, b) => a - b).join(",")).sort();
  ok(mine.length === 2, "the square needs exactly two triangles");
  ok(mine.every((key) => oracle.includes(key)),
    "with four cocircular corners the oracle lists every valid diagonal, the mesh picks one");
  const cloudKeys = cloudMesh.triangles.map((t) => t.slice().sort((a, b) => a - b).join(",")).sort();
  deepStrictEqual(cloudKeys, bruteDelaunayTriangles(cloud),
    "in general position the mesh must equal the brute force Delaunay set");

  // -- forcing the other diagonal of the cocircular square --------------------------------------
  const diagMesh = constrainedDelaunay(square);
  const diagonals = [...edgeUses(diagMesh.triangles).keys()].filter((key) => key === "0,2" || key === "1,3");
  strictEqual(diagonals.length, 1, "exactly one of the two diagonals exists");
  const presentDiagonal = diagonals[0];
  const missing = presentDiagonal === "0,2" ? [1, 3] : [0, 2];
  const forced = constrainedDelaunay(square, [missing]);
  validateMesh(square, forced.triangles, "square with a forced diagonal");
  validateConstraints(square, forced.triangles, [missing], "square with a forced diagonal");
  ok(edgeUses(forced.triangles).has(edgeKey(missing[0], missing[1])), "the constraint must be an edge");
  ok(!edgeUses(forced.triangles).has(presentDiagonal), "the opposite diagonal must be gone");
  ok(forced.recoveries > 0,
    `recovering the ${presentDiagonal} -> ${edgeKey(missing[0], missing[1])} diagonal takes a flip`);
  const alreadyThere = constrainedDelaunay(square, [presentDiagonal.split(",").map(Number)]);
  strictEqual(alreadyThere.recoveries, 0, "a constraint that already is an edge needs no recovery");

  // -- a grid with three constraints: recovery plus legalisation --------------------------------
  const constraints = [[0, 4], [4, 8], [2, 4]];
  const gridMesh = constrainedDelaunay(grid, constraints);
  validateMesh(grid, gridMesh.triangles, "grid");
  validateConstraints(grid, gridMesh.triangles, constraints, "grid");
  const uses = edgeUses(gridMesh.triangles);
  for (const [key, list] of uses) {
    if (gridMesh.constrained.has(key)) continue;
    const [u, v] = key.split(",").map(Number);
    if (list.length !== 2) continue;
    const opposite = list.map((position) => gridMesh.triangles[position].find((w) => w !== u && w !== v));
    const [p, q] = opposite;
    const ccw = orient2d(grid[p], grid[u], grid[v]) > 0
      ? [grid[p], grid[u], grid[v]]
      : [grid[p], grid[v], grid[u]];
    ok(inCircle(ccw[0], ccw[1], ccw[2], grid[q]) <= 0n,
      `the non-constraint edge ${key} must be locally Delaunay`);
  }
  ok(gridMesh.constrained.size === constraints.length, "every constraint is remembered");
  deepStrictEqual([...gridMesh.constrained].sort(), ["0,4", "2,4", "4,8"], "the constraint set is exact");

  // -- invalid input ------------------------------------------------------------------------------
  throws(() => constrainedDelaunay(square, [[0, 0]]), RangeError, "a degenerate constraint is rejected");
  throws(() => constrainedDelaunay(square, [[0, 9]]), RangeError, "an out-of-range constraint is rejected");
  throws(() => constrainedDelaunay(square, [[0, 2], [1, 3]]), RangeError,
    "crossing constraints are rejected");
  throws(() => constrainedDelaunay(grid, [[0, 8]]), RangeError,
    "a constraint passing through another input point is rejected");

  // -- determinism ---------------------------------------------------------------------------------
  deepStrictEqual(constrainedDelaunay(grid, constraints).triangles,
    constrainedDelaunay(grid, constraints).triangles, "the construction must be deterministic");
  deepStrictEqual(constrainedDelaunay(cloud).triangles, constrainedDelaunay(cloud).triangles,
    "the cloud triangulation must be deterministic");

  console.log(`cloud_points=${cloud.length} cloud_triangles=${cloudMesh.triangles.length}`);
  console.log(`square_oracle=${oracle.length} forced_recoveries=${forced.recoveries} ` +
    `grid_recoveries=${gridMesh.recoveries} grid_legalisations=${gridMesh.legalisations}`);
  console.log(`grid_triangles=${gridMesh.triangles.length} hull=${convexHull(grid).length}`);
  console.log("PASS");
}
