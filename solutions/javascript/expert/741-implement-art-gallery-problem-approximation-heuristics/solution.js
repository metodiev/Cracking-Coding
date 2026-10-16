/** Task 741 - Implement Art Gallery problem approximation heuristics.
 *  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Two deterministic heuristics over an ear-clipping triangulation of a simple polygon.
 * (1) Fisk's 3-colouring: the triangulation's dual is a tree, so a DFS assigns one of three
 *     colours to every polygon vertex such that each triangle carries all three colours (the
 *     shared edge of a neighbouring triangle already owns two colours, so the third vertex gets
 *     the remaining one). Every triangle is then watched by its colour-i vertex, hence the
 *     smallest colour class is a valid guard set of size at most floor(n/3).
 * (2) A greedy cover: repeatedly place the guard that geometrically sees the most still-unwatched
 *     triangles (visibility checked in the polygon: no proper edge crossing, no polygon vertex
 *     inside the segment, midpoint interior), ties broken by the lowest vertex index.
 *
 * What is simplified: the polygon must be simple (no holes) and is triangulated by ear clipping,
 * the greedy cover falls back on the low-index vertex when it cannot improve, and the coverage
 * test uses the standard sufficient criterion "the guard sees all three vertices of the
 * triangle". What is verified: the triangulation has n-2 triangles and covers the polygon area
 * exactly (integer doubled areas), every triangle is positively oriented with its centroid
 * strictly inside, the colouring gives each triangle three distinct colours, each colour class
 * really covers every triangle geometrically, both heuristic guard sets satisfy the floor(n/3)
 * bound, a convex polygon needs exactly one guard and a four-toothed comb needs more than one,
 * and the whole pipeline is deterministic.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

const EPS = 1e-9;
const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
const sign = (value) => (value > EPS ? 1 : value < -EPS ? -1 : 0);

function onSegment(a, b, p) {
  return sign(cross(a, b, p)) === 0 &&
    Math.min(a[0], b[0]) - EPS <= p[0] && p[0] <= Math.max(a[0], b[0]) + EPS &&
    Math.min(a[1], b[1]) - EPS <= p[1] && p[1] <= Math.max(a[1], b[1]) + EPS;
}

function properCross(a, b, c, d) {
  return sign(cross(c, d, a)) * sign(cross(c, d, b)) < 0 &&
    sign(cross(a, b, c)) * sign(cross(a, b, d)) < 0;
}

/** 1 = strictly inside, 0 = outside, 2 = on the boundary. */
function pointInPolygon(p, polygon) {
  let inside = false;
  const n = polygon.length;
  for (let i = 0; i < n; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % n];
    if (onSegment(a, b, p)) return 2;
    if ((a[1] > p[1]) !== (b[1] > p[1])) {
      const x = a[0] + ((p[1] - a[1]) * (b[0] - a[0])) / (b[1] - a[1]);
      if (x > p[0]) inside = !inside;
    }
  }
  return inside ? 1 : 0;
}

/** Twice the signed area (positive for a counter-clockwise polygon). */
function doubledArea(points) {
  let total = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    total += a[0] * b[1] - b[0] * a[1];
  }
  return total;
}

/** Does the segment (p, q) stay inside the polygon's closed region? (A line of sight along an
 *  edge is legitimate: the polygon is simple and hole-free, so its boundary is part of it.) */
function segmentSees(p, q, polygon) {
  const mid = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
  if (pointInPolygon(mid, polygon) === 0) return false;
  const n = polygon.length;
  for (let i = 0; i < n; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % n];
    if (properCross(p, q, a, b)) return false;
    if (onSegment(p, q, a) && !((Math.abs(a[0] - p[0]) < EPS && Math.abs(a[1] - p[1]) < EPS) ||
      (Math.abs(a[0] - q[0]) < EPS && Math.abs(a[1] - q[1]) < EPS))) return false;
  }
  return true;
}

function strictlyInsideTriangle(p, a, b, c) {
  return sign(cross(a, b, p)) > 0 && sign(cross(b, c, p)) > 0 && sign(cross(c, a, p)) > 0;
}

/** Ear clipping: returns triangles as index triples, counter-clockwise. */
function earClip(polygon) {
  const n = polygon.length;
  if (n < 3) throw new RangeError("a polygon needs at least three vertices");
  let remaining = [...Array(n).keys()];
  const triangles = [];
  while (remaining.length > 3) {
    let found = -1;
    for (let i = 0; i < remaining.length; i++) {
      const size = remaining.length;
      const prev = polygon[remaining[(i - 1 + size) % size]];
      const cur = polygon[remaining[i]];
      const next = polygon[remaining[(i + 1) % size]];
      if (sign(cross(prev, cur, next)) <= 0) continue; // reflex or collinear: not an ear tip
      const base = [prev, next];
      let blocked = false;
      for (const index of remaining) {
        const point = polygon[index];
        if (index === remaining[i] || index === remaining[(i - 1 + size) % size] ||
          index === remaining[(i + 1) % size]) continue;
        if (strictlyInsideTriangle(point, prev, cur, next) || onSegment(base[0], base[1], point)) {
          blocked = true;
          break;
        }
      }
      if (!blocked) {
        found = i;
        break;
      }
    }
    if (found < 0) throw new Error("no ear found: the polygon must be simple and counter-clockwise");
    const size = remaining.length;
    triangles.push([
      remaining[(found - 1 + size) % size],
      remaining[found],
      remaining[(found + 1) % size],
    ]);
    remaining.splice(found, 1);
  }
  triangles.push(remaining.slice());
  return triangles;
}

/** Dual adjacency: two triangles are neighbours when they share an edge. */
function dualAdjacency(triangles) {
  const adjacency = triangles.map(() => []);
  for (let i = 0; i < triangles.length; i++) {
    for (let j = i + 1; j < triangles.length; j++) {
      const shared = triangles[i].filter((v) => triangles[j].includes(v));
      if (shared.length === 2) {
        adjacency[i].push(j);
        adjacency[j].push(i);
      }
    }
  }
  return adjacency;
}

/** Fisk's three colouring; returns { colors, classes } with classes sorted by size. */
function fiskColoring(vertexCount, triangles) {
  const colors = new Array(vertexCount).fill(-1);
  colors[triangles[0][0]] = 0;
  colors[triangles[0][1]] = 1;
  colors[triangles[0][2]] = 2;
  const adjacency = dualAdjacency(triangles);
  const seen = new Array(triangles.length).fill(false);
  const stack = [0];
  seen[0] = true;
  while (stack.length > 0) {
    const current = stack.pop();
    for (const neighbour of adjacency[current]) {
      const triangle = triangles[neighbour];
      const missing = triangle.filter((v) => colors[v] === -1);
      if (missing.length > 0) {
        const used = new Set(triangle.map((v) => colors[v]).filter((c) => c >= 0));
        colors[missing[0]] = [0, 1, 2].find((c) => !used.has(c));
      }
      if (!seen[neighbour]) {
        seen[neighbour] = true;
        stack.push(neighbour);
      }
    }
  }
  if (colors.some((c) => c < 0)) throw new Error("the dual graph must be connected (it is a tree)");
  const classes = [0, 1, 2]
    .map((c) => colors.map((value, index) => (value === c ? index : -1)).filter((i) => i >= 0))
    .sort((a, b) => a.length - b.length);
  return { colors, classes };
}

/** A guard watches a triangle when it sees all three of its vertices. */
function guardCovers(guard, triangle, polygon) {
  return triangle.every((v) => segmentSees(polygon[guard], polygon[v], polygon));
}

function greedyGuards(vertexCount, triangles, polygon) {
  const uncovered = new Set(triangles.map((_, i) => i));
  const guards = [];
  while (uncovered.size > 0) {
    let best = -1;
    let bestCount = -1;
    for (let v = 0; v < vertexCount; v++) {
      let count = 0;
      for (const index of uncovered) if (guardCovers(v, triangles[index], polygon)) count++;
      if (count > bestCount) {
        bestCount = count;
        best = v;
      }
    }
    guards.push(best);
    for (const index of [...uncovered]) if (guardCovers(best, triangles[index], polygon)) uncovered.delete(index);
  }
  return guards;
}

/** Independent check: does the guard set really watch every triangle geometrically? */
function coverageHolds(guards, triangles, polygon) {
  return triangles.every((triangle) => guards.some((g) => guardCovers(g, triangle, polygon)));
}

function triangulationCheck(polygon, triangles) {
  strictEqual(triangles.length, polygon.length - 2, "a triangulation has n - 2 triangles");
  let total = 0;
  for (const triangle of triangles) {
    const [a, b, c] = triangle.map((i) => polygon[i]);
    const area = cross(a, b, c);
    ok(area > EPS, "every triangle must be positively oriented and non-degenerate");
    total += area;
    const centroid = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3];
    strictEqual(pointInPolygon(centroid, polygon), 1, "every triangle centroid must be inside the polygon");
  }
  ok(Math.abs(total - doubledArea(polygon)) < 1e-6, "the triangles must cover the polygon area exactly");
}

module.exports = { earClip, fiskColoring, greedyGuards, guardCovers, coverageHolds, pointInPolygon };

if (require.main === module) {
  const square = [[0, 0], [10, 0], [10, 10], [0, 10]];
  const comb = [
    [0, 0], [40, 0], [40, 30], [35, 30], [35, 10], [30, 10], [30, 30], [25, 30], [25, 10], [20, 10],
    [20, 30], [15, 30], [15, 10], [10, 10], [10, 30], [5, 30], [5, 10], [0, 10],
  ];
  const arrowhead = [[0, 0], [10, 4], [20, 0], [20, 12], [0, 12]];
  const lShape = [[0, 0], [12, 0], [12, 4], [4, 4], [4, 12], [0, 12]];
  const dart = [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]];

  // -- triangulation -----------------------------------------------------------------------
  for (const [name, polygon] of [["square", square], ["comb", comb], ["arrow", arrowhead], ["L", lShape],
    ["dart", dart]]) {
    ok(doubledArea(polygon) > 0, `${name} must be given counter-clockwise`);
    const triangles = earClip(polygon);
    triangulationCheck(polygon, triangles);
  }
  throws(() => earClip([[0, 0], [1, 1]]), RangeError, "fewer than three vertices is rejected");
  throws(() => earClip([[0, 0], [0, 10], [10, 10], [10, 0]]), Error,
    "a clockwise outline has no ear tip and must be rejected");

  // -- the comb: bound, cover and comparison -----------------------------------------------
  const combTriangles = earClip(comb);
  const combColoring = fiskColoring(comb.length, combTriangles);
  const combTrianglesColors = combTriangles.map((t) => t.map((v) => combColoring.colors[v]));
  for (const [i, colors] of combTrianglesColors.entries()) {
    strictEqual(new Set(colors).size, 3, `triangle ${i} must carry all three colours`);
  }
  const bound = Math.floor(comb.length / 3);
  const smallestClass = combColoring.classes[0];
  ok(smallestClass.length <= bound, `Fisk guards ${smallestClass.length} must respect floor(n/3) = ${bound}`);
  ok(smallestClass.length >= 1, "at least one guard is needed");
  ok(coverageHolds(smallestClass, combTriangles, comb), "the colour class must really watch every triangle");
  const combGreedy = greedyGuards(comb.length, combTriangles, comb);
  ok(coverageHolds(combGreedy, combTriangles, comb), "the greedy guards must watch every triangle");
  ok(combGreedy.length <= bound, `greedy guards ${combGreedy.length} must respect floor(n/3) = ${bound}`);
  ok(combGreedy.length >= 2, "a four-toothed comb cannot be watched from a single point");
  ok(new Set(combGreedy).size === combGreedy.length, "a guard is never placed twice");

  // -- convex polygons need exactly one guard ----------------------------------------------
  const convex = [[0, 0], [20, 0], [26, 12], [10, 22], [-6, 12]];
  const convexTriangles = earClip(convex);
  triangulationCheck(convex, convexTriangles);
  const convexGreedy = greedyGuards(convex.length, convexTriangles, convex);
  strictEqual(convexGreedy.length, 1, "a convex polygon is watched by one vertex");
  ok(coverageHolds(convexGreedy, convexTriangles, convex), "that single guard must watch everything");
  const convexColoring = fiskColoring(convex.length, convexTriangles);
  ok(coverageHolds(convexColoring.classes[0], convexTriangles, convex), "Fisk also covers the convex case");
  ok(convexColoring.classes[0].length <= Math.floor(convex.length / 3), "Fisk bound holds for the convex polygon");

  // -- triangle and L shape ------------------------------------------------------------------
  const triangle = [[0, 0], [10, 0], [0, 10]];
  const triangleTriangles = earClip(triangle);
  strictEqual(triangleTriangles.length, 1, "a triangle triangulates to itself");
  deepStrictEqual(triangleTriangles, [[0, 1, 2]], "the single triangle keeps the vertex order");
  strictEqual(greedyGuards(3, triangleTriangles, triangle).length, 1, "a triangle needs one guard");
  const lTriangles = earClip(lShape);
  triangulationCheck(lShape, lTriangles);
  const lGreedy = greedyGuards(lShape.length, lTriangles, lShape);
  ok(coverageHolds(lGreedy, lTriangles, lShape), "the L shape must be covered");
  ok(lGreedy.length <= Math.floor(lShape.length / 3), "the L shape must respect the bound");
  ok(lGreedy.every((v) => v >= 0 && v < lShape.length), "guard indices must be polygon vertices");

  // -- visibility sanity ---------------------------------------------------------------------
  ok(segmentSees(lShape[0], lShape[2], lShape), "the L shape's two far corners see each other");
  ok(!segmentSees(lShape[1], lShape[5], lShape), "the missing corner cuts the opposite diagonal");
  ok(segmentSees(lShape[0], lShape[4], lShape), "the reflex corner is on the diagonal from (0,0)");
  ok(segmentSees(comb[1], comb[2], comb), "neighbouring comb vertices see each other");
  ok(!segmentSees(comb[0], comb[6], comb), "a tooth blocks the sight line to a distant tooth tip");
  ok(segmentSees(comb[0], comb[4], comb), "the base of the comb is fully visible");
  ok(guardCovers(0, [0, 1, 2], square), "a square vertex watches its own triangle");
  ok(!guardCovers(0, [4, 5, 6], comb), "a distant guard cannot watch a comb tooth triangle");

  // -- determinism ---------------------------------------------------------------------------
  deepStrictEqual(earClip(comb), combTriangles, "ear clipping must be deterministic");
  deepStrictEqual(fiskColoring(comb.length, combTriangles), combColoring, "the colouring must be deterministic");
  deepStrictEqual(greedyGuards(comb.length, combTriangles, comb), combGreedy, "greedy must be deterministic");

  console.log(`comb_vertices=${comb.length} triangles=${combTriangles.length} fisk=${smallestClass.length} ` +
    `greedy=${combGreedy.length} bound=${bound}`);
  console.log(`convex_guards=${convexGreedy.length} lshape_guards=${lGreedy.length} comb_classes=` +
    combColoring.classes.map((c) => c.length).join(","));
  console.log("PASS");
}
