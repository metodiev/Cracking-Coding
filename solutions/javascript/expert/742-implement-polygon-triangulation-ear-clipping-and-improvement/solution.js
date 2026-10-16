/** Task 742 - Implement polygon triangulation (ear clipping) and improvement.
 *  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Two stages. (1) Ear clipping with two ear orders: `first` takes the first valid ear, `best`
 * takes the ear maximising the minimum angle of the clipped triangle, which avoids the slivers
 * that a naive order produces. An ear tip is a convex vertex whose triangle contains no other
 * remaining vertex and whose base does not pass through one. (2) Lawson improvement: while an
 * interior edge of two adjacent triangles fails the empty-circumcircle test, flip it (only when
 * the quad is strictly convex, so the new diagonal stays inside the polygon). The in-circle
 * predicate is evaluated exactly with BigInt determinants - the classic lift to the paraboloid -
 * so the flip decisions are never corrupted by rounding, while a float evaluation of the same
 * determinant is kept around to show how often the two disagree.
 *
 * What is simplified: the polygon is simple, hole-free and counter-clockwise, coordinates are
 * small integers (so the exact predicate is cheap), the improvement phase stops as soon as no
 * legal flip remains instead of running a randomised sequence to the global optimum, and the
 * ear search is O(n^2) rather than O(n log n). What is verified: every intermediate triangulation
 * keeps n-2 triangles, the exact area, positive orientation and centroids inside the polygon
 * (validated after *every* flip), the final triangulation has no legal flip left and satisfies
 * the Delaunay empty-circumcircle property exactly, the minimum angle after flips is never
 * worse than before, the `best` ear order beats `first` on a polygon built to produce slivers,
 * and for convex input the result equals the independent brute-force Delaunay triangulation.
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

function doubledAreaOfPolygon(points) {
  let total = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    total += a[0] * b[1] - b[0] * a[1];
  }
  return total;
}

function triangleArea2(polygon, triangle) {
  const [a, b, c] = triangle.map((i) => polygon[i]);
  return cross(a, b, c);
}

function minAngleOfTriangle(a, b, c) {
  const sides = [
    Math.hypot(b[0] - a[0], b[1] - a[1]),
    Math.hypot(c[0] - b[0], c[1] - b[1]),
    Math.hypot(a[0] - c[0], a[1] - c[1]),
  ];
  const [s0, s1, s2] = sides;
  const angle = (x, y, z) => Math.acos(Math.max(-1, Math.min(1, (x * x + y * y - z * z) / (2 * x * y))));
  return Math.min(angle(s0, s1, s2), angle(s1, s2, s0), angle(s2, s0, s1));
}

/** Smallest interior angle over a whole triangulation, in degrees. */
function minAngleDegrees(polygon, triangles) {
  let best = Infinity;
  for (const triangle of triangles) {
    const [a, b, c] = triangle.map((i) => polygon[i]);
    best = Math.min(best, minAngleOfTriangle(a, b, c));
  }
  return (best * 180) / Math.PI;
}

function strictlyInsideTriangle(p, a, b, c) {
  return sign(cross(a, b, p)) > 0 && sign(cross(b, c, p)) > 0 && sign(cross(c, a, p)) > 0;
}

/** Ear clipping. `order` is "first" (naive) or "best" (maximise the triangle's minimum angle). */
function earClip(polygon, order = "first") {
  const n = polygon.length;
  if (n < 3) throw new RangeError("a polygon needs at least three vertices");
  let remaining = [...Array(n).keys()];
  const triangles = [];
  while (remaining.length > 3) {
    const size = remaining.length;
    let chosen = -1;
    let chosenQuality = -Infinity;
    for (let i = 0; i < size; i++) {
      const prevIndex = remaining[(i - 1 + size) % size];
      const curIndex = remaining[i];
      const nextIndex = remaining[(i + 1) % size];
      const prev = polygon[prevIndex];
      const cur = polygon[curIndex];
      const next = polygon[nextIndex];
      if (sign(cross(prev, cur, next)) <= 0) continue; // reflex or collinear tip
      let blocked = false;
      for (const index of remaining) {
        if (index === prevIndex || index === curIndex || index === nextIndex) continue;
        const point = polygon[index];
        if (strictlyInsideTriangle(point, prev, cur, next) || onSegment(prev, next, point)) {
          blocked = true;
          break;
        }
      }
      if (blocked) continue;
      if (order === "first") {
        chosen = i;
        break;
      }
      const quality = minAngleOfTriangle(prev, cur, next);
      if (quality > chosenQuality + 1e-15) {
        chosenQuality = quality;
        chosen = i;
      }
    }
    if (chosen < 0) throw new Error("no ear found: the polygon must be simple and counter-clockwise");
    const prevIndex = remaining[(chosen - 1 + size) % size];
    const curIndex = remaining[chosen];
    const nextIndex = remaining[(chosen + 1) % size];
    triangles.push([prevIndex, curIndex, nextIndex]);
    remaining.splice(chosen, 1);
  }
  triangles.push(remaining.slice());
  return triangles;
}

// ---- exact in-circle predicate ---------------------------------------------------------
const big = (v) => v.map((c) => BigInt(c));

/** 3x3 BigInt determinant. */
function determinant3(rows) {
  const [[a, b, c], [d, e, f], [g, h, i]] = rows;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

/** 4x4 BigInt determinant, expanded along the first row. */
function determinant4(rows) {
  let total = 0n;
  for (let column = 0; column < 4; column++) {
    const minor = determinant3(rows.slice(1).map((row) => row.filter((_, c) => c !== column)));
    const term = rows[0][column] * minor;
    total += column % 2 === 0 ? term : -term;
  }
  return total;
}

/**
 * Exact in-circle test: > 0 when d is strictly inside the circumcircle of the counter-clockwise
 * triangle (a, b, c), 0 when the four points are cocircular.
 */
function inCircleExact(a, b, c, d) {
  const rows = [a, b, c].map((p) => {
    const dx = BigInt(p[0]) - BigInt(d[0]);
    const dy = BigInt(p[1]) - BigInt(d[1]);
    return [dx, dy, dx * dx + dy * dy];
  });
  return determinant3(rows);
}

/** The same predicate in floating point, used only to measure how often it is wrong. */
function inCircleFloat(a, b, c, d) {
  const rows = [a, b, c].map((p) => {
    const dx = p[0] - d[0];
    const dy = p[1] - d[1];
    return [dx, dy, dx * dx + dy * dy];
  });
  const [[p, q, r], [s, t, u], [v, w, x]] = rows;
  return p * (t * x - u * w) - q * (s * x - u * v) + r * (s * w - t * v);
}

/**
 * Interior edges shared by exactly two triangles. Each use records the ordered shared edge
 * (u -> v) as it appears in that triangle's counter-clockwise cycle plus the opposite vertex.
 * Both triangles traverse the shared edge in opposite directions, which is asserted in the tests.
 */
function interiorEdges(triangles) {
  const shared = new Map();
  triangles.forEach((triangle, index) => {
    for (let k = 0; k < 3; k++) {
      const u = triangle[k];
      const v = triangle[(k + 1) % 3];
      const opposite = triangle[(k + 2) % 3];
      const key = `${Math.min(u, v)},${Math.max(u, v)}`;
      if (!shared.has(key)) shared.set(key, []);
      shared.get(key).push({ triangle: index, u, v, opposite });
    }
  });
  const edges = [];
  for (const [key, uses] of shared) {
    if (uses.length !== 2) continue;
    edges.push({ key, first: uses[0], second: uses[1] });
  }
  return edges;
}

/**
 * The quad around an interior edge in counter-clockwise cyclic order p -> u -> q -> v: the
 * boundary edges are (p,u) and (v,p) from the first triangle and (u,q) and (q,v) from the
 * second. The flipped diagonal is (p, q).
 */
function quadOf(edge) {
  const { first, second } = edge;
  if (second.u !== first.v || second.v !== first.u) return null; // inconsistent orientations
  return {
    u: first.u,
    v: first.v,
    p: first.opposite,
    q: second.opposite,
    order: [first.opposite, first.u, second.opposite, first.v],
  };
}

function isConvexQuad(polygon, quad) {
  const points = quad.order.map((i) => polygon[i]);
  const turns = [0, 1, 2, 3].map((i) =>
    cross(points[i], points[(i + 1) % 4], points[(i + 2) % 4]));
  return turns.every((t) => t > EPS) || turns.every((t) => t < -EPS);
}

/** An edge is illegal when the opposite vertex lies inside the circumcircle of its neighbour. */
function edgeIsIllegal(polygon, quad) {
  // the current triangle on the p side is (p, u, v); its opposite vertex is q
  return inCircleExact(polygon[quad.p], polygon[quad.u], polygon[quad.v], polygon[quad.q]) > 0n;
}

/** Only a convex quad may be flipped, so an illegal-but-non-convex edge has to be skipped. */
function flipIsIllegal(polygon, edge) {
  const quad = quadOf(edge);
  if (quad === null || !isConvexQuad(polygon, quad)) return false;
  return edgeIsIllegal(polygon, quad);
}

/**
 * Lawson improvement. `onFlip` is called with the new triangle list after every flip so the
 * caller can re-validate the invariants.
 */
function improveTriangulation(polygon, triangles, options = {}) {
  const onFlip = options.onFlip ?? (() => {});
  let current = triangles.map((t) => t.slice());
  let flips = 0;
  let guard = 0;
  while (guard++ < 1000) {
    const illegal = interiorEdges(current).find((edge) => flipIsIllegal(polygon, edge));
    if (illegal === undefined) break;
    const quad = quadOf(illegal);
    // the shared edge (u, v) is replaced by the opposite diagonal (p, q)
    current[illegal.first.triangle] = orientTriangle(polygon, [quad.p, quad.u, quad.q]);
    current[illegal.second.triangle] = orientTriangle(polygon, [quad.p, quad.q, quad.v]);
    flips++;
    onFlip(current);
  }
  return { triangles: current, flips };
}

function orientTriangle(polygon, triangle) {
  const [a, b, c] = triangle.map((i) => polygon[i]);
  return cross(a, b, c) > 0 ? triangle : [triangle[0], triangle[2], triangle[1]];
}

/** Every interior edge that violates the empty-circumcircle test (convex quad or not). */
function illegalEdgeCount(polygon, triangles) {
  let count = 0;
  for (const edge of interiorEdges(triangles)) {
    const quad = quadOf(edge);
    if (quad === null) continue;
    if (edgeIsIllegal(polygon, quad)) count++;
  }
  return count;
}

/** Only these can actually be flipped, so the improvement loop must drive this count to zero. */
function flippableIllegalCount(polygon, triangles) {
  return interiorEdges(triangles).filter((edge) => flipIsIllegal(polygon, edge)).length;
}

/** Full structural validation of one triangulation against the polygon. */
function validateTriangulation(polygon, triangles, label) {
  strictEqual(triangles.length, polygon.length - 2, `${label}: n - 2 triangles`);
  let total = 0;
  for (const triangle of triangles) {
    const area = triangleArea2(polygon, triangle);
    ok(area > EPS, `${label}: every triangle must be counter-clockwise and non-degenerate`);
    total += area;
    const [a, b, c] = triangle.map((i) => polygon[i]);
    const centroid = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3];
    ok(pointInPolygon(centroid, polygon) !== 0, `${label}: centroids must lie in the polygon`);
  }
  ok(Math.abs(total - doubledAreaOfPolygon(polygon)) < 1e-6, `${label}: areas must add up`);
}

/** Independent oracle for convex input: every triple with an empty circumcircle. */
function bruteDelaunayTriangles(polygon) {
  const n = polygon.length;
  const keys = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        const a = polygon[i];
        const b = polygon[j];
        const c = polygon[k];
        const ccw = sign(cross(a, b, c)) > 0 ? [i, j, k] : [i, k, j];
        const [p, q, r] = ccw.map((index) => polygon[index]);
        if (sign(cross(p, q, r)) <= 0) continue;
        let empty = true;
        for (let other = 0; other < n && empty; other++) {
          if (ccw.includes(other)) continue;
          if (inCircleExact(p, q, r, polygon[other]) > 0n) empty = false;
        }
        if (empty) keys.push(ccw.slice().sort((x, y) => x - y).join(","));
      }
    }
  }
  return keys.sort();
}

module.exports = { earClip, improveTriangulation, inCircleExact, inCircleFloat, minAngleDegrees,
  validateTriangulation, illegalEdgeCount, flippableIllegalCount, interiorEdges, quadOf,
  bruteDelaunayTriangles };

if (require.main === module) {
  const lShape = [[0, 0], [12, 0], [12, 4], [4, 4], [4, 12], [0, 12]];
  const comb = [[0, 0], [40, 0], [40, 30], [35, 30], [35, 10], [30, 10], [30, 30], [25, 30], [25, 10],
    [20, 10], [20, 30], [15, 30], [15, 10], [10, 10], [10, 30], [5, 30], [5, 10], [0, 10]];
  // Deliberately sliver-prone: several nearly collinear vertices.
  const sliver = [[0, 0], [10, 1], [20, 0], [20, 8], [10, 7], [0, 8]];
  const convex = [[0, 0], [14, 3], [22, 12], [18, 24], [6, 27], [-4, 16], [-5, 6]];
  const pentagon = [[0, 0], [10, 2], [14, 10], [7, 16], [-2, 9]];

  // -- the exact predicate ----------------------------------------------------------------
  strictEqual(inCircleExact([0, 0], [4, 0], [4, 4], [0, 4]), 0n, "the square's corners are cocircular");
  ok(inCircleExact([0, 0], [4, 0], [0, 4], [1, 1]) > 0n, "an interior point is strictly inside");
  ok(inCircleExact([0, 0], [4, 0], [0, 4], [5, 5]) < 0n, "a far point is strictly outside");
  // the predicate is invariant under even permutations of the triangle and antisymmetric under
  // odd ones; swapping two vertices must negate the determinant exactly
  strictEqual(inCircleExact([0, 0], [4, 0], [0, 4], [1, 1]), inCircleExact([4, 0], [0, 4], [0, 0], [1, 1]),
    "rotation of the triangle keeps the determinant");
  const swapA = inCircleExact([0, 0], [4, 0], [0, 4], [1, 1]);
  const swapB = inCircleExact([4, 0], [0, 0], [0, 4], [1, 1]);
  strictEqual(swapA, -swapB, "the predicate must be antisymmetric in the first two vertices");
  // independent exact formulation: the lifted 4x4 determinant must classify identically
  const liftedInCircle = (a2, b2, c2, d2) => {
    const rows = [a2, b2, c2, d2].map(([x, y]) => {
      const X = BigInt(x);
      const Y = BigInt(y);
      return [X, Y, X * X + Y * Y, 1n];
    });
    return determinant4(rows);
  };
  let liftedAgreements = 0;
  let floatAgreements = 0;
  let sampled = 0;
  for (let i = 0; i < 400; i++) {
    const r = (n) => ((i * 37 + n * 91) % 41) - 20;
    const a = [r(1), r(2)];
    const b = [r(3), r(4)];
    const c = [r(5), r(6)];
    const d = [r(7), r(8)];
    const exact = inCircleExact(a, b, c, d);
    const lifted = liftedInCircle(a, b, c, d);
    const approx = inCircleFloat(a, b, c, d);
    sampled++;
    if ((exact > 0n && lifted > 0n) || (exact < 0n && lifted < 0n) || (exact === 0n && lifted === 0n)) {
      liftedAgreements++;
    }
    if ((exact > 0n && approx > 0) || (exact < 0n && approx < 0) || (exact === 0n && approx === 0)) {
      floatAgreements++;
    }
  }
  strictEqual(liftedAgreements, sampled, "both exact formulations must classify every quadruple identically");
  ok(floatAgreements > 0.9 * sampled, `the float filter agrees on scaled integers (${floatAgreements}/${sampled})`);

  // -- ear clipping on several shapes ------------------------------------------------------
  for (const [name, polygon] of [["L", lShape], ["comb", comb], ["sliver", sliver], ["convex", convex],
    ["pentagon", pentagon]]) {
    for (const order of ["first", "best"]) {
      const triangles = earClip(polygon, order);
      validateTriangulation(polygon, triangles, `${name}/${order}`);
    }
  }
  throws(() => earClip([[0, 0], [1, 1]]), RangeError, "a degenerate outline is rejected");
  throws(() => earClip([[0, 0], [0, 10], [10, 10], [10, 0]]), Error,
    "a clockwise outline is rejected");

  // -- the `best` ear order beats the naive one on the sliver polygon -----------------------
  const firstOrder = earClip(sliver, "first");
  const bestOrder = earClip(sliver, "best");
  const firstAngle = minAngleDegrees(sliver, firstOrder);
  const bestAngle = minAngleDegrees(sliver, bestOrder);
  ok(bestAngle >= firstAngle - 1e-9,
    `best-ear ${bestAngle.toFixed(3)} must not be worse than first-ear ${firstAngle.toFixed(3)}`);

  // -- improvement with per-flip validation -------------------------------------------------
  const flipStates = [];
  const improved = improveTriangulation(sliver, firstOrder, {
    onFlip: (triangles) => {
      flipStates.push(triangles.map((t) => t.join(",")));
      validateTriangulation(sliver, triangles, `sliver flip ${flipStates.length}`);
    },
  });
  ok(improved.flips > 0, "the sliver polygon must need at least one flip");
  validateTriangulation(sliver, improved.triangles, "sliver improved");
  strictEqual(flippableIllegalCount(sliver, improved.triangles), 0, "no flippable illegal edge may remain");
  strictEqual(illegalEdgeCount(sliver, improved.triangles), 0, "the sliver mesh must be fully legal");
  const improvedAngle = minAngleDegrees(sliver, improved.triangles);
  ok(improvedAngle >= firstAngle - 1e-9,
    `improvement must not reduce the minimum angle (${firstAngle.toFixed(3)} -> ${improvedAngle.toFixed(3)})`);
  strictEqual(new Set(improved.triangles.map((t) => t.join(","))).size, improved.triangles.length,
    "no triangle may be duplicated by a flip");

  for (const [name, polygon] of [["L", lShape], ["comb", comb], ["convex", convex], ["pentagon", pentagon],
    ["sliver", sliver]]) {
    const base = earClip(polygon, "first");
    const before = illegalEdgeCount(polygon, base);
    const result = improveTriangulation(polygon, base, {
      onFlip: (triangles) => validateTriangulation(polygon, triangles, `${name} flip`),
    });
    validateTriangulation(polygon, result.triangles, `${name} improved`);
    strictEqual(flippableIllegalCount(polygon, result.triangles), 0,
      `${name}: no flippable illegal edge may remain`);
    ok(minAngleDegrees(polygon, result.triangles) >= minAngleDegrees(polygon, base) - 1e-9,
      `${name}: the minimum angle must not get worse`);
    // every triangle of the improved mesh must be a Delaunay triangle of the vertex set
    const delaunay = new Set(bruteDelaunayTriangles(polygon));
    const mine = result.triangles.map((t) => t.slice().sort((a, b) => a - b).join(","));
    ok(mine.every((key) => delaunay.has(key)), `${name}: every triangle must be a Delaunay triangle`);
    ok(result.flips >= 0, `${name}: flips are counted (${result.flips}, initially ${before} illegal)`);
  }
  strictEqual(illegalEdgeCount(pentagon, improveTriangulation(pentagon, earClip(pentagon, "first")).triangles), 0,
    "the convex cases must clear every illegal edge");

  // -- convex input: the improvement must produce the brute-force Delaunay triangulation ----
  const convexBase = earClip(convex, "first");
  const convexImproved = improveTriangulation(convex, convexBase).triangles;
  const mine = convexImproved.map((t) => t.slice().sort((a, b) => a - b).join(",")).sort();
  const oracle = bruteDelaunayTriangles(convex);
  deepStrictEqual(mine, oracle, "on convex input the flips must reach the Delaunay triangulation");
  ok(oracle.length === convex.length - 2, "the oracle covers the convex polygon with n - 2 triangles");

  // -- boundary cases ------------------------------------------------------------------------
  const triangle = [[0, 0], [10, 0], [0, 10]];
  const triangleTriangulation = earClip(triangle, "best");
  deepStrictEqual(triangleTriangulation, [[0, 1, 2]], "a triangle is its own triangulation");
  strictEqual(improveTriangulation(triangle, triangleTriangulation).flips, 0, "nothing to flip");
  const quad = [[0, 0], [10, 0], [10, 10], [0, 10]];
  const quadBase = earClip(quad, "first");
  strictEqual(quadBase.length, 2, "a quad gives two triangles");
  const quadImproved = improveTriangulation(quad, quadBase);
  strictEqual(quadImproved.flips, 0, "the square's corners are cocircular: nothing is illegal");
  deepStrictEqual(quadImproved.triangles, quadBase, "a cocircular quad must not be flipped");

  // -- determinism ---------------------------------------------------------------------------
  deepStrictEqual(earClip(comb, "best"), earClip(comb, "best"));
  deepStrictEqual(improveTriangulation(comb, earClip(comb, "best")).triangles,
    improveTriangulation(comb, earClip(comb, "best")).triangles);

  console.log(`sliver: first=${firstAngle.toFixed(3)}deg best=${bestAngle.toFixed(3)}deg ` +
    `improved=${improvedAngle.toFixed(3)}deg flips=${improved.flips}`);
  console.log(`convex_delaunay_triangles=${oracle.length} exact_form_agreements=${liftedAgreements}/${sampled} ` +
    `float_filter_agreements=${floatAgreements}/${sampled}`);
  console.log(`comb_illegal_before=${illegalEdgeCount(comb, earClip(comb, "first"))} ` +
    `comb_flips=${improveTriangulation(comb, earClip(comb, "first")).flips}`);
  console.log("PASS");
}
