/**
 * Task 372 - Polygon area via the shoelace formula (+ orientation). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The shoelace (Gauss) formula sums the cross products x_i*y_{i+1} - x_{i+1}*y_i over
 * the edges of the closed polygon.  Half of that sum is the SIGNED area: positive when
 * the vertices are listed counter-clockwise, negative when clockwise, and its absolute
 * value is the area for any *simple* polygon.  Because the vertices are integers the
 * doubled area is an exact integer and the area is an exact multiple of 0.5, so all
 * comparisons below are exact (no tolerance) except for the deliberately tolerant
 * rasterization cross-check of a non-axis-aligned triangle.
 *
 * Self-intersecting / self-touching (figure-8, "bowtie") polygons: the formula is not
 * an area at all.  It returns the sum of the lobe areas weighted by their traversal
 * orientation, so a bowtie whose two lobes cancel reports 0 (or the signed difference
 * for unequal lobes).  abs() does not repair that - only simple polygons are guaranteed
 * to yield the true area.  Two bowties below document this behaviour.
 *
 * Independent oracle: rasterizedArea counts unit cells (or finer) of the bounding box
 * whose centres lie inside the polygon via ray casting.  For polygons whose edges are
 * axis-parallel with integer coordinates every cell lies entirely inside or outside,
 * so the count is EXACT and asserted equal to |shoelace area|.  For a general triangle
 * the comparison uses a documented tolerance (cells crossed by the boundary contribute
 * at most ~perimeter/samplesPerUnit).
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

/** Twice the signed area (an exact integer for integer vertices). */
function signedDoubleArea(points) {
  let total = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    total += x1 * y2 - x2 * y1;
  }
  return total === 0 ? 0 : total;             // normalises any -0 the sum might produce
}

/** Signed area; exact (a multiple of 0.5) because vertices are integers. */
function polygonArea(points) {
  const area = signedDoubleArea(points) / 2.0;
  return area === 0 ? 0 : area;               // deepStrictEqual distinguishes -0 from 0
}

/** +1 counter-clockwise, -1 clockwise, 0 degenerate (all collinear). */
function orientation(points) {
  const doubleArea = signedDoubleArea(points);
  return doubleArea > 0 ? 1 : (doubleArea < 0 ? -1 : 0);
}

/** Ray casting; sample points used below never lie on an edge. */
function pointInPolygon(x, y, points) {
  let inside = false;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    if ((y1 > y) !== (y2 > y)) {
      const xCross = x1 + ((y - y1) * (x2 - x1)) / (y2 - y1);
      if (xCross > x) inside = !inside;
    }
  }
  return inside;
}

/** Count cell centres inside the polygon; exact for integer axis-aligned edges. */
function rasterizedArea(points, samplesPerUnit) {
  if (points.length < 3) return 0;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const epsilon = 1e-6;                       // nudges samples off integer/diagonal lines
  let inside = 0;
  for (let i = minX * samplesPerUnit; i < maxX * samplesPerUnit; i++) {
    const x = (i + 0.5) / samplesPerUnit + epsilon;
    for (let j = minY * samplesPerUnit; j < maxY * samplesPerUnit; j++) {
      const y = (j + 0.5) / samplesPerUnit + epsilon;
      if (pointInPolygon(x, y, points)) inside++;
    }
  }
  return inside / (samplesPerUnit * samplesPerUnit);
}

function lcgRectangle(lcg) {
  const width = 1 + lcg.next(6);
  const height = 1 + lcg.next(6);
  return [[0, 0], [width, 0], [width, height], [0, height]];
}

function lcgLShape(lcg) {
  const a = 2 + lcg.next(7);
  const b = 1 + lcg.next(4);
  const c = 1 + lcg.next(a - 1);
  const d = b + 1 + lcg.next(5);
  return { shape: [[0, 0], [a, 0], [a, b], [c, b], [c, d], [0, d]], expected: a * b + c * (d - b) };
}

module.exports = { signedDoubleArea, polygonArea, orientation, pointInPolygon, rasterizedArea };

if (require.main === module) {
  const assert = require("node:assert");

  // --- hand-checked polygons -------------------------------------------
  const square = [[0, 0], [2, 0], [2, 2], [0, 2]];
  assert.strictEqual(signedDoubleArea(square), 8, "square doubled area");
  assert.strictEqual(polygonArea(square), 4, "square area 4");
  assert.strictEqual(orientation(square), 1, "square is CCW");
  assert.strictEqual(rasterizedArea(square, 1), 4, "square rasterized exactly");

  const smallTriangle = [[0, 0], [1, 0], [0, 1]];
  assert.strictEqual(polygonArea(smallTriangle), 0.5, "unit triangle 0.5");
  assert.strictEqual(orientation(smallTriangle), 1, "unit triangle CCW");

  const bigTriangle = [[0, 0], [2, 0], [0, 2]];
  assert.strictEqual(polygonArea(bigTriangle), 2, "leg-2 triangle 2");

  const clockwise = [[0, 0], [0, 2], [2, 2], [2, 0]];
  assert.strictEqual(signedDoubleArea(clockwise), -8, "clockwise doubled area");
  assert.strictEqual(polygonArea(clockwise), -4, "clockwise negative area");
  assert.strictEqual(Math.abs(polygonArea(clockwise)), 4, "clockwise absolute area 4");
  assert.strictEqual(orientation(clockwise), -1, "clockwise orientation");

  const lShape = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]];
  assert.strictEqual(polygonArea(lShape), 5, "L-shape area 5");
  assert.strictEqual(rasterizedArea(lShape, 1), 5, "L-shape rasterized exactly");

  // general triangle with diagonal edges: rasterized within the documented tolerance
  const general = [[0, 0], [5, 1], [2, 4]];
  assert.strictEqual(polygonArea(general), 9, "general triangle 9");
  const estimate = rasterizedArea(general, 32);
  assert.ok(Math.abs(estimate - 9) <= 0.5, `general triangle rasterized within 0.5, got ${estimate}`);

  // collinear vertices do not change the area
  const collinearExtra = [[0, 0], [1, 0], [2, 0], [2, 2], [0, 2]];
  assert.strictEqual(polygonArea(collinearExtra), 4, "collinear extra vertex");
  assert.strictEqual(orientation([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]), 1, "repeated vertex");

  // --- figure-8 (bowtie) documented behaviour --------------------------
  const bowtie = [[0, 0], [2, 2], [2, 0], [0, 2]];              // lobes +1 and -1
  assert.strictEqual(signedDoubleArea(bowtie), 0, "bowtie lobes cancel");
  assert.strictEqual(polygonArea(bowtie), 0, "bowtie signed area 0");
  const skewedBowtie = [[0, 0], [2, 4], [2, 0], [0, 2]];        // 8/3 - 2/3 = 2
  assert.strictEqual(signedDoubleArea(skewedBowtie), -4, "skewed bowtie doubled area");
  assert.strictEqual(polygonArea(skewedBowtie), -2, "skewed bowtie -2");

  // --- LCG cross-checks against the exact rasterization oracle ---------
  const lcg = new Lcg(37211);
  for (let trial = 0; trial < 30; trial++) {
    const rectangle = lcgRectangle(lcg);
    const width = Math.max(...rectangle.map((p) => p[0]));
    const height = Math.max(...rectangle.map((p) => p[1]));
    assert.strictEqual(polygonArea(rectangle), width * height, `rect area ${trial}`);
    assert.strictEqual(rasterizedArea(rectangle, 1), width * height, `rect raster ${trial}`);
    const reversedRectangle = rectangle.slice().reverse();
    assert.strictEqual(polygonArea(reversedRectangle), -polygonArea(rectangle), `rect reversed ${trial}`);
    assert.strictEqual(orientation(reversedRectangle), -orientation(rectangle), `rect reversed orientation ${trial}`);
  }
  for (let trial = 0; trial < 30; trial++) {
    const { shape, expected } = lcgLShape(lcg);
    assert.strictEqual(signedDoubleArea(shape), 2 * expected, `L doubled area ${trial}`);
    assert.strictEqual(polygonArea(shape), expected, `L area ${trial}`);
    assert.strictEqual(rasterizedArea(shape, 1), expected, `L raster ${trial}`);
    assert.strictEqual(orientation(shape), 1, `L orientation ${trial}`);
  }
  console.log("PASS");
}
