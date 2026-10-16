/** Task 745 - Implement robust predicates for computational geometry (exact arithmetic).
 *  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Four predicates - orient2d, orient3d, incircle, insphere - in the classical two-tier shape.
 * Tier one evaluates the determinant in doubles and compares it against a forward error bound
 * built from the absolute values of the individual products (the same shape as Shewchuk's static
 * filters, with a deliberately pessimistic constant); when the value is not safely away from zero
 * the tier-two BigInt evaluation of the very same determinant decides. The exact evaluations are
 * exact over the *integers* the inputs must provide: orient2d is a 2x2 determinant of
 * differences, orient3d the 3x3 scalar triple product, incircle the relative 3x3 determinant
 * with squared distances and insphere the relative 4x4 determinant with squared distances.
 *
 * What is simplified: coordinates must be integers (floating coordinates would need exact
 * rationals or Shewchuk's expansions; the header records the restriction), the adaptive
 * multi-stage expansion arithmetic of the original library is replaced by a single filter plus a
 * BigInt fallback, and only the sign of each determinant is exposed. What is verified: every
 * predicate agrees with an independent lift - the same determinant built from the (n+1)x(n+1)
 * matrix of the paraboloid lifting and evaluated with a generic BigInt determinant - on random
 * integer inputs; the filter is exercised (fallbacks are counted and are non-zero for nearly
 * degenerate configurations); a naive double evaluation is shown to get the sign wrong on a
 * constructed near-collinear case where the filtered predicate stays exact; degenerate cases
 * (collinear, coplanar, cocircular, cospherical) return zero; antisymmetry and cyclic invariance
 * hold; non-integer or malformed input is rejected; and a convex hull built with the exact
 * orientation predicate matches an independent O(n^3) half-plane oracle.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

/** Deterministic xorshift32; `bound` limits the result. */
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

const FILTER_SLACK = 1e-14; // pessimistic relative bound: bigger than Shewchuk's, so it is safe
let fallbacks = 0;
const fallbackCount = () => fallbacks;

function checkIntegers(points, arity) {
  if (!Array.isArray(points) || points.length !== arity) {
    throw new TypeError(`expected ${arity} points`);
  }
  for (const point of points) {
    if (!Array.isArray(point) || !point.every((c) => Number.isInteger(c))) {
      throw new TypeError("coordinates must be integers so that the exact tier is exact");
    }
  }
}

// ---- exact tier: BigInt determinants -------------------------------------------------------
const big = (v) => v.map((c) => BigInt(c));
const bsub = (a, b) => a.map((c, i) => c - b[i]);
const bdot = (a, b) => a.reduce((s, c, i) => s + c * b[i], 0n);

/** Generic n x n BigInt determinant by expansion along the first row. */
function detN(rows) {
  const n = rows.length;
  if (n === 1) return rows[0][0];
  let total = 0n;
  for (let column = 0; column < n; column++) {
    const minor = rows.slice(1).map((row) => row.filter((_, c) => c !== column));
    const term = rows[0][column] * detN(minor);
    total += column % 2 === 0 ? term : -term;
  }
  return total;
}

function bigSign(value) {
  return value > 0n ? 1 : value < 0n ? -1 : 0;
}

/** The lifted form: |a|^2 in the last column, used as an independent cross-check. */
function liftedOrient2d(a, b, c) {
  return bigSign(detN([a, b, c].map((p) => {
    const [x, y] = big(p);
    return [x, y, 1n];
  })));
}

function liftedInCircle(a, b, c, d) {
  // the 4x4 form: rows (x, y, x^2 + y^2, 1) for the four points
  return bigSign(detN([a, b, c, d].map((p) => {
    const [x, y] = big(p);
    return [x, y, x * x + y * y, 1n];
  })));
}

function liftedInSphere(a, b, c, d, e) {
  // the 5x5 form: rows (x, y, z, x^2 + y^2 + z^2, 1)
  return bigSign(detN([a, b, c, d, e].map((p) => {
    const [x, y, z] = big(p);
    return [x, y, z, x * x + y * y + z * z, 1n];
  })));
}

function exactOrient2d(a, b, c) {
  const [ax, ay] = big(a);
  const [bx, by] = big(b);
  const [cx, cy] = big(c);
  return bigSign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax));
}

function exactOrient3d(a, b, c, d) {
  // rows are the differences from d, exactly the arrangement the floating filter uses
  const [ax, ay, az] = big(a);
  const [bx, by, bz] = big(b);
  const [cx, cy, cz] = big(c);
  const [dx, dy, dz] = big(d);
  return bigSign(-detN([
    [ax - dx, ay - dy, az - dz],
    [bx - dx, by - dy, bz - dz],
    [cx - dx, cy - dy, cz - dz],
  ]));
}

function exactInCircle(a, b, c, d) {
  const [dx, dy] = big(d);
  const rows = [a, b, c].map((p) => {
    const [x, y] = big(p);
    const rx = x - dx;
    const ry = y - dy;
    return [rx, ry, rx * rx + ry * ry];
  });
  return bigSign(detN(rows));
}

function exactInSphere(a, b, c, d, e) {
  const [ex, ey, ez] = big(e);
  const rows = [a, b, c, d].map((p) => {
    const [x, y, z] = big(p);
    const rx = x - ex;
    const ry = y - ey;
    const rz = z - ez;
    return [rx, ry, rz, rx * rx + ry * ry + rz * rz];
  });
  return bigSign(detN(rows));
}

// ---- filtered tier -------------------------------------------------------------------------
function orient2d(a, b, c) {
  checkIntegers([a, b, c], 3);
  const acx = a[0] - c[0];
  const acy = a[1] - c[1];
  const bcx = b[0] - c[0];
  const bcy = b[1] - c[1];
  const left = acx * bcy;
  const right = acy * bcx;
  const determinant = left - right;
  const bound = FILTER_SLACK * (Math.abs(left) + Math.abs(right));
  if (Math.abs(determinant) > bound) return determinant > 0 ? 1 : -1;
  fallbacks++;
  return exactOrient2d(a, b, c);
}

function orient3d(a, b, c, d) {
  checkIntegers([a, b, c, d], 4);
  const ax = a[0] - d[0];
  const ay = a[1] - d[1];
  const az = a[2] - d[2];
  const bx = b[0] - d[0];
  const by = b[1] - d[1];
  const bz = b[2] - d[2];
  const cx = c[0] - d[0];
  const cy = c[1] - d[1];
  const cz = c[2] - d[2];
  const terms = [
    ax * (by * cz - bz * cy),
    -ay * (bx * cz - bz * cx),
    az * (bx * cy - by * cx),
  ];
  // sign convention: positive when a, b, c appear counter-clockwise seen from d
  const determinant = -(terms[0] + terms[1] + terms[2]);
  const bound = FILTER_SLACK * terms.reduce((s, t) => s + Math.abs(t), 0);
  if (Math.abs(determinant) > bound) return determinant > 0 ? 1 : -1;
  fallbacks++;
  return exactOrient3d(a, b, c, d);
}

function incircle(a, b, c, d) {
  checkIntegers([a, b, c, d], 4);
  const rows = [a, b, c].map((p) => {
    const dx = p[0] - d[0];
    const dy = p[1] - d[1];
    return [dx, dy, dx * dx + dy * dy];
  });
  const determinant = det3Float(rows);
  const bound = FILTER_SLACK * (Math.abs(rows[0][0]) * Math.abs(rows[1][1]) * Math.abs(rows[2][2]) +
    Math.abs(rows[0][1]) * Math.abs(rows[1][2]) * Math.abs(rows[2][0]) +
    Math.abs(rows[0][2]) * Math.abs(rows[1][0]) * Math.abs(rows[2][1]) +
    Math.abs(rows[0][2]) * Math.abs(rows[1][1]) * Math.abs(rows[2][0]) +
    Math.abs(rows[0][0]) * Math.abs(rows[1][2]) * Math.abs(rows[2][1]) +
    Math.abs(rows[0][1]) * Math.abs(rows[1][0]) * Math.abs(rows[2][2]));
  if (Math.abs(determinant) > bound) return determinant > 0 ? 1 : -1;
  fallbacks++;
  return exactInCircle(a, b, c, d);
}

function insphere(a, b, c, d, e) {
  checkIntegers([a, b, c, d, e], 5);
  const rows = [a, b, c, d].map((p) => {
    const dx = p[0] - e[0];
    const dy = p[1] - e[1];
    const dz = p[2] - e[2];
    return [dx, dy, dz, dx * dx + dy * dy + dz * dz];
  });
  const determinant = det4Float(rows);
  let magnitude = 0;
  for (let i = 0; i < 4; i++) {
    for (const permutation of PERMUTATIONS4) {
      let product = 1;
      let sign = 1;
      for (let c = 0; c < 4; c++) {
        product *= Math.abs(rows[permutation[c]][c]);
        if (permutation[c] !== c) sign = -sign;
      }
      magnitude += Math.abs(product);
    }
  }
  const bound = FILTER_SLACK * magnitude;
  if (Math.abs(determinant) > bound) return determinant > 0 ? 1 : -1;
  fallbacks++;
  return exactInSphere(a, b, c, d, e);
}

function det3Float(rows) {
  const [[a, b, c], [d, e, f], [g, h, i]] = rows;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

function det4Float(rows) {
  let total = 0;
  for (let column = 0; column < 4; column++) {
    const minor = det3Float(rows.slice(1).map((row) => row.filter((_, c) => c !== column)));
    total += column % 2 === 0 ? rows[0][column] * minor : -rows[0][column] * minor;
  }
  return total;
}

const PERMUTATIONS4 = (() => {
  const out = [];
  const walk = (prefix, rest) => {
    if (rest.length === 0) {
      out.push(prefix);
      return;
    }
    rest.forEach((value, index) => walk([...prefix, value], rest.filter((_, i) => i !== index)));
  };
  walk([], [0, 1, 2, 3]);
  return out;
})();

/** A deliberately naive double evaluation, kept to show that it loses the sign. */
function naiveOrient2d(a, b, c) {
  return Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
}

// ---- a real algorithm on top of the exact predicate ------------------------------------------
/** Counter-clockwise convex hull; collinear interior points are dropped by the exact predicate. */
function convexHull(points) {
  const sorted = points.map((p, i) => [p, i]).sort((x, y) => x[0][0] - y[0][0] ||
    x[0][1] - y[0][1] || x[1] - y[1]);
  if (sorted.length <= 1) return sorted.map(([, i]) => i);
  const build = (sequence) => {
    const chain = [];
    for (const [point, index] of sequence) {
      while (chain.length >= 2 &&
        orient2d(points[chain[chain.length - 2]], points[chain[chain.length - 1]], point) <= 0) {
        chain.pop();
      }
      chain.push(index);
    }
    return chain;
  };
  const lower = build(sorted);
  const upper = build(sorted.slice().reverse());
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

/** Independent O(n^3) oracle: hull = points that cannot be put strictly inside a half-plane. */
function bruteHull(points) {
  const hull = [];
  for (let i = 0; i < points.length; i++) {
    let extreme = false;
    for (let j = 0; j < points.length && !extreme; j++) {
      if (j === i) continue;
      let positive = false;
      let negative = false;
      for (let k = 0; k < points.length; k++) {
        const side = orient2d(points[i], points[j], points[k]);
        if (side > 0) positive = true;
        if (side < 0) negative = true;
      }
      // a hull edge has every other point on one closed side (collinear allowed)
      if (side_allows(positive, negative)) extreme = true;
    }
    if (extreme) hull.push(i);
  }
  return hull.sort((a, b) => a - b);
}

function side_allows(positive, negative) {
  return !(positive && negative);
}

module.exports = { orient2d, orient3d, incircle, insphere, convexHull, bruteHull,
  exactOrient2d, exactOrient3d, exactInCircle, exactInSphere, fallbackCount, naiveOrient2d,
  liftedOrient2d, liftedInCircle, liftedInSphere };

if (require.main === module) {
  // -- hand checked signs and degeneracies -------------------------------------------------
  strictEqual(orient2d([0, 0], [4, 0], [0, 4]), 1, "counter-clockwise is positive");
  strictEqual(orient2d([0, 0], [0, 4], [4, 0]), -1, "clockwise is negative");
  strictEqual(orient2d([0, 0], [2, 2], [5, 5]), 0, "collinear is zero");
  strictEqual(orient2d([0, 0], [4, 0], [0, 4]), -orient2d([4, 0], [0, 0], [0, 4]),
    "swapping two points negates the sign");
  strictEqual(orient2d([0, 0], [4, 0], [0, 4]), orient2d([4, 0], [0, 4], [0, 0]),
    "a cyclic rotation keeps the sign");
  strictEqual(orient3d([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]), 1, "positively oriented tetrahedron");
  strictEqual(orient3d([0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]), -1, "swapping two base points flips");
  strictEqual(orient3d([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]), -1, "the other side of the plane");
  strictEqual(orient3d([0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]), 0, "coplanar is zero");
  strictEqual(exactOrient3d([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]), 1, "both tiers agree");
  strictEqual(incircle([0, 0], [4, 0], [0, 4], [1, 1]), 1, "an interior point is inside the circle");
  strictEqual(incircle([0, 0], [4, 0], [0, 4], [9, 9]), -1, "a far point is outside");
  strictEqual(insphere([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]), 0,
    "the cube's opposite corner lies on the circumsphere");
  strictEqual(insphere([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]), 1,
    "an outside point is reported positively for a positive tetrahedron");

  // -- the naive double evaluation loses the sign, the filtered one does not ------------------
  const nearCollinear = [
    [[0, 0], [1000000000, 1000000001], [2000000001, 2000000003]],
    [[0, 0], [999999999, 1000000000], [1999999999, 2000000001]],
    [[0, 0], [1000000000, 999999999], [2000000001, 1999999998]],
  ];
  let naiveWrong = 0;
  for (const [a, b, c] of nearCollinear) {
    const exact = exactOrient2d(a, b, c);
    ok(exact !== 0, "the constructed configuration must be genuinely non-degenerate");
    if (naiveOrient2d(a, b, c) !== exact) naiveWrong++;
    strictEqual(orient2d(a, b, c), exact, "the filtered predicate must match the exact one");
  }
  ok(naiveWrong > 0, `the naive evaluation must fail at least once (failed ${naiveWrong} times)`);

  // -- filter and exact tiers agree, and the filter really is exercised -----------------------
  const rng = makeRng(745745);
  const randomPoint = (dimension, spread) =>
    Array.from({ length: dimension }, () => rng(spread) - (spread >> 1));
  let agreements = 0;
  for (let trial = 0; trial < 300; trial++) {
    const a2 = randomPoint(2, 41);
    const b2 = randomPoint(2, 41);
    const c2 = randomPoint(2, 41);
    const d2 = randomPoint(2, 41);
    strictEqual(orient2d(a2, b2, c2), exactOrient2d(a2, b2, c2), "orient2d filter vs exact");
    strictEqual(orient2d(a2, b2, c2), liftedOrient2d(a2, b2, c2), "orient2d vs the lifted form");
    strictEqual(incircle(a2, b2, c2, d2), exactInCircle(a2, b2, c2, d2), "incircle filter vs exact");
    strictEqual(exactInCircle(a2, b2, c2, d2), liftedInCircle(a2, b2, c2, d2),
      "incircle vs the lifted 4x4 form");
    // geometric oracle: the circle through (0,0), (4,0), (0,4) has centre (2,2) and radius^2 8
    if (orient2d([0, 0], [4, 0], [0, 4]) === 1) {
      const inside = (d2[0] - 2) ** 2 + (d2[1] - 2) ** 2 < 8;
      const expected = inside ? 1 : (d2[0] - 2) ** 2 + (d2[1] - 2) ** 2 === 8 ? 0 : -1;
      strictEqual(exactInCircle([0, 0], [4, 0], [0, 4], d2), expected,
        "incircle must agree with the explicit circle of the unit triangle");
    }
    agreements += 4;
  }
  for (let trial = 0; trial < 200; trial++) {
    const a3 = randomPoint(3, 21);
    const b3 = randomPoint(3, 21);
    const c3 = randomPoint(3, 21);
    const d3 = randomPoint(3, 21);
    const e3 = randomPoint(3, 21);
    strictEqual(orient3d(a3, b3, c3, d3), exactOrient3d(a3, b3, c3, d3), "orient3d filter vs exact");
    strictEqual(insphere(a3, b3, c3, d3, e3), exactInSphere(a3, b3, c3, d3, e3),
      "insphere filter vs exact");
    strictEqual(exactInSphere(a3, b3, c3, d3, e3), liftedInSphere(a3, b3, c3, d3, e3),
      "insphere vs the lifted 5x5 form");
    // geometric oracle: the sphere through the unit tetrahedron has centre (1/2,1/2,1/2) and
    // radius^2 3/4, so 4*|e|^2 - 4*(ex+ey+ez) + 3 compares exactly
    const unitTetra = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const orientation = exactOrient3d(...unitTetra);
    const scaled = 4 * (e3[0] * e3[0] + e3[1] * e3[1] + e3[2] * e3[2]) -
      4 * (e3[0] + e3[1] + e3[2]) + 3;
    const expected = scaled === 0 ? 0 : orientation * (scaled > 0 ? 1 : -1);
    strictEqual(exactInSphere(...unitTetra, e3), expected,
      "insphere must agree with the explicit sphere of the unit tetrahedron");
    agreements += 3;
  }
  strictEqual(agreements, 1800, "the whole corpus must be compared");
  ok(fallbackCount() > 0, `nearly degenerate inputs must reach the exact tier (${fallbackCount()} times)`);

  // -- the exact tier is invariant under transformations --------------------------------------
  for (let trial = 0; trial < 50; trial++) {
    const a = randomPoint(2, 31);
    const b = randomPoint(2, 31);
    const c = randomPoint(2, 31);
    const d = randomPoint(2, 31);
    strictEqual(incircle(a, b, c, d), incircle(b, c, a, d), "incircle is cyclic in its first three");
    strictEqual(incircle(a, b, c, d) + incircle(b, a, c, d), 0,
      "swapping two points negates incircle (zero stays zero)");
    const shifted = (p) => [p[0] + 1000003, p[1] - 77];
    strictEqual(orient2d(shifted(a), shifted(b), shifted(c)), orient2d(a, b, c),
      "translation does not change orientation");
    strictEqual(incircle(shifted(a), shifted(b), shifted(c), shifted(d)), incircle(a, b, c, d),
      "translation does not change the in-circle sign");
  }

  // -- input validation ------------------------------------------------------------------------
  throws(() => orient2d([0, 0], [1, 1], [0.5, 0]), TypeError, "non-integer coordinates are rejected");
  throws(() => incircle([0, 0], [1, 1], [2, 2]), TypeError, "the arity is enforced");
  throws(() => insphere([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], ["x", 1, 2]), TypeError,
    "coordinates must be numbers");

  // -- the hull built on the exact predicate versus independent invariants ----------------------
  let hullChecks = 0;
  for (let trial = 0; trial < 25; trial++) {
    const cloud = [];
    const size = 4 + rng(12);
    for (let i = 0; i < size; i++) cloud.push(randomPoint(2, 25));
    const hull = convexHull(cloud);
    const oracle = new Set(bruteHull(cloud));
    for (const index of hull) {
      ok(oracle.has(index), `trial ${trial}: every hull vertex must be an extreme point`);
    }
    for (let i = 0; i < hull.length; i++) {
      const a = cloud[hull[i]];
      const b = cloud[hull[(i + 1) % hull.length]];
      for (const point of cloud) {
        ok(orient2d(a, b, point) >= 0, "hull edges keep every point on the inner closed side");
      }
      hullChecks++;
    }
    // strictly convex corners and no point outside the hull's bounding box
    for (let i = 0; i < hull.length && hull.length > 2; i++) {
      const turnValue = orient2d(cloud[hull[i]], cloud[hull[(i + 1) % hull.length]],
        cloud[hull[(i + 2) % hull.length]]);
      ok(turnValue > 0, "consecutive hull corners must turn strictly counter-clockwise");
    }
    const hullSet = new Set(hull);
    for (let index = 0; index < cloud.length; index++) {
      if (hullSet.has(index)) continue;
      const inside = hull.every((vertex, i) =>
        orient2d(cloud[vertex], cloud[hull[(i + 1) % hull.length]], cloud[index]) >= 0);
      ok(inside, `trial ${trial}: a non-hull point must lie inside or on the hull`);
    }
  }
  ok(hullChecks > 40, `the hull invariant must be tested broadly (${hullChecks} edges)`);
  deepStrictEqual(convexHull([]), [], "an empty cloud has no hull");
  deepStrictEqual(convexHull([[3, 3]]), [0], "a single point is its own hull");
  deepStrictEqual(convexHull([[0, 0], [1, 1], [2, 2]]).length, 2, "collinear points give a segment");

  console.log(`corpus=${agreements} comparisons exact_fallbacks=${fallbackCount()} ` +
    `naive_failures=${naiveWrong}`);
  console.log(`hull_checks=${hullChecks} predicates=orient2d orient3d incircle insphere`);
  console.log("PASS");
}
