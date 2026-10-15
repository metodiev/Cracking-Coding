/**
 * Task 371 - Line-segment intersection (Bentley-Ottmann sweep outline). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Detect all intersecting pairs among n segments.  The full Bentley-Ottmann algorithm
 * runs in O((n + k) log n) for k intersection points; this task implements the sweep
 * *skeleton* (event queue + status structure) and handles the small cases this repo
 * tests (n <= ~12) exactly, plus a plain O(n^2) fallback used as the oracle.
 *
 * Bentley-Ottmann invariant (the full algorithm this skeleton approximates):
 *   1. Events: every segment endpoint, ordered lexicographically by (x, y).  A left
 *      endpoint inserts the segment into the status structure; a right endpoint
 *      removes it.  Intersection points found during the sweep are also pushed as
 *      events so that the order of the status structure is restored after a crossing.
 *   2. Status structure: the segments currently crossing the vertical sweep line,
 *      kept ordered by their y coordinate at the current sweep x.  In general
 *      position no two of them cross between events, so the order is well defined
 *      and stable there.
 *   3. Crossing detection: only neighbours in the status can become adjacent-and-
 *      swapped between events, so a crossing is reported when two segments are
 *      adjacent in the status (on insertion/removal, and after processing a found
 *      intersection event) and their exact intersection test succeeds.  Each found
 *      intersection is reported once, at the first event x where the two segments
 *      become adjacent.
 *   4. Degeneracies (documented but only partially handled here): vertical segments
 *      (processed entirely at one event x), several events with equal x, endpoint
 *      touches, collinear overlaps and overlapping collinear bundles.
 *
 * Small-case exact handling used here: at every event x, after all left endpoints of
 * that x have been inserted and finished segments dropped, every *pair* of segments in
 * the status structure is tested with the exact integer predicate.  An intersecting
 * pair always spans the event x = max(left_x1, left_x2) (the later segment's left
 * endpoint), so every intersecting pair is seen at least once, duplicate reports
 * collapse into a set, and the count matches the O(n^2) oracle.
 *
 * All predicates use exact integer arithmetic (cross products); no floating point.
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

/** Twice the signed area of (p, q, r): >0 CCW, <0 CW, 0 collinear. */
function orient(px, py, qx, qy, rx, ry) {
  return (qx - px) * (ry - py) - (qy - py) * (rx - px);
}

/** True when collinear q lies inside segment pr (endpoints included). */
function onSegment(px, py, qx, qy, rx, ry) {
  return Math.min(px, rx) <= qx && qx <= Math.max(px, rx)
    && Math.min(py, ry) <= qy && qy <= Math.max(py, ry);
}

/** Exact closed-segment intersection test (touches and overlaps count). */
function segmentsIntersect(s, t) {
  const [p1x, p1y, p2x, p2y] = s;
  const [p3x, p3y, p4x, p4y] = t;
  const o1 = orient(p1x, p1y, p2x, p2y, p3x, p3y);
  const o2 = orient(p1x, p1y, p2x, p2y, p4x, p4y);
  const o3 = orient(p3x, p3y, p4x, p4y, p1x, p1y);
  const o4 = orient(p3x, p3y, p4x, p4y, p2x, p2y);
  if (o1 * o2 < 0 && o3 * o4 < 0) return true;                  // proper crossing
  if (o1 === 0 && onSegment(p1x, p1y, p3x, p3y, p2x, p2y)) return true;
  if (o2 === 0 && onSegment(p1x, p1y, p4x, p4y, p2x, p2y)) return true;
  if (o3 === 0 && onSegment(p3x, p3y, p1x, p1y, p4x, p4y)) return true;
  if (o4 === 0 && onSegment(p3x, p3y, p2x, p2y, p4x, p4y)) return true;
  return false;
}

/** Normalise [x1, y1, x2, y2] so that (x1, y1) <= (x2, y2) lexicographically. */
function normalize(segments) {
  return segments.map((s) => {
    const copy = s.slice();
    if (copy[0] > copy[2] || (copy[0] === copy[2] && copy[1] > copy[3])) {
      return [copy[2], copy[3], copy[0], copy[1]];
    }
    return copy;
  });
}

/** -1/0/1 comparing the segments' exact y values at the sweep line x. */
function compareY(a, b, x) {
  let na, da, nb, db;
  if (a[2] === a[0]) {
    na = a[1];
    da = 1;
  } else {
    na = a[1] * (a[2] - a[0]) + (a[3] - a[1]) * (x - a[0]);
    da = a[2] - a[0];
  }
  if (b[2] === b[0]) {
    nb = b[1];
    db = 1;
  } else {
    nb = b[1] * (b[2] - b[0]) + (b[3] - b[1]) * (x - b[0]);
    db = b[2] - b[0];
  }
  const left = na * db;
  const right = nb * da;
  return left < right ? -1 : (left > right ? 1 : 0);
}

/** Sweep skeleton: event queue + status structure; returns the intersecting pairs. */
function sweepIntersectionPairs(rawSegments) {
  const segments = normalize(rawSegments);

  const events = [];
  segments.forEach((s, idx) => {
    events.push([s[0], s[1], idx, 0]);              // 0 = left endpoint
    events.push([s[2], s[3], idx, 1]);              // 1 = right endpoint
  });
  events.sort((u, v) => u[0] - v[0] || u[1] - v[1] || u[2] - v[2] || u[3] - v[3]);

  let status = [];                                   // active ids, y-ordered
  const found = new Set();
  let i = 0;
  while (i < events.length) {
    const x = events[i][0];
    let j = i;
    while (j < events.length && events[j][0] === x) {
      const event = events[j];
      if (event[3] === 0) {                          // insert, keeping y-order
        let pos = 0;
        while (pos < status.length && compareY(segments[status[pos]], segments[event[2]], x) <= 0) pos++;
        status.splice(pos, 0, event[2]);
      }
      j++;
    }
    status = status.filter((k) => segments[k][2] >= x);   // finish finished segments
    for (let a = 0; a < status.length; a++) {
      for (let b = a + 1; b < status.length; b++) {
        if (segmentsIntersect(segments[status[a]], segments[status[b]])) {
          const low = Math.min(status[a], status[b]);
          const high = Math.max(status[a], status[b]);
          found.add(`${low}:${high}`);
        }
      }
    }
    i = j;
  }
  return [...found].sort();
}

/** O(n^2) oracle: test every pair with the same exact predicate. */
function bruteForceIntersectionPairs(segments) {
  const found = [];
  for (let i = 0; i < segments.length; i++) {
    for (let j = i + 1; j < segments.length; j++) {
      if (segmentsIntersect(segments[i], segments[j])) found.push(`${i}:${j}`);
    }
  }
  return found.sort();
}

function lcgSegments(count, lcg) {
  const segments = [];
  for (let i = 0; i < count; i++) {
    let x1, y1, x2, y2;
    do {
      x1 = lcg.next(41); y1 = lcg.next(41);
      x2 = lcg.next(41); y2 = lcg.next(41);
    } while (x1 === x2 && y1 === y2);
    segments.push([x1, y1, x2, y2]);
  }
  return segments;
}

module.exports = {
  orient, onSegment, segmentsIntersect, sweepIntersectionPairs, bruteForceIntersectionPairs,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- hand predicates ------------------------------------------------
  assert.strictEqual(segmentsIntersect([0, 0, 4, 4], [0, 4, 4, 0]), true, "proper crossing");
  assert.strictEqual(segmentsIntersect([0, 0, 2, 2], [2, 2, 4, 0]), true, "shared endpoint");
  assert.strictEqual(segmentsIntersect([0, 0, 4, 0], [2, 0, 6, 0]), true, "collinear overlap");
  assert.strictEqual(segmentsIntersect([0, 0, 2, 0], [0, 1, 2, 1]), false, "parallel disjoint");
  assert.strictEqual(segmentsIntersect([0, 0, 4, 0], [2, 0, 2, 3]), true, "T-touch");
  assert.strictEqual(segmentsIntersect([0, 0, 1, 1], [3, 0, 4, 1]), false, "separated");
  assert.strictEqual(segmentsIntersect([0, 0, 1, 1], [1, 1, 2, 0]), true, "endpoint meet");

  // --- hand sweep set: 4 intersecting pairs ---------------------------
  const hand = [[0, 0, 4, 4], [0, 4, 4, 0], [2, 2, 6, 2], [6, 0, 6, 6], [8, 8, 9, 9]];
  const expected = ["0:1", "0:2", "1:2", "2:3"];
  assert.deepStrictEqual(sweepIntersectionPairs(hand), expected, "hand sweep set");
  assert.deepStrictEqual(bruteForceIntersectionPairs(hand), expected, "hand brute force");
  assert.strictEqual(expected.length, 4, "hand count is 4");

  // --- collinear / touch family ----------------------------------------
  const collinear = [[0, 0, 4, 0], [2, 0, 6, 0], [0, 1, 6, 1], [0, 0, 0, 1]];
  assert.deepStrictEqual(sweepIntersectionPairs(collinear), ["0:1", "0:3", "2:3"], "collinear family");
  assert.deepStrictEqual(sweepIntersectionPairs(collinear), bruteForceIntersectionPairs(collinear),
    "collinear family vs oracle");

  const disjoint = [[0, 0, 1, 1], [3, 0, 4, 1], [5, 5, 6, 6]];
  assert.deepStrictEqual(sweepIntersectionPairs(disjoint), [], "disjoint set");
  assert.deepStrictEqual(sweepIntersectionPairs([]), [], "empty input");

  // --- LCG cross-checks against the O(n^2) oracle -----------------------
  const lcg = new Lcg(37113);
  for (let trial = 0; trial < 40; trial++) {
    const count = 5 + lcg.next(8);
    const segments = lcgSegments(count, lcg);
    assert.deepStrictEqual(sweepIntersectionPairs(segments), bruteForceIntersectionPairs(segments),
      `random sweep vs brute force ${trial}`);
  }
  console.log("PASS");
}
