/** Task 369 - Closest pair of points (divide & conquer). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
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

function dist2(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

/**
 * O(n log n) divide & conquer.  Each recursive call returns its slice sorted by y
 * (merged from the children), so the strip is already y-sorted; strip pairs with
 * dy^2 >= best_so_far cannot improve the answer and are skipped.  Distances are
 * squared exact integers - no epsilon anywhere.
 */
function closestPair(points) {
  const pts = points.map((p) => p.slice()).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 2) throw new Error("need at least two points");
  const result = rec(pts);
  return { distance: result.best, pair: [result.pair[0].slice(), result.pair[1].slice()] };
}

function rec(ptsByX) {
  const n = ptsByX.length;
  let best;
  let pair;
  if (n <= 3) {
    best = -1;
    pair = null;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = dist2(ptsByX[i], ptsByX[j]);
        if (best < 0 || d < best) {
          best = d;
          pair = [ptsByX[i], ptsByX[j]];
        }
      }
    }
    return { best, pair, byY: sortByY(ptsByX) };
  }

  const mid = n >> 1;
  const left = rec(ptsByX.slice(0, mid));
  const right = rec(ptsByX.slice(mid));
  best = left.best <= right.best ? left.best : right.best;
  pair = left.best <= right.best ? left.pair : right.pair;

  const mergedY = mergeByY(left.byY, right.byY);
  const midX = ptsByX[mid][0];
  const strip = mergedY.filter((p) => {
    const dx = p[0] - midX;
    return dx * dx < best;
  });
  for (let i = 0; i < strip.length; i++) {
    for (let j = i + 1; j < strip.length; j++) {
      const dy = strip[j][1] - strip[i][1];
      if (dy * dy >= best) break;
      const d = dist2(strip[i], strip[j]);
      if (d < best) {
        best = d;
        pair = [strip[i], strip[j]];
      }
    }
  }
  return { best, pair, byY: mergedY };
}

function sortByY(pts) {
  return pts.slice().sort((a, b) => a[1] - b[1] || a[0] - b[0]);
}

function mergeByY(a, b) {
  const merged = [];
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    if (j >= b.length || (i < a.length &&
        (a[i][1] < b[j][1] || (a[i][1] === b[j][1] && a[i][0] <= b[j][0])))) {
      merged.push(a[i++]);
    } else {
      merged.push(b[j++]);
    }
  }
  return merged;
}

/** Independent O(n^2) oracle: same (squared distance) semantics. */
function bruteForce(points) {
  let best = -1;
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const d = dist2(points[i], points[j]);
      if (best < 0 || d < best) best = d;
    }
  }
  return best;
}

module.exports = { closestPair, bruteForce, dist2, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand case: (0,0)-(3,4) and (3,4)-(7,1) both have squared distance 25.
  const hand = [[0, 0], [3, 4], [7, 1]];
  let result = closestPair(hand);
  assert.strictEqual(result.distance, 25);
  assert.strictEqual(dist2(result.pair[0], result.pair[1]), 25);
  assert.strictEqual(bruteForce(hand), 25);

  // Duplicate point: distance 0.
  const dup = [[5, 5], [5, 5], [9, 9]];
  result = closestPair(dup);
  assert.strictEqual(result.distance, 0);
  assert.deepStrictEqual(result.pair, [[5, 5], [5, 5]]);
  assert.strictEqual(bruteForce(dup), 0);

  assert.strictEqual(closestPair([[0, 0], [-3, -4]]).distance, 25);

  const grid = [];
  for (let x = 0; x < 4; x++) for (let y = 0; y < 4; y++) grid.push([x, y]);
  result = closestPair(grid);
  assert.strictEqual(result.distance, 1);
  assert.strictEqual(dist2(result.pair[0], result.pair[1]), 1);
  assert.strictEqual(bruteForce(grid), 1);

  // LCG cross-check: sizes up to ~60.
  const lcg = new Lcg(271828);
  for (let trial = 0; trial < 30; trial++) {
    const size = 2 + lcg.next(59);
    const modulus = 10 + lcg.next(90);
    const points = [];
    for (let i = 0; i < size; i++) points.push([lcg.next(modulus), lcg.next(modulus)]);
    result = closestPair(points);
    assert.strictEqual(result.distance, bruteForce(points), `trial ${trial} size ${size}`);
    assert.strictEqual(dist2(result.pair[0], result.pair[1]), result.distance,
      `pair attains best trial ${trial}`);
    const keys = new Set(points.map((p) => `${p[0]},${p[1]}`));
    for (const p of result.pair) assert.ok(keys.has(`${p[0]},${p[1]}`), `member trial ${trial}`);
  }

  // Concentrated points: many duplicates and very small distances.
  for (let trial = 0; trial < 10; trial++) {
    const size = 2 + lcg.next(30);
    const points = [];
    for (let i = 0; i < size; i++) points.push([lcg.next(5), lcg.next(5)]);
    assert.strictEqual(closestPair(points).distance, bruteForce(points), `dense ${trial}`);
  }

  // Worst-case strip shape: a vertical line.
  const line = [];
  for (let y = 0; y < 60; y++) line.push([7, y]);
  assert.strictEqual(closestPair(line).distance, 1);
  console.log("PASS");
}
