/** Task 365 - Skyline of buildings via divide & conquer (detailed). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
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

/** Returns critical points [x, height]; degenerate buildings (zero width/height) ignored. */
function skylineDivideConquer(buildings) {
  const ordered = buildings
    .filter((b) => b[1] > b[0] && b[2] > 0)
    .map((b) => b.slice())
    .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  return solve(ordered);
}

function solve(ordered) {
  if (ordered.length === 0) return [];
  if (ordered.length === 1) return [[ordered[0][0], ordered[0][2]], [ordered[0][1], 0]];
  const mid = ordered.length >> 1;
  return merge(solve(ordered.slice(0, mid)), solve(ordered.slice(mid)));
}

/** Linear two-pointer merge of two x-sorted skylines of [x, height] critical points. */
function merge(skyA, skyB) {
  const merged = [];
  let i = 0;
  let j = 0;
  let heightA = 0;
  let heightB = 0;
  while (i < skyA.length || j < skyB.length) {
    let x;
    if (j >= skyB.length || (i < skyA.length && skyA[i][0] < skyB[j][0])) {
      x = skyA[i][0];
      heightA = skyA[i][1];
      i++;
    } else if (i >= skyA.length || skyB[j][0] < skyA[i][0]) {
      x = skyB[j][0];
      heightB = skyB[j][1];
      j++;
    } else {
      x = skyA[i][0];
      heightA = skyA[i][1];
      heightB = skyB[j][1];
      i++;
      j++;
    }
    const height = Math.max(heightA, heightB);
    if (merged.length === 0 || merged[merged.length - 1][1] !== height) {
      merged.push([x, height]);
    }
  }
  return merged;
}

/** Independent oracle: sweep every boundary x keeping counts of active heights on [l, r). */
function skylineSweep(buildings) {
  const events = new Map();
  for (const [left, right, height] of buildings) {
    if (right <= left || height <= 0) continue;
    if (!events.has(left)) events.set(left, []);
    if (!events.has(right)) events.set(right, []);
    events.get(left).push([height, 1]);
    events.get(right).push([height, -1]);
  }
  const counts = new Map();
  const result = [];
  const xs = [...events.keys()].sort((a, b) => a - b);
  for (const x of xs) {
    for (const [height, delta] of events.get(x)) {
      counts.set(height, (counts.get(height) || 0) + delta);
    }
    let current = 0;
    for (const [height, count] of counts) {
      if (count > 0) current = Math.max(current, height);
    }
    if (result.length === 0 || result[result.length - 1][1] !== current) {
      result.push([x, current]);
    }
  }
  return result;
}

module.exports = { skylineDivideConquer, skylineSweep, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  // Canonical example: at x=7 the 15-block ends (12 remains); at x=9 the 10-block ends
  // but 12 still stands, so no point is emitted there.
  const canonical = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]];
  const canonicalSkyline = [[2, 10], [3, 15], [7, 12], [12, 0], [15, 10], [20, 8], [24, 0]];
  assert.deepStrictEqual(skylineDivideConquer(canonical), canonicalSkyline);
  assert.deepStrictEqual(skylineSweep(canonical), canonicalSkyline);

  assert.deepStrictEqual(skylineDivideConquer([]), []);
  assert.deepStrictEqual(skylineSweep([]), []);
  assert.deepStrictEqual(skylineDivideConquer([[1, 3, 4]]), [[1, 4], [3, 0]]);
  assert.deepStrictEqual(skylineSweep([[1, 3, 4]]), [[1, 4], [3, 0]]);
  assert.deepStrictEqual(skylineDivideConquer([[0, 2, 5], [2, 4, 7]]), [[0, 5], [2, 7], [4, 0]]);
  assert.deepStrictEqual(skylineDivideConquer([[0, 2, 5], [5, 8, 3]]),
    [[0, 5], [2, 0], [5, 3], [8, 0]]);
  assert.deepStrictEqual(skylineDivideConquer([[0, 10, 5], [2, 4, 3]]), [[0, 5], [10, 0]]);
  assert.deepStrictEqual(skylineDivideConquer([[0, 10, 5], [2, 4, 9]]),
    [[0, 5], [2, 9], [4, 5], [10, 0]]);
  assert.deepStrictEqual(skylineDivideConquer([[0, 4, 3], [0, 4, 5]]), [[0, 5], [4, 0]]);
  assert.deepStrictEqual(skylineDivideConquer([[0, 4, 5], [0, 4, 5], [1, 2, 1]]),
    [[0, 5], [4, 0]]);

  // Degenerate buildings are ignored by both implementations.
  assert.deepStrictEqual(skylineDivideConquer([[3, 3, 9]]), []);
  assert.deepStrictEqual(skylineDivideConquer([[0, 5, 0]]), []);
  assert.deepStrictEqual(skylineDivideConquer([[0, 5, 4], [2, 2, 9]]), [[0, 4], [5, 0]]);
  assert.deepStrictEqual(skylineSweep([[0, 5, 4], [2, 2, 9]]), [[0, 4], [5, 0]]);

  // LCG cross-check with many building sets; both algorithms must agree exactly.
  const lcg = new Lcg(8113);
  for (let trial = 0; trial < 400; trial++) {
    const count = 1 + lcg.next(10);
    const buildings = [];
    for (let i = 0; i < count; i++) {
      const left = lcg.next(50);
      const width = 1 + lcg.next(12);
      const height = 1 + lcg.next(15);
      buildings.push([left, left + width, height]);
    }
    assert.deepStrictEqual(skylineDivideConquer(buildings), skylineSweep(buildings),
      `trial ${trial}`);
  }
  console.log("PASS");
}
