/** Task 361 - Implement interval stabbing queries with segment tree. (Tier: Intermediate, CTCI Ch 4) */
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

function lowerBound(values, target) {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function upperBound(values, target) {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] <= target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * All query points are known up front, so they are coordinate-compressed to indices and
 * the segment tree stores cover-count adds; a point query walks leaf -> root.
 */
class StabbingSegmentTree {
  constructor(points) {
    this.points = [...new Set(points)].sort((a, b) => a - b);
    this.n = this.points.length;
    this.tree = new Array(2 * Math.max(this.n, 1)).fill(0);
  }

  addInterval(left, right) {
    if (this.n === 0) return;
    let l = lowerBound(this.points, left) + this.n;
    let r = upperBound(this.points, right) + this.n;
    while (l < r) {
      if (l & 1) this.tree[l++] += 1;
      if (r & 1) this.tree[--r] += 1;
      l >>= 1;
      r >>= 1;
    }
  }

  coverCount(point) {
    const index = lowerBound(this.points, point);
    if (index >= this.n || this.points[index] !== point) {
      throw new Error("point was not registered");
    }
    let total = 0;
    for (let i = index + this.n; i >= 1; i >>= 1) total += this.tree[i];
    return total;
  }
}

function naiveCoverCount(intervals, point) {
  return intervals.filter(([left, right]) => left <= point && point <= right).length;
}

module.exports = { StabbingSegmentTree, naiveCoverCount, lowerBound, upperBound, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-built example from the task description: intervals [1,4], [3,6], [5,8].
  const intervals = [[1, 4], [3, 6], [5, 8]];
  const points = [0, 2, 4, 6, 7, 9];
  const expected = [0, 1, 2, 2, 1, 0];
  const tree = new StabbingSegmentTree(points);
  for (const [left, right] of intervals) tree.addInterval(left, right);
  assert.deepStrictEqual(points.map((p) => tree.coverCount(p)), expected);
  points.forEach((p, k) => {
    assert.strictEqual(naiveCoverCount(intervals, p), expected[k], `naive at ${p}`);
  });

  // Adding the same interval twice doubles the counts.
  const doubled = new StabbingSegmentTree(points);
  for (const [left, right] of intervals) {
    doubled.addInterval(left, right);
    doubled.addInterval(left, right);
  }
  assert.deepStrictEqual(points.map((p) => doubled.coverCount(p)),
    expected.map((value) => 2 * value));

  // Duplicate query points collapse; single point works; no points is a no-op.
  const dup = new StabbingSegmentTree([3, 3, 5, 3]);
  dup.addInterval(1, 4);
  assert.deepStrictEqual(dup.points, [3, 5]);
  assert.strictEqual(dup.coverCount(3), 1);
  assert.strictEqual(dup.coverCount(5), 0);
  const solo = new StabbingSegmentTree([7]);
  solo.addInterval(7, 7);
  solo.addInterval(8, 9);
  assert.strictEqual(solo.coverCount(7), 1);
  new StabbingSegmentTree([]).addInterval(-3, 12);   // must not fail

  // LCG cross-check against the naive cover count.
  const lcg = new Lcg(20261);
  for (let trial = 0; trial < 25; trial++) {
    const count = 1 + lcg.next(40);
    const randomIntervals = [];
    for (let i = 0; i < count; i++) {
      const left = lcg.next(30);
      const width = 1 + lcg.next(10);
      randomIntervals.push([left, left + width]);
    }
    const probe = [];
    for (let p = -2; p < 46; p++) probe.push(p);
    const randomTree = new StabbingSegmentTree(probe);
    for (const [left, right] of randomIntervals) randomTree.addInterval(left, right);
    for (const p of probe) {
      assert.strictEqual(randomTree.coverCount(p), naiveCoverCount(randomIntervals, p),
        `trial ${trial} point ${p}`);
    }
  }
  console.log("PASS");
}
