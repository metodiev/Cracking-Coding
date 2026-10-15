/** Task 329 - Range intersection queries (interval trees basics). (Tier: Intermediate, CTCI Ch 4) */
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

/** Augmented BST over interval starts with subtree max-end values. */
class IntervalTree {
  constructor(intervals) {
    const ordered = intervals
      .map((iv) => iv.slice())
      .sort((u, v) => u[0] - v[0] || u[1] - v[1]);
    this.root = IntervalTree.build(ordered, 0, ordered.length);
  }

  static build(ordered, from, to) {
    if (from >= to) return null;
    const mid = (from + to) >> 1;
    const node = {
      start: ordered[mid][0],
      end: ordered[mid][1],
      maxEnd: ordered[mid][1],
      left: null,
      right: null,
    };
    node.left = IntervalTree.build(ordered, from, mid);
    node.right = IntervalTree.build(ordered, mid + 1, to);
    if (node.left !== null) node.maxEnd = Math.max(node.maxEnd, node.left.maxEnd);
    if (node.right !== null) node.maxEnd = Math.max(node.maxEnd, node.right.maxEnd);
    return node;
  }

  /** All stored intervals intersecting the closed range [lo, hi], in start order. */
  query(lo, hi) {
    const found = [];
    IntervalTree.walk(this.root, lo, hi, found);
    return found;
  }

  static walk(node, lo, hi, found) {
    if (node === null) return;
    if (node.left !== null && node.left.maxEnd >= lo) IntervalTree.walk(node.left, lo, hi, found);
    if (node.start <= hi && node.end >= lo) found.push([node.start, node.end]);
    if (node.start <= hi) IntervalTree.walk(node.right, lo, hi, found);
  }
}

/** Independent oracle: filter every interval. */
function naiveQuery(intervals, lo, hi) {
  return intervals
    .filter((iv) => iv[0] <= hi && iv[1] >= lo)
    .map((iv) => iv.slice())
    .sort((u, v) => u[0] - v[0] || u[1] - v[1]);
}

function sortedPoints(points) {
  return points.map((p) => p.slice()).sort((u, v) => u[0] - v[0] || u[1] - v[1]);
}

function lcgIntervals(count, lcg) {
  const intervals = [];
  for (let i = 0; i < count; i++) {
    const start = lcg.next(60);
    intervals.push([start, start + 1 + lcg.next(12)]);
  }
  return intervals;
}

const FIXED = [[15, 20], [10, 30], [17, 19], [5, 20], [12, 15], [30, 40]];

module.exports = { IntervalTree, naiveQuery, sortedPoints, lcgIntervals, Lcg };

if (require.main === module) {
  const assert = require("node:assert");
  const tree = new IntervalTree(FIXED);
  assert.deepStrictEqual(tree.query(6, 7), [[5, 20]]);
  assert.deepStrictEqual(tree.query(14, 16), [[5, 20], [10, 30], [12, 15], [15, 20]]);
  assert.deepStrictEqual(tree.query(30, 35), [[10, 30], [30, 40]]);
  assert.deepStrictEqual(tree.query(19, 19), [[5, 20], [10, 30], [15, 20], [17, 19]]);
  assert.deepStrictEqual(tree.query(0, 4), []);
  assert.deepStrictEqual(tree.query(41, 50), []);

  assert.deepStrictEqual(new IntervalTree([]).query(0, 10), []);
  assert.deepStrictEqual(new IntervalTree([[4, 4]]).query(4, 4), [[4, 4]]); // point interval

  const lcg = new Lcg(7071);
  for (let trial = 0; trial < 40; trial++) {
    const intervals = lcgIntervals(30, lcg);
    const built = new IntervalTree(intervals);
    for (let q = 0; q < 30; q++) {
      const lo = lcg.next(70);
      const hi = lo + lcg.next(25);
      assert.deepStrictEqual(
        sortedPoints(built.query(lo, hi)),
        naiveQuery(intervals, lo, hi),
        `trial ${trial}/${q}`
      );
    }
  }
  console.log("PASS");
}
