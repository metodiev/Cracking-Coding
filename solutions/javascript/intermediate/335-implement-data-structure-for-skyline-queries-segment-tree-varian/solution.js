/** Task 335 - Implement data structure for skyline queries (segment tree variant). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

/**
 * A segment tree over the x-range supports "raise the skyline to height h on [l, r)"
 * updates, point queries (skyline height at x) and range-max queries (tallest point in
 * [l, r)), each O(log n). Updates only raise heights, so applying max is monotone and no
 * push-down is needed. Cross-checked against a brute-force per-position max array.
 */

class SkylineTree {
  constructor(n) {
    if (n < 1) throw new Error("n must be >= 1");
    let size = 1;
    while (size < n) size <<= 1;
    this.n = n;
    this.size = size;
    this.tag = new Array(2 * size).fill(0);    // height applied to this node's whole range
    this.best = new Array(2 * size).fill(0);   // tallest point in this node's range
  }

  /** Raises the skyline to at least `height` on the half-open range [left, right). */
  update(left, right, height) {
    if (left < 0 || left >= right || right > this.n) throw new Error("invalid range");
    this.updateNode(1, 0, this.size, left, right, height);
  }

  /** Skyline height at `index`. */
  point(index) {
    if (index < 0 || index >= this.n) throw new Error("index out of range");
    let node = 1, lo = 0, hi = this.size, height = 0;
    for (;;) {
      height = Math.max(height, this.tag[node]);
      if (hi - lo === 1) return height;
      const mid = (lo + hi) >> 1;
      if (index < mid) {
        node = 2 * node;
        hi = mid;
      } else {
        node = 2 * node + 1;
        lo = mid;
      }
    }
  }

  /** Tallest point in [left, right); 0 when the range is empty. */
  rangeMax(left, right) {
    if (left < 0 || left > right || right > this.n) throw new Error("invalid range");
    if (left === right) return 0;
    return this.rangeMaxNode(1, 0, this.size, left, right);
  }

  skyline() {
    const heights = [];
    for (let i = 0; i < this.n; i++) heights.push(this.point(i));
    return heights;
  }

  updateNode(node, lo, hi, left, right, height) {
    if (right <= lo || hi <= left) return;
    if (left <= lo && hi <= right) {
      this.tag[node] = Math.max(this.tag[node], height);
      this.best[node] = Math.max(this.best[node], height);
      return;
    }
    const mid = (lo + hi) >> 1;
    this.updateNode(2 * node, lo, mid, left, right, height);
    this.updateNode(2 * node + 1, mid, hi, left, right, height);
    this.best[node] = Math.max(this.tag[node], this.best[2 * node], this.best[2 * node + 1]);
  }

  rangeMaxNode(node, lo, hi, left, right) {
    if (left >= right || right <= lo || hi <= left) return 0;
    if (left <= lo && hi <= right) return this.best[node];
    const mid = (lo + hi) >> 1;
    return Math.max(this.tag[node],
      this.rangeMaxNode(2 * node, lo, mid, left, right),
      this.rangeMaxNode(2 * node + 1, mid, hi, left, right));
  }
}

/** Oracle: keep a height per position and apply each update directly. */
function bruteSchedule(n, updates) {
  const heights = new Array(n).fill(0);
  for (const [left, right, height] of updates) {
    for (let i = left; i < right; i++) heights[i] = Math.max(heights[i], height);
  }
  return heights;
}

let seed = 7788;

function lcgUpdates(count, n) {
  const updates = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    const left = seed % n;
    seed = (seed * 75 + 74) % 65537;
    const right = left + 1 + (seed % (n - left));
    seed = (seed * 75 + 74) % 65537;
    updates.push([left, right, seed % 100]);
  }
  return updates;
}

module.exports = { SkylineTree, bruteSchedule };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-checked: [1,4)->5, [2,3)->7, [4,6)->2 over 7 positions.
  const tree = new SkylineTree(7);
  for (const update of [[1, 4, 5], [2, 3, 7], [4, 6, 2]]) tree.update(...update);
  assert.deepStrictEqual(tree.skyline(), [0, 5, 7, 5, 2, 2, 0]);
  assert.strictEqual(tree.point(2), 7);
  assert.strictEqual(tree.point(6), 0);
  assert.strictEqual(tree.point(0), 0);
  assert.strictEqual(tree.rangeMax(0, 3), 7);
  assert.strictEqual(tree.rangeMax(2, 5), 7);
  assert.strictEqual(tree.rangeMax(3, 6), 5);
  assert.strictEqual(tree.rangeMax(0, 7), 7);
  assert.strictEqual(tree.rangeMax(5, 5), 0);    // empty range
  assert.strictEqual(tree.rangeMax(6, 7), 0);

  // Updates never lower a height: a smaller update is a no-op, a larger one raises.
  tree.update(2, 3, 4);
  assert.strictEqual(tree.point(2), 7);
  tree.update(2, 3, 9);
  assert.strictEqual(tree.point(2), 9);
  assert.strictEqual(tree.rangeMax(1, 3), 9);

  const tiny = new SkylineTree(1);
  assert.strictEqual(tiny.point(0), 0);
  assert.strictEqual(tiny.rangeMax(0, 1), 0);
  tiny.update(0, 1, 42);
  assert.strictEqual(tiny.point(0), 42);
  assert.strictEqual(tiny.rangeMax(0, 1), 42);

  const boundary = new SkylineTree(4);
  for (const update of [[0, 4, 1], [0, 1, 2], [3, 4, 3], [1, 3, 4]]) boundary.update(...update);
  assert.deepStrictEqual(boundary.skyline(), [2, 4, 4, 3]);
  assert.strictEqual(boundary.rangeMax(0, 4), 4);
  assert.strictEqual(boundary.rangeMax(0, 2), 4);
  assert.strictEqual(boundary.rangeMax(2, 4), 4);
  assert.strictEqual(boundary.rangeMax(3, 4), 3);

  for (let trial = 0; trial < 20; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 1 + (seed % 40);
    const updates = lcgUpdates(1 + (seed % 60), n);
    const randomTree = new SkylineTree(n);
    for (const update of updates) randomTree.update(...update);
    const expected = bruteSchedule(n, updates);
    assert.deepStrictEqual(randomTree.skyline(), expected, `random skyline ${trial}`);
    for (let left = 0; left < n; left++) {
      for (let right = left + 1; right <= n; right++) {
        const naive = Math.max(...expected.slice(left, right));
        assert.strictEqual(randomTree.rangeMax(left, right), naive, `range ${trial} ${left} ${right}`);
      }
    }
  }
  console.log("PASS");
}
