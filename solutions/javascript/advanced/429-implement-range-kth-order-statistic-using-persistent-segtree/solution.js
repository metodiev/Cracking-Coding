/** Task 429 - Implement range kth order statistic using persistent segtree. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

class PersistentKthSmallest {
  constructor(values) {
    const sorted = values.slice().sort((a, b) => a - b);
    const distinct = [];
    for (const value of sorted) {
      if (distinct.length === 0 || distinct[distinct.length - 1] !== value) distinct.push(value);
    }
    this.sortedValues = distinct;
    this.domain = distinct.length;
    this.ranks = new Map();
    for (let i = 0; i < this.domain; i++) this.ranks.set(distinct[i], i);
    this.left = [0];
    this.right = [0];
    this.count = [0];
    this.roots = [0];
    let root = 0;
    for (const value of values) {
      root = this.insert(root, 0, this.domain - 1, this.ranks.get(value));
      this.roots.push(root);
    }
  }

  makeNode(left, right, count) {
    this.left.push(left);
    this.right.push(right);
    this.count.push(count);
    return this.count.length - 1;
  }

  insert(node, lo, hi, position) {
    if (lo === hi) return this.makeNode(0, 0, this.count[node] + 1);
    const mid = (lo + hi) >> 1;
    if (position <= mid) {
      const left = this.insert(this.left[node], lo, mid, position);
      const right = this.right[node];
      return this.makeNode(left, right, this.count[left] + this.count[right]);
    }
    const left = this.left[node];
    const right = this.insert(this.right[node], mid + 1, hi, position);
    return this.makeNode(left, right, this.count[left] + this.count[right]);
  }

  kth(from, to, k) {
    let low = this.roots[from];
    let high = this.roots[to + 1];
    let lo = 0;
    let hi = this.domain - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      const leftCount = this.count[this.left[high]] - this.count[this.left[low]];
      if (k <= leftCount) {
        low = this.left[low];
        high = this.left[high];
        hi = mid;
      } else {
        k -= leftCount;
        low = this.right[low];
        high = this.right[high];
        lo = mid + 1;
      }
    }
    return this.sortedValues[lo];
  }
}

function lcgArray(n, seed) {
  const values = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    values.push((seed % 13) - 6);
  }
  return values;
}

module.exports = { PersistentKthSmallest, lcgArray };

if (require.main === module) {
  const assert = require("node:assert");

  const tree = new PersistentKthSmallest([5, 1, 4, 2, 3]);
  assert.strictEqual(tree.kth(0, 4, 1), 1);
  assert.strictEqual(tree.kth(0, 4, 2), 2);
  assert.strictEqual(tree.kth(0, 4, 3), 3);
  assert.strictEqual(tree.kth(0, 4, 4), 4);
  assert.strictEqual(tree.kth(0, 4, 5), 5);
  assert.strictEqual(tree.kth(1, 3, 1), 1);
  assert.strictEqual(tree.kth(1, 3, 2), 2);
  assert.strictEqual(tree.kth(1, 3, 3), 4);
  assert.strictEqual(tree.kth(2, 2, 1), 4);
  assert.strictEqual(tree.kth(4, 4, 1), 3);
  assert.strictEqual(tree.kth(0, 0, 1), 5);

  const dupes = new PersistentKthSmallest([2, 2, 2]);
  assert.strictEqual(dupes.kth(0, 2, 1), 2);
  assert.strictEqual(dupes.kth(0, 2, 2), 2);
  assert.strictEqual(dupes.kth(0, 2, 3), 2);
  assert.strictEqual(dupes.kth(1, 1, 1), 2);

  const negatives = new PersistentKthSmallest([-3, 7, -3, 0]);
  assert.strictEqual(negatives.kth(0, 3, 1), -3);
  assert.strictEqual(negatives.kth(0, 3, 2), -3);
  assert.strictEqual(negatives.kth(0, 3, 3), 0);
  assert.strictEqual(negatives.kth(0, 3, 4), 7);
  assert.strictEqual(negatives.kth(1, 2, 1), -3);
  assert.strictEqual(negatives.kth(1, 2, 2), 7);

  for (const seed of [429, 12345, 54321]) {
    const values = lcgArray(1 + (seed % 40), seed);
    const structure = new PersistentKthSmallest(values);
    let local = seed;
    for (let step = 0; step < 40; step++) {
      local = (local * 75 + 74) % 65537;
      let from = local % values.length;
      let to = ((local / 7) >> 0) % values.length;
      if (from > to) [from, to] = [to, from];
      const segment = values.slice(from, to + 1).sort((a, b) => a - b);
      local = (local * 75 + 74) % 65537;
      const k = 1 + (local % segment.length);
      const got = structure.kth(from, to, k);
      assert.strictEqual(got, segment[k - 1], `step ${step} seed ${seed}`);
    }
  }

  console.log("PASS");
}
