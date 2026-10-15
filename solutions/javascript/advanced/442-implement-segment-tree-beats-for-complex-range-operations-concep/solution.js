/** Task 442 - Implement segment tree beats for complex range operations (conceptual). (Tier: Advanced, CTCI Ch 4) */
"use strict";

class SegmentTreeBeats {
  constructor(values) {
    this.n = values.length;
    const size = 4 * Math.max(1, this.n);
    this.nodeMax = new Array(size).fill(0);
    this.nodeSum = new Array(size).fill(0);
    if (this.n > 0) this._build(1, 0, this.n - 1, values);
  }

  _build(node, lo, hi, values) {
    if (lo === hi) {
      this.nodeMax[node] = values[lo];
      this.nodeSum[node] = values[lo];
      return;
    }
    const mid = (lo + hi) >> 1;
    this._build(2 * node, lo, mid, values);
    this._build(2 * node + 1, mid + 1, hi, values);
    this.nodeMax[node] = Math.max(this.nodeMax[2 * node], this.nodeMax[2 * node + 1]);
    this.nodeSum[node] = this.nodeSum[2 * node] + this.nodeSum[2 * node + 1];
  }

  chmin(left, right, cap) {
    this._chmin(1, 0, this.n - 1, left, right, cap);
  }

  _chmin(node, lo, hi, left, right, cap) {
    if (right < lo || hi < left || this.nodeMax[node] <= cap) return;
    if (lo === hi) {
      this.nodeMax[node] = cap;
      this.nodeSum[node] = cap;
      return;
    }
    const mid = (lo + hi) >> 1;
    this._chmin(2 * node, lo, mid, left, right, cap);
    this._chmin(2 * node + 1, mid + 1, hi, left, right, cap);
    this.nodeMax[node] = Math.max(this.nodeMax[2 * node], this.nodeMax[2 * node + 1]);
    this.nodeSum[node] = this.nodeSum[2 * node] + this.nodeSum[2 * node + 1];
  }

  rangeMax(left, right) {
    return this._rangeMax(1, 0, this.n - 1, left, right);
  }

  _rangeMax(node, lo, hi, left, right) {
    if (left <= lo && hi <= right) return this.nodeMax[node];
    const mid = (lo + hi) >> 1;
    let best = -Infinity;
    if (left <= mid) best = Math.max(best, this._rangeMax(2 * node, lo, mid, left, right));
    if (right > mid) best = Math.max(best, this._rangeMax(2 * node + 1, mid + 1, hi, left, right));
    return best;
  }

  rangeSum(left, right) {
    return this._rangeSum(1, 0, this.n - 1, left, right);
  }

  _rangeSum(node, lo, hi, left, right) {
    if (left <= lo && hi <= right) return this.nodeSum[node];
    const mid = (lo + hi) >> 1;
    let total = 0;
    if (left <= mid) total += this._rangeSum(2 * node, lo, mid, left, right);
    if (right > mid) total += this._rangeSum(2 * node + 1, mid + 1, hi, left, right);
    return total;
  }
}

function lcgOps(count, seed, n) {
  const ops = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    const lo = seed % n;
    seed = (seed * 75 + 74) % 65537;
    const hi = lo + (seed % (n - lo));
    seed = (seed * 75 + 74) % 65537;
    const kind = seed % 3;
    seed = (seed * 75 + 74) % 65537;
    const cap = seed % 41 - 20;
    ops.push([kind, lo, hi, cap]);
  }
  return ops;
}

module.exports = { SegmentTreeBeats, lcgOps };

if (require.main === module) {
  const assert = require("node:assert");

  let tree = new SegmentTreeBeats([1, 5, 3, 4, 2]);
  assert.strictEqual(tree.rangeMax(0, 4), 5);
  assert.strictEqual(tree.rangeSum(0, 4), 15);
  tree.chmin(1, 3, 3);
  assert.strictEqual(tree.rangeMax(0, 4), 3);
  assert.strictEqual(tree.rangeMax(1, 1), 3);
  assert.strictEqual(tree.rangeSum(0, 4), 12);
  assert.strictEqual(tree.rangeSum(2, 3), 6);
  tree.chmin(0, 4, -1);
  assert.strictEqual(tree.rangeMax(0, 4), -1);
  assert.strictEqual(tree.rangeSum(0, 4), -5);
  tree.chmin(2, 2, 10);
  assert.strictEqual(tree.rangeMax(0, 4), -1);
  assert.strictEqual(tree.rangeSum(0, 4), -5);

  const n = 16;
  let seed = 1234;
  const base = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    base.push(seed % 50 - 25);
  }
  tree = new SegmentTreeBeats(base);
  const brute = base.slice();
  for (const [kind, lo, hi, cap] of lcgOps(300, 777, n)) {
    if (kind === 0) {
      tree.chmin(lo, hi, cap);
      for (let i = lo; i <= hi; i++) brute[i] = Math.min(brute[i], cap);
    } else if (kind === 1) {
      let best = -Infinity;
      for (let i = lo; i <= hi; i++) best = Math.max(best, brute[i]);
      assert.strictEqual(tree.rangeMax(lo, hi), best, `max ${lo}..${hi}`);
    } else {
      let total = 0;
      for (let i = lo; i <= hi; i++) total += brute[i];
      assert.strictEqual(tree.rangeSum(lo, hi), total, `sum ${lo}..${hi}`);
    }
  }
  console.log("PASS");
}
