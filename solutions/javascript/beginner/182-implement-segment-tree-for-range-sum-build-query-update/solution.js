/** Task 182 - Implement segment tree for range sum (build/query/update). (Tier: Beginner, CTCI Ch 4) */
"use strict";

/**
 * Recursive segment tree in a flat array of size 4n; node 1 is the root and covers
 * [0, n - 1], the children of node i are 2i and 2i + 1. Build O(n), point update and
 * inclusive range sum O(log n).
 */
class SegmentTree {
  constructor(values) {
    this.n = values.length;
    this.tree = new Array(4 * Math.max(this.n, 1)).fill(0);
    if (this.n > 0) this.build(1, 0, this.n - 1, values);
  }

  build(node, lo, hi, values) {
    if (lo === hi) {
      this.tree[node] = values[lo];
      return;
    }
    const mid = (lo + hi) >> 1;
    this.build(2 * node, lo, mid, values);
    this.build(2 * node + 1, mid + 1, hi, values);
    this.tree[node] = this.tree[2 * node] + this.tree[2 * node + 1];
  }

  update(index, value) {
    if (index < 0 || index >= this.n) throw new RangeError("index out of range");
    this.updateNode(1, 0, this.n - 1, index, value);
  }

  updateNode(node, lo, hi, index, value) {
    if (lo === hi) {
      this.tree[node] = value;
      return;
    }
    const mid = (lo + hi) >> 1;
    if (index <= mid) this.updateNode(2 * node, lo, mid, index, value);
    else this.updateNode(2 * node + 1, mid + 1, hi, index, value);
    this.tree[node] = this.tree[2 * node] + this.tree[2 * node + 1];
  }

  rangeSum(lo, hi) {
    if (lo > hi) return 0;
    if (lo < 0 || hi >= this.n) throw new RangeError("range out of bounds");
    return this.query(1, 0, this.n - 1, lo, hi);
  }

  query(node, lo, hi, qlo, qhi) {
    if (qlo <= lo && hi <= qhi) return this.tree[node];
    const mid = (lo + hi) >> 1;
    let total = 0;
    if (qlo <= mid) total += this.query(2 * node, lo, mid, qlo, qhi);
    if (qhi > mid) total += this.query(2 * node + 1, mid + 1, hi, qlo, qhi);
    return total;
  }
}

const allRanges = (st, size) => {
  const sums = [];
  for (let lo = 0; lo < size; lo++) {
    for (let hi = lo; hi < size; hi++) sums.push([lo, hi, st.rangeSum(lo, hi)]);
  }
  return sums;
};

module.exports = { SegmentTree };

if (require.main === module) {
  const assert = require("node:assert");

  const values = [1, 3, 5, 7, 9, 11];
  const st = new SegmentTree(values);
  assert.strictEqual(st.rangeSum(0, 5), 36);
  assert.strictEqual(st.rangeSum(0, 0), 1);
  assert.strictEqual(st.rangeSum(5, 5), 11);
  assert.strictEqual(st.rangeSum(2, 3), 12);
  assert.strictEqual(st.rangeSum(3, 2), 0);

  st.update(2, 10);
  values[2] = 10;
  assert.strictEqual(st.rangeSum(1, 3), 20);
  st.update(0, -4);
  values[0] = -4;
  assert.strictEqual(st.rangeSum(0, 5), 36);
  assert.strictEqual(st.rangeSum(0, 0), -4);

  const single = new SegmentTree([7]);
  assert.strictEqual(single.rangeSum(0, 0), 7);
  single.update(0, 2);
  assert.strictEqual(single.rangeSum(0, 0), 2);

  const empty = new SegmentTree([]);
  assert.strictEqual(empty.rangeSum(0, -1), 0);

  assert.throws(() => st.rangeSum(-1, 2), RangeError);
  assert.throws(() => st.update(6, 0), RangeError);

  // cross-check every range against a plain array after each random update
  const data = [4, -7, 12, 0, 5, -3, 9, 8, -1, 6, -11, 3];
  const st2 = new SegmentTree(data);
  let state = 21;
  const nextInt = (bound) => {
    state = (state * 48271) % 2147483647;
    return state % bound;
  };
  const naiveSum = (lo, hi) => {
    let total = 0;
    for (let k = lo; k <= hi; k++) total += data[k];
    return total;
  };
  for (let t = 0; t < 40; t++) {
    const i = nextInt(data.length);
    const value = nextInt(41) - 20;
    data[i] = value;
    st2.update(i, value);
    for (const [lo, hi, actual] of allRanges(st2, data.length)) {
      assert.strictEqual(actual, naiveSum(lo, hi), `range ${lo}..${hi}`);
    }
  }
  console.log("PASS");
}
