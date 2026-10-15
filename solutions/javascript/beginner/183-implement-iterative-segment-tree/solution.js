/** Task 183 - Implement iterative segment tree. (Tier: Beginner, CTCI Ch 4) */
"use strict";

/**
 * Bottom-up segment tree in a flat array of size 2n: leaves live at [n, 2n) and
 * parents at [1, n) with tree[i] = tree[2i] + tree[2i + 1]. Everything is a loop, so
 * there is no recursion overhead; compared to the recursive tree of task 182 it needs
 * 2n slots instead of 4n and never recurses, while producing the same answers.
 */
class IterativeSegmentTree {
  constructor(values) {
    this.n = values.length;
    this.tree = new Array(2 * this.n).fill(0);
    for (let i = 0; i < this.n; i++) this.tree[this.n + i] = values[i];
    for (let i = this.n - 1; i >= 1; i--) this.tree[i] = this.tree[2 * i] + this.tree[2 * i + 1];
  }

  update(index, value) {
    if (index < 0 || index >= this.n) throw new RangeError("index out of range");
    let i = this.n + index;
    this.tree[i] = value;
    for (i >>= 1; i >= 1; i >>= 1) {
      this.tree[i] = this.tree[2 * i] + this.tree[2 * i + 1];
    }
  }

  rangeSum(lo, hi) {
    if (lo > hi) return 0;
    if (lo < 0 || hi >= this.n) throw new RangeError("range out of bounds");
    let total = 0;
    let left = lo + this.n;
    let right = hi + 1 + this.n; // classic half-open [left, right)
    while (left < right) {
      if (left & 1) total += this.tree[left++];
      if (right & 1) total += this.tree[--right];
      left >>= 1;
      right >>= 1;
    }
    return total;
  }
}

module.exports = { IterativeSegmentTree };

if (require.main === module) {
  const assert = require("node:assert");

  const values = [1, 3, 5, 7, 9, 11];
  const st = new IterativeSegmentTree(values);
  assert.strictEqual(st.rangeSum(0, 5), 36);
  assert.strictEqual(st.rangeSum(0, 0), 1);
  assert.strictEqual(st.rangeSum(5, 5), 11);
  assert.strictEqual(st.rangeSum(2, 3), 12);
  assert.strictEqual(st.rangeSum(1, 4), 24);
  assert.strictEqual(st.rangeSum(3, 2), 0);

  st.update(2, 10);
  values[2] = 10;
  assert.strictEqual(st.rangeSum(0, 5), 41);
  assert.strictEqual(st.rangeSum(2, 2), 10);
  st.update(0, -4);
  values[0] = -4;
  assert.strictEqual(st.rangeSum(0, 5), 36);

  const single = new IterativeSegmentTree([7]);
  assert.strictEqual(single.rangeSum(0, 0), 7);
  single.update(0, 2);
  assert.strictEqual(single.rangeSum(0, 0), 2);

  const empty = new IterativeSegmentTree([]);
  assert.strictEqual(empty.rangeSum(0, -1), 0);

  assert.throws(() => st.rangeSum(-1, 2), RangeError);
  assert.throws(() => st.update(6, 0), RangeError);

  // cross-check every range against a naive loop after each random update
  const data = [4, -7, 12, 0, 5, -3, 9, 8, -1, 6, -11, 3, 15, -2, 7, 1, -6];
  const st2 = new IterativeSegmentTree(data);
  let state = 33;
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
    for (let lo = 0; lo < data.length; lo++) {
      for (let hi = lo; hi < data.length; hi++) {
        assert.strictEqual(st2.rangeSum(lo, hi), naiveSum(lo, hi), `range ${lo}..${hi}`);
      }
    }
  }
  console.log("PASS");
}
