/** Task 421 - Implement persistent segment tree (versioned). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

class PersistentSegTree {
  constructor(values) {
    this.n = values.length;
    this.left = [0];                    // node 0 acts as the null node
    this.right = [0];
    this.total = [0];
    this.roots = [this.build(values, 0, this.n - 1)];
  }

  versionCount() {
    return this.roots.length;
  }

  makeNode(left, right, total) {
    this.left.push(left);
    this.right.push(right);
    this.total.push(total);
    return this.total.length - 1;
  }

  build(values, lo, hi) {
    if (lo === hi) return this.makeNode(0, 0, values[lo]);
    const mid = (lo + hi) >> 1;
    const left = this.build(values, lo, mid);
    const right = this.build(values, mid + 1, hi);
    return this.makeNode(left, right, this.total[left] + this.total[right]);
  }

  update(version, pos, value) {
    const root = this.updateNode(this.roots[version], 0, this.n - 1, pos, value);
    this.roots.push(root);
    return this.roots.length - 1;
  }

  updateNode(node, lo, hi, pos, value) {
    if (lo === hi) return this.makeNode(0, 0, value);
    const mid = (lo + hi) >> 1;
    if (pos <= mid) {
      const left = this.updateNode(this.left[node], lo, mid, pos, value);
      const right = this.right[node];
      return this.makeNode(left, right, this.total[left] + this.total[right]);
    }
    const left = this.left[node];
    const right = this.updateNode(this.right[node], mid + 1, hi, pos, value);
    return this.makeNode(left, right, this.total[left] + this.total[right]);
  }

  rangeSum(version, lo, hi) {
    return this.rangeSumNode(this.roots[version], 0, this.n - 1, lo, hi);
  }

  rangeSumNode(node, lo, hi, ql, qr) {
    if (node === 0 || qr < lo || hi < ql) return 0;
    if (ql <= lo && hi <= qr) return this.total[node];
    const mid = (lo + hi) >> 1;
    return this.rangeSumNode(this.left[node], lo, mid, ql, qr)
         + this.rangeSumNode(this.right[node], mid + 1, hi, ql, qr);
  }
}

function lcgArray(n, seed) {
  const values = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    values.push((seed % 41) - 20);
  }
  return values;
}

module.exports = { PersistentSegTree, lcgArray };

if (require.main === module) {
  const assert = require("node:assert");

  const st = new PersistentSegTree([1, 2, 3, 4, 5]);
  assert.strictEqual(st.versionCount(), 1);
  assert.strictEqual(st.rangeSum(0, 0, 4), 15);
  assert.strictEqual(st.rangeSum(0, 1, 3), 9);
  assert.strictEqual(st.rangeSum(0, 2, 2), 3);
  assert.strictEqual(st.rangeSum(0, 0, 0), 1);

  const v1 = st.update(0, 2, 10);                  // [1, 2, 10, 4, 5]
  assert.strictEqual(v1, 1);
  assert.strictEqual(st.rangeSum(1, 0, 4), 22);
  assert.strictEqual(st.rangeSum(1, 2, 2), 10);
  assert.strictEqual(st.rangeSum(1, 1, 3), 16);
  assert.strictEqual(st.rangeSum(0, 0, 4), 15);    // version 0 untouched
  assert.strictEqual(st.rangeSum(0, 2, 2), 3);

  const v2 = st.update(1, 0, -1);                  // [-1, 2, 10, 4, 5]
  assert.strictEqual(v2, 2);
  assert.strictEqual(st.rangeSum(2, 0, 4), 20);
  assert.strictEqual(st.rangeSum(2, 0, 2), 11);
  assert.strictEqual(st.rangeSum(1, 0, 2), 13);

  const v3 = st.update(0, 4, 7);                   // branch off OLD version: [1, 2, 3, 4, 7]
  assert.strictEqual(v3, 3);
  assert.strictEqual(st.rangeSum(3, 0, 4), 17);
  assert.strictEqual(st.rangeSum(3, 3, 4), 11);
  assert.strictEqual(st.rangeSum(3, 2, 2), 3);
  assert.strictEqual(st.rangeSum(2, 0, 4), 20);
  assert.strictEqual(st.rangeSum(1, 0, 4), 22);
  assert.strictEqual(st.rangeSum(0, 0, 4), 15);

  for (const seed of [11, 421, 9001]) {
    const n = 1 + (seed % 29);
    const values = lcgArray(n, seed);
    const tree = new PersistentSegTree(values);
    const snapshots = [values.slice()];
    let local = seed;
    for (let step = 0; step < 60; step++) {
      local = (local * 75 + 74) % 65537;
      if (local % 3 === 0) {
        const version = local % snapshots.length;
        let left = local % n;
        let right = ((local / n) >> 0) % n;
        if (left > right) [left, right] = [right, left];
        const expected = snapshots[version].slice(left, right + 1).reduce((a, b) => a + b, 0);
        assert.strictEqual(tree.rangeSum(version, left, right), expected, `query seed=${seed}`);
      } else {
        const pos = local % n;
        local = (local * 75 + 74) % 65537;
        const value = (local % 201) - 100;
        const version = tree.update(tree.versionCount() - 1, pos, value);
        const state = snapshots[snapshots.length - 1].slice();
        state[pos] = value;
        snapshots.push(state);
        assert.strictEqual(version, snapshots.length - 1);
        assert.strictEqual(tree.rangeSum(version, 0, n - 1), state.reduce((a, b) => a + b, 0));
      }
    }
  }

  const small = new PersistentSegTree([42]);
  assert.strictEqual(small.rangeSum(0, 0, 0), 42);
  small.update(0, 0, -8);
  assert.strictEqual(small.rangeSum(1, 0, 0), -8);
  assert.strictEqual(small.rangeSum(0, 0, 0), 42);

  console.log("PASS");
}
