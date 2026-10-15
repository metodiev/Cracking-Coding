/** Task 441 - Implement range add + range max with lazy propagation. (Tier: Advanced, CTCI Ch 4) */
"use strict";

class RangeAddRangeMax {
  constructor(values) {
    this.n = values.length;
    this.tree = new Array(4 * Math.max(1, this.n)).fill(0);
    this.lazy = new Array(4 * Math.max(1, this.n)).fill(0);
    if (this.n) this.build(1, 0, this.n - 1, values);
  }

  build(node, lo, hi, values) {
    if (lo === hi) {
      this.tree[node] = values[lo];
      return;
    }
    const mid = (lo + hi) >> 1;
    this.build(2 * node, lo, mid, values);
    this.build(2 * node + 1, mid + 1, hi, values);
    this.tree[node] = Math.max(this.tree[2 * node], this.tree[2 * node + 1]);
  }

  push(node) {
    const delta = this.lazy[node];
    if (delta !== 0) {
      for (const child of [2 * node, 2 * node + 1]) {
        this.tree[child] += delta;
        this.lazy[child] += delta;
      }
      this.lazy[node] = 0;
    }
  }

  add(left, right, value) {
    this.addRange(1, 0, this.n - 1, left, right, value);
  }

  addRange(node, lo, hi, left, right, value) {
    if (right < lo || hi < left) return;
    if (left <= lo && hi <= right) {
      this.tree[node] += value;
      this.lazy[node] += value;
      return;
    }
    this.push(node);
    const mid = (lo + hi) >> 1;
    this.addRange(2 * node, lo, mid, left, right, value);
    this.addRange(2 * node + 1, mid + 1, hi, left, right, value);
    this.tree[node] = Math.max(this.tree[2 * node], this.tree[2 * node + 1]);
  }

  max(left, right) {
    return this.queryMax(1, 0, this.n - 1, left, right);
  }

  queryMax(node, lo, hi, left, right) {
    if (left <= lo && hi <= right) return this.tree[node];
    this.push(node);
    const mid = (lo + hi) >> 1;
    let best = -Infinity;
    if (left <= mid) best = Math.max(best, this.queryMax(2 * node, lo, mid, left, right));
    if (right > mid) best = Math.max(best, this.queryMax(2 * node + 1, mid + 1, hi, left, right));
    return best;
  }
}

function lcgOps(count, seed, n) {
  const ops = [];
  for (let k = 0; k < count; k++) {
    seed = (seed * 75 + 74) % 65537;
    const lo = seed % n;
    seed = (seed * 75 + 74) % 65537;
    const hi = lo + (seed % (n - lo));
    seed = (seed * 75 + 74) % 65537;
    const kind = seed % 2;
    seed = (seed * 75 + 74) % 65537;
    const value = (seed % 21) - 10;
    ops.push([kind, lo, hi, value]);
  }
  return ops;
}

module.exports = { RangeAddRangeMax, lcgOps };

if (require.main === module) {
  const assert = require("node:assert");
  const tree = new RangeAddRangeMax([1, 3, 2, 5, 4]);
  assert.strictEqual(tree.max(0, 4), 5);
  assert.strictEqual(tree.max(3, 3), 5);
  tree.add(1, 3, 2);
  assert.strictEqual(tree.max(0, 4), 7);
  assert.strictEqual(tree.max(0, 0), 1);
  assert.strictEqual(tree.max(1, 1), 5);
  assert.strictEqual(tree.max(2, 3), 7);
  tree.add(0, 4, -10);
  assert.strictEqual(tree.max(0, 4), -3);
  tree.add(4, 4, 100);
  assert.strictEqual(tree.max(0, 4), 94);

  const n = 16;
  let seed = 1234;
  const base = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    base.push((seed % 50) - 25);
  }
  const random = new RangeAddRangeMax(base);
  const brute = base.slice();
  for (const [kind, lo, hi, value] of lcgOps(200, 777, n)) {
    if (kind === 0) {
      random.add(lo, hi, value);
      for (let i = lo; i <= hi; i++) brute[i] += value;
    } else {
      assert.strictEqual(random.max(lo, hi), Math.max(...brute.slice(lo, hi + 1)), `${lo}..${hi}`);
    }
  }
  console.log("PASS");
}
