/** Task 444 - Implement 2D segment tree for range sum rectangle queries. (Tier: Advanced, CTCI Ch 4) */
"use strict";

class SegmentTree2D {
  constructor(grid) {
    this.rows = grid.length;
    this.cols = this.rows > 0 ? grid[0].length : 0;
    this.grid = grid.map((row) => row.slice());
    this.tree = Array.from({ length: 4 * Math.max(1, this.rows) }, () => new Array(4 * Math.max(1, this.cols)).fill(0));
    if (this.rows > 0) this._build(1, 0, this.rows - 1, grid);
  }

  _innerBuild(arr, node, lo, hi, row) {
    if (lo === hi) {
      arr[node] = row[lo];
      return;
    }
    const mid = (lo + hi) >> 1;
    this._innerBuild(arr, 2 * node, lo, mid, row);
    this._innerBuild(arr, 2 * node + 1, mid + 1, hi, row);
    arr[node] = arr[2 * node] + arr[2 * node + 1];
  }

  _build(node, lo, hi, grid) {
    if (lo === hi) {
      this._innerBuild(this.tree[node], 1, 0, this.cols - 1, grid[lo]);
      return;
    }
    const mid = (lo + hi) >> 1;
    this._build(2 * node, lo, mid, grid);
    this._build(2 * node + 1, mid + 1, hi, grid);
    const current = this.tree[node];
    const left = this.tree[2 * node];
    const right = this.tree[2 * node + 1];
    for (let k = 1; k < current.length; k++) current[k] = left[k] + right[k];
  }

  setValue(row, col, value) {
    const delta = value - this.grid[row][col];
    this.grid[row][col] = value;
    this._outerUpdate(1, 0, this.rows - 1, row, col, delta);
  }

  _outerUpdate(node, lo, hi, row, col, delta) {
    this._innerUpdate(this.tree[node], 1, 0, this.cols - 1, col, delta);
    if (lo === hi) return;
    const mid = (lo + hi) >> 1;
    if (row <= mid) this._outerUpdate(2 * node, lo, mid, row, col, delta);
    else this._outerUpdate(2 * node + 1, mid + 1, hi, row, col, delta);
  }

  _innerUpdate(arr, node, lo, hi, pos, delta) {
    arr[node] += delta;
    if (lo === hi) return;
    const mid = (lo + hi) >> 1;
    if (pos <= mid) this._innerUpdate(arr, 2 * node, lo, mid, pos, delta);
    else this._innerUpdate(arr, 2 * node + 1, mid + 1, hi, pos, delta);
  }

  _innerQuery(arr, node, lo, hi, left, right) {
    if (left <= lo && hi <= right) return arr[node];
    const mid = (lo + hi) >> 1;
    let total = 0;
    if (left <= mid) total += this._innerQuery(arr, 2 * node, lo, mid, left, right);
    if (right > mid) total += this._innerQuery(arr, 2 * node + 1, mid + 1, hi, left, right);
    return total;
  }

  rectSum(r1, c1, r2, c2) {
    return this._rect(1, 0, this.rows - 1, r1, c1, r2, c2);
  }

  _rect(node, lo, hi, r1, c1, r2, c2) {
    if (r2 < lo || hi < r1) return 0;
    if (r1 <= lo && hi <= r2) return this._innerQuery(this.tree[node], 1, 0, this.cols - 1, c1, c2);
    const mid = (lo + hi) >> 1;
    return this._rect(2 * node, lo, mid, r1, c1, r2, c2)
      + this._rect(2 * node + 1, mid + 1, hi, r1, c1, r2, c2);
  }
}

function lcgGridOps(count, seed, n) {
  const ops = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    const r1 = seed % n;
    seed = (seed * 75 + 74) % 65537;
    const c1 = seed % n;
    seed = (seed * 75 + 74) % 65537;
    const r2 = r1 + (seed % (n - r1));
    seed = (seed * 75 + 74) % 65537;
    const c2 = c1 + (seed % (n - c1));
    seed = (seed * 75 + 74) % 65537;
    const kind = seed % 2;
    seed = (seed * 75 + 74) % 65537;
    const value = seed % 41 - 20;
    ops.push([kind, r1, c1, r2, c2, value]);
  }
  return ops;
}

function bruteRect(grid, r1, c1, r2, c2) {
  let total = 0;
  for (let r = r1; r <= r2; r++) {
    for (let c = c1; c <= c2; c++) total += grid[r][c];
  }
  return total;
}

module.exports = { SegmentTree2D, lcgGridOps, bruteRect };

if (require.main === module) {
  const assert = require("node:assert");

  let tree = new SegmentTree2D([[1, 2, 3], [4, 5, 6]]);
  assert.strictEqual(tree.rectSum(0, 0, 0, 1), 3);
  assert.strictEqual(tree.rectSum(0, 0, 1, 2), 21);
  assert.strictEqual(tree.rectSum(1, 0, 1, 1), 9);
  tree.setValue(1, 1, 50);
  assert.strictEqual(tree.rectSum(1, 1, 1, 1), 50);
  assert.strictEqual(tree.rectSum(0, 0, 1, 2), 66);
  assert.strictEqual(tree.rectSum(1, 0, 1, 1), 54);
  tree.setValue(0, 2, 1);
  assert.strictEqual(tree.rectSum(0, 2, 0, 2), 1);
  assert.strictEqual(tree.rectSum(0, 0, 1, 2), 64);

  const n = 16;
  const ops = lcgGridOps(300, 444, n);
  let seed = 99;
  const grid = [];
  for (let r = 0; r < n; r++) {
    const row = [];
    for (let c = 0; c < n; c++) {
      seed = (seed * 75 + 74) % 65537;
      row.push(seed % 50 - 25);
    }
    grid.push(row);
  }
  tree = new SegmentTree2D(grid);
  const brute = grid.map((row) => row.slice());
  for (const [kind, r1, c1, r2, c2, value] of ops) {
    if (kind === 0) {
      tree.setValue(r1, c1, value);
      brute[r1][c1] = value;
    } else {
      assert.strictEqual(tree.rectSum(r1, c1, r2, c2), bruteRect(brute, r1, c1, r2, c2));
    }
  }
  console.log("PASS");
}
