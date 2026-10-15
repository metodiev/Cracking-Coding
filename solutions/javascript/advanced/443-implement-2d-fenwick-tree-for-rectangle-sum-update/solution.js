/** Task 443 - Implement 2D Fenwick tree for rectangle sum/update. (Tier: Advanced, CTCI Ch 4) */
"use strict";

class Fenwick2D {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.tree = Array.from({ length: rows + 1 }, () => new Array(cols + 1).fill(0));
  }

  add(row, col, delta) {
    for (let i = row + 1; i <= this.rows; i += i & -i) {
      for (let j = col + 1; j <= this.cols; j += j & -j) {
        this.tree[i][j] += delta;
      }
    }
  }

  prefix(row, col) {
    let total = 0;
    for (let i = row + 1; i > 0; i -= i & -i) {
      for (let j = col + 1; j > 0; j -= j & -j) {
        total += this.tree[i][j];
      }
    }
    return total;
  }

  rectSum(r1, c1, r2, c2) {
    return this.prefix(r2, c2) - this.prefix(r1 - 1, c2) - this.prefix(r2, c1 - 1) + this.prefix(r1 - 1, c1 - 1);
  }
}

class DiffFenwick2D {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.inner = new Fenwick2D(rows, cols);
  }

  rectAdd(r1, c1, r2, c2, delta) {
    this.inner.add(r1, c1, delta);
    if (r2 + 1 < this.rows) this.inner.add(r2 + 1, c1, -delta);
    if (c2 + 1 < this.cols) this.inner.add(r1, c2 + 1, -delta);
    if (r2 + 1 < this.rows && c2 + 1 < this.cols) this.inner.add(r2 + 1, c2 + 1, delta);
  }

  point(row, col) {
    return this.inner.prefix(row, col);
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
    const value = seed % 21 - 10;
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

module.exports = { Fenwick2D, DiffFenwick2D, lcgGridOps, bruteRect };

if (require.main === module) {
  const assert = require("node:assert");

  let bit = new Fenwick2D(4, 4);
  bit.add(0, 0, 1);
  bit.add(1, 2, 5);
  bit.add(3, 3, 2);
  assert.strictEqual(bit.rectSum(0, 0, 3, 3), 8);
  assert.strictEqual(bit.rectSum(1, 2, 1, 2), 5);
  assert.strictEqual(bit.rectSum(0, 1, 2, 1), 0);
  assert.strictEqual(bit.rectSum(0, 0, 0, 0), 1);
  assert.strictEqual(bit.rectSum(1, 2, 3, 3), 7);

  let diff = new DiffFenwick2D(4, 4);
  diff.rectAdd(0, 0, 1, 1, 3);
  assert.strictEqual(diff.point(0, 0), 3);
  assert.strictEqual(diff.point(1, 1), 3);
  assert.strictEqual(diff.point(1, 2), 0);
  assert.strictEqual(diff.point(2, 2), 0);
  diff.rectAdd(1, 1, 2, 2, 10);
  assert.strictEqual(diff.point(1, 1), 13);
  assert.strictEqual(diff.point(2, 2), 10);
  assert.strictEqual(diff.point(3, 3), 0);
  assert.strictEqual(diff.point(0, 0), 3);

  const n = 16;
  const ops = lcgGridOps(300, 443, n);

  bit = new Fenwick2D(n, n);
  let grid = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const [kind, r1, c1, r2, c2, value] of ops) {
    if (kind === 0) {
      bit.add(r1, c1, value);
      grid[r1][c1] += value;
    } else {
      assert.strictEqual(bit.rectSum(r1, c1, r2, c2), bruteRect(grid, r1, c1, r2, c2));
    }
  }

  diff = new DiffFenwick2D(n, n);
  grid = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const [kind, r1, c1, r2, c2, value] of ops) {
    if (kind === 0) {
      diff.rectAdd(r1, c1, r2, c2, value);
      for (let r = r1; r <= r2; r++) {
        for (let c = c1; c <= c2; c++) grid[r][c] += value;
      }
    } else {
      assert.strictEqual(diff.point(r1, c1), grid[r1][c1]);
    }
  }
  console.log("PASS");
}
