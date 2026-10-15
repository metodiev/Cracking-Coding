/** Task 177 - Implement basic sparse matrix storage (coordinate list). (Tier: Beginner, CTCI Ch 7) */
"use strict";

class SparseMatrix {
  constructor(rows, cols) {
    if (rows < 0 || cols < 0) throw new Error("dimensions must be non-negative");
    this.rows = rows;
    this.cols = cols;
    this.data = new Map();          // "r,c" -> value (non-zero only)
  }

  key(row, col) { return `${row},${col}`; }

  check(row, col) {
    if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) {
      throw new RangeError(`index out of range: (${row}, ${col})`);
    }
  }

  set(row, col, value) {
    this.check(row, col);
    if (value === 0) this.data.delete(this.key(row, col));
    else this.data.set(this.key(row, col), value);
  }

  get(row, col) {
    this.check(row, col);
    return this.data.get(this.key(row, col)) ?? 0;
  }

  get nnz() { return this.data.size; }

  toDense() {
    const dense = Array.from({ length: this.rows }, () => new Array(this.cols).fill(0));
    for (const [k, v] of this.data) {
      const [r, c] = k.split(",").map(Number);
      dense[r][c] = v;
    }
    return dense;
  }
}

module.exports = { SparseMatrix };

if (require.main === module) {
  const assert = require("node:assert");
  const m = new SparseMatrix(3, 4);
  m.set(0, 0, 5);
  m.set(1, 2, 7);
  m.set(2, 3, -1);
  assert.strictEqual(m.nnz, 3);
  assert.strictEqual(m.get(0, 0), 5);
  assert.strictEqual(m.get(0, 1), 0);
  assert.deepStrictEqual(m.toDense(), [[5, 0, 0, 0], [0, 0, 7, 0], [0, 0, 0, -1]]);
  m.set(1, 2, 0);
  assert.strictEqual(m.nnz, 2);
  assert.strictEqual(m.get(1, 2), 0);
  assert.throws(() => m.get(5, 0), RangeError);
  console.log("PASS");
}
