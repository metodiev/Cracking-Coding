/** Task 338 - Implement sparse table for RMQ (immutable). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

/**
 * Precompute row j holding the minimum of every window of length 2^j; a query combines the
 * two overlapping power-of-two windows that cover [left, right], giving O(n log n) build
 * and O(1) queries. Cross-checked for every (l, r) pair against a naive scan.
 */
class SparseTable {
  constructor(values) {
    if (values.length === 0) throw new Error("values must not be empty");
    const n = values.length;
    this.n = n;
    this.logs = new Array(n + 1).fill(0);
    for (let i = 2; i <= n; i++) this.logs[i] = this.logs[i >> 1] + 1;
    this.table = [values.slice()];
    for (let level = 1; (1 << level) <= n; level++) {
      const half = 1 << (level - 1);
      const previous = this.table[level - 1];
      const width = n - (1 << level) + 1;
      const row = [];
      for (let i = 0; i < width; i++) row.push(Math.min(previous[i], previous[i + half]));
      this.table.push(row);
    }
  }

  /** Minimum of values[left..right], both ends included. */
  query(left, right) {
    if (left < 0 || left > right || right >= this.n) throw new Error("invalid range");
    const level = this.logs[right - left + 1];
    const row = this.table[level];
    return Math.min(row[left], row[right - (1 << level) + 1]);
  }
}

function naiveMin(values, left, right) {
  return Math.min(...values.slice(left, right + 1));
}

module.exports = { SparseTable, naiveMin };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-checked array; every expectation below is verified by eye.
  const values = [4, 2, 7, 1, 3, 9, 0, 5];
  const table = new SparseTable(values);
  assert.strictEqual(table.query(0, 7), 0);
  assert.strictEqual(table.query(1, 3), 1);
  assert.strictEqual(table.query(0, 0), 4);
  assert.strictEqual(table.query(1, 1), 2);
  assert.strictEqual(table.query(2, 2), 7);
  assert.strictEqual(table.query(3, 7), 0);
  assert.strictEqual(table.query(4, 6), 0);
  assert.strictEqual(table.query(2, 5), 1);
  assert.strictEqual(table.query(5, 7), 0);
  assert.strictEqual(table.query(0, 2), 2);
  assert.strictEqual(table.query(6, 6), 0);
  assert.deepStrictEqual(table.logs, [0, 0, 1, 1, 2, 2, 2, 2, 3]);

  const dup = new SparseTable([5, 5, 5, 5]);
  assert.strictEqual(dup.query(0, 3), 5);
  assert.strictEqual(dup.query(1, 2), 5);
  const mixed = new SparseTable([3, -1, 4, -1, 5, -9, 2, 6, -9, 7]);
  assert.strictEqual(mixed.query(0, 9), -9);
  assert.strictEqual(mixed.query(5, 8), -9);
  assert.strictEqual(mixed.query(0, 3), -1);
  const single = new SparseTable([42]);
  assert.strictEqual(single.query(0, 0), 42);

  let seed = 8383;
  for (let trial = 0; trial < 8; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const size = 1 + (seed % 64);
    const randomValues = [];
    for (let i = 0; i < size; i++) {
      seed = (seed * 75 + 74) % 65537;
      randomValues.push(seed % 1000);
    }
    const randomTable = new SparseTable(randomValues);
    for (let left = 0; left < size; left++) {
      for (let right = left; right < size; right++) {
        assert.strictEqual(randomTable.query(left, right), naiveMin(randomValues, left, right),
          `random ${trial} [${left},${right}]`);
      }
    }
  }
  console.log("PASS");
}
