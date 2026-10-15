/** Task 181 - Range update, point query with Fenwick tree. (Tier: Beginner, CTCI Ch 4) */
"use strict";

/**
 * Dual of task 180's Fenwick tree: the tree stores a *difference* array, so
 * rangeAdd costs two point updates and pointQuery is a prefix sum (all O(log n)).
 */
class RangeUpdateFenwick {
  constructor(size) {
    this.n = size;
    this.tree = new Array(size + 1).fill(0);
  }

  add(i, delta) {
    while (i <= this.n) {
      this.tree[i] += delta;
      i += i & -i;
    }
  }

  rangeAdd(lo, hi, value) {
    if (lo < 0 || hi >= this.n || lo > hi) throw new RangeError("range out of bounds");
    this.add(lo + 1, value);
    this.add(hi + 2, -value); // hi + 2 > n is a silent no-op suffix update
  }

  pointQuery(index) {
    if (index < 0 || index >= this.n) throw new RangeError("index out of range");
    let total = 0;
    for (let i = index + 1; i > 0; i -= i & -i) total += this.tree[i];
    return total;
  }
}

const snapshot = (fw, size) => Array.from({ length: size }, (_, i) => fw.pointQuery(i));

module.exports = { RangeUpdateFenwick };

if (require.main === module) {
  const assert = require("node:assert");

  const fw = new RangeUpdateFenwick(6);
  fw.rangeAdd(1, 3, 5);
  assert.deepStrictEqual(snapshot(fw, 6), [0, 5, 5, 5, 0, 0]);
  fw.rangeAdd(0, 5, -2);
  assert.deepStrictEqual(snapshot(fw, 6), [-2, 3, 3, 3, -2, -2]);
  fw.rangeAdd(2, 2, 10);
  assert.deepStrictEqual(snapshot(fw, 6), [-2, 3, 13, 3, -2, -2]);
  fw.rangeAdd(5, 5, 7);
  assert.strictEqual(fw.pointQuery(5), 5);
  fw.rangeAdd(0, 0, -3);
  assert.strictEqual(fw.pointQuery(0), -5);

  assert.throws(() => fw.rangeAdd(3, 2, 1), RangeError);
  assert.throws(() => fw.rangeAdd(0, 6, 1), RangeError);
  assert.throws(() => fw.pointQuery(-1), RangeError);
  assert.throws(() => new RangeUpdateFenwick(0).pointQuery(0), RangeError);

  // cross-check every position against a plain array after each random range add
  const size = 12;
  const values = new Array(size).fill(0);
  const fw2 = new RangeUpdateFenwick(size);
  let state = 11;
  const nextInt = (bound) => {
    state = (state * 48271) % 2147483647;
    return state % bound;
  };
  for (let t = 0; t < 80; t++) {
    const lo = nextInt(size);
    const hi = lo + nextInt(size - lo);
    const value = nextInt(19) - 9;
    for (let k = lo; k <= hi; k++) values[k] += value;
    fw2.rangeAdd(lo, hi, value);
    assert.deepStrictEqual(snapshot(fw2, size), values);
  }
  console.log("PASS");
}
