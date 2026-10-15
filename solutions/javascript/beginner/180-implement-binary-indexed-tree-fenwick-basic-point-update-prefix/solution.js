/** Task 180 - Implement binary indexed tree (Fenwick) basic. (Tier: Beginner, CTCI Ch 4) */
"use strict";

class Fenwick {
  constructor(values) {
    this.n = values.length;
    this.tree = new Array(this.n + 1).fill(0);
    values.forEach((v, i) => this.updateIndex(i + 1, v));
  }

  updateIndex(i, delta) {
    while (i <= this.n) {
      this.tree[i] += delta;
      i += i & -i;
    }
  }

  update(index, delta) {
    if (index < 0 || index >= this.n) throw new RangeError("index out of range");
    this.updateIndex(index + 1, delta);
  }

  prefixSum(index) {
    let i = index + 1;
    if (i < 0 || i > this.n) throw new RangeError("index out of range");
    let total = 0;
    while (i > 0) {
      total += this.tree[i];
      i -= i & -i;
    }
    return total;
  }

  rangeSum(lo, hi) {
    if (lo > hi) return 0;
    return this.prefixSum(hi) - this.prefixSum(lo - 1);
  }
}

module.exports = { Fenwick };

if (require.main === module) {
  const assert = require("node:assert");
  const values = [1, 3, 5, 7, 9, 11];
  const fw = new Fenwick(values);
  assert.strictEqual(fw.prefixSum(0), 1);
  assert.strictEqual(fw.prefixSum(5), 36);
  assert.strictEqual(fw.rangeSum(1, 3), 15);
  assert.strictEqual(fw.rangeSum(3, 2), 0);

  fw.update(2, 5);
  values[2] += 5;
  assert.strictEqual(fw.prefixSum(5), 41);
  fw.update(0, -1);
  values[0] -= 1;
  assert.strictEqual(fw.prefixSum(0), 0);
  console.log("PASS");
}
