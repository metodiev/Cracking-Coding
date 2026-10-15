/** Task 439 - Implement compressed Fenwick / sparse Fenwick idea. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A Fenwick tree that is touched at only a few coordinates does not need a dense array: the sparse
 * version keeps a map of just the cells an update really visits (i, i + lowbit(i), ...) and treats
 * every untouched cell as 0 while walking a prefix query (i, i - lowbit(i), ...). Documented setup:
 * SIZE = 2^20 = 1048576, coordinates 1..SIZE, updates propagate while i <= SIZE. A dense Fenwick of
 * the same size is the oracle: LCG updates are applied to both and every prefix query must agree,
 * with a brute-force sweep over the recorded updates as a third opinion. The touched-cell count is
 * asserted to stay within 21 * (#updates) - at most log2(SIZE) = 20 cells are visited per update -
 * and far below SIZE, which is the evidence that the representation really is compressed.
 */
"use strict";

const SIZE = 1 << 20;

class SparseFenwick {
  constructor(size) {
    this.size = size;
    this.tree = new Map();
  }

  add(index, delta) {
    let i = index;
    while (i <= this.size) {
      this.tree.set(i, (this.tree.get(i) || 0) + delta);
      i += i & -i;
    }
  }

  prefix(index) {
    let total = 0;
    let i = index;
    while (i > 0) {
      total += this.tree.get(i) || 0;
      i -= i & -i;
    }
    return total;
  }

  touched() {
    return this.tree.size;
  }
}

class DenseFenwick {
  constructor(size) {
    this.size = size;
    this.tree = new Array(size + 1).fill(0);
  }

  add(index, delta) {
    let i = index;
    while (i <= this.size) {
      this.tree[i] += delta;
      i += i & -i;
    }
  }

  prefix(index) {
    let total = 0;
    let i = index;
    while (i > 0) {
      total += this.tree[i];
      i -= i & -i;
    }
    return total;
  }
}

module.exports = { SIZE, SparseFenwick, DenseFenwick };

if (require.main === module) {
  const assert = require("node:assert");
  let sparse = new SparseFenwick(16);
  const dense = new DenseFenwick(16);
  sparse.add(1, 5);
  sparse.add(3, 2);
  sparse.add(16, 4);
  dense.add(1, 5);
  dense.add(3, 2);
  dense.add(16, 4);
  assert.strictEqual(sparse.prefix(0), 0);
  assert.strictEqual(sparse.prefix(1), 5);
  assert.strictEqual(sparse.prefix(2), 5);
  assert.strictEqual(sparse.prefix(3), 7);
  assert.strictEqual(sparse.prefix(15), 7);
  assert.strictEqual(sparse.prefix(16), 11);
  assert.strictEqual(sparse.touched(), 6, `touched ${sparse.touched()}`);
  for (let index = 0; index <= 16; index++) {
    assert.strictEqual(sparse.prefix(index), dense.prefix(index), `small agreement ${index}`);
  }

  sparse = new SparseFenwick(SIZE);
  const bigDense = new DenseFenwick(SIZE);
  assert.strictEqual(sparse.prefix(0), 0);
  assert.strictEqual(sparse.prefix(SIZE), 0);
  assert.strictEqual(sparse.touched(), 0);

  const recorded = [];
  let seed = 4390;
  let queries = 0;
  let updates = 0;
  for (let step = 0; step < 800; step++) {
    seed = (seed * 75 + 74) % 65537;
    seed = (seed * 75 + 74) % 65537;
    const coord = 1 + ((seed * 1024 + (seed >> 4)) % (SIZE - 1));
    seed = (seed * 75 + 74) % 65537;
    if (step % 2 === 0) {
      const delta = (seed % 21) - 10;
      sparse.add(coord, delta);
      bigDense.add(coord, delta);
      recorded.push([coord, delta]);
      updates++;
    } else {
      assert.strictEqual(sparse.prefix(coord), bigDense.prefix(coord), `random agreement ${step}`);
      queries++;
    }
  }
  assert.strictEqual(updates, 400);
  assert.strictEqual(queries, 400);

  for (const coord of [1, 12345, SIZE >> 1, SIZE - 1, SIZE]) {
    let expected = 0;
    for (const [position, delta] of recorded) {
      if (position <= coord) expected += delta;
    }
    assert.strictEqual(sparse.prefix(coord), expected, `brute agreement ${coord}`);
    assert.strictEqual(sparse.prefix(coord), bigDense.prefix(coord), `dense agreement ${coord}`);
  }
  assert.ok(sparse.touched() >= 1, "some cells touched");
  assert.ok(sparse.touched() <= 21 * updates, `compressed within bound ${sparse.touched()}`);
  assert.ok(sparse.touched() < SIZE / 100, `compressed far below dense ${sparse.touched()}`);
  assert.strictEqual(
    sparse.prefix(SIZE),
    recorded.reduce((acc, [, delta]) => acc + delta, 0),
    "total sum"
  );
  console.log("PASS");
}
