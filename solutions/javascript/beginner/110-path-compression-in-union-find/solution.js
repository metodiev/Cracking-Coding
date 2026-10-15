/** Task 110 - Path compression in union-find. (Tier: Beginner, CTCI Ch 4) */
"use strict";

class Dsu {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.components = n;
  }

  find(x) {
    let root = x;
    while (this.parent[root] !== root) root = this.parent[root];
    while (this.parent[x] !== root) {      // second pass flattens the path
      const next = this.parent[x];
      this.parent[x] = root;
      x = next;
    }
    return root;
  }

  union(a, b) { // plain linking on purpose
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return false;
    this.parent[ra] = rb;
    this.components -= 1;
    return true;
  }

  connected(a, b) {
    return this.find(a) === this.find(b);
  }

  count() {
    return this.components;
  }

  depth(x) {
    let d = 0;
    while (this.parent[x] !== x) {
      x = this.parent[x];
      d += 1;
    }
    return d;
  }
}

module.exports = { Dsu };

if (require.main === module) {
  const assert = require("node:assert");
  const d = new Dsu(9);
  assert.strictEqual(d.count(), 9);
  for (let i = 0; i < 7; i++) assert.strictEqual(d.union(i, i + 1), true);
  assert.strictEqual(d.count(), 2);
  assert.strictEqual(d.depth(0), 7);
  assert.strictEqual(d.find(0), 7);
  assert.strictEqual(d.depth(0), 1);
  for (let i = 1; i <= 6; i++) assert.strictEqual(d.depth(i), 1);
  assert.strictEqual(d.depth(7), 0);
  assert.strictEqual(d.connected(0, 6), true);
  assert.strictEqual(d.find(0), d.find(0));
  assert.strictEqual(d.depth(0), 1);
  assert.strictEqual(d.union(7, 8), true);
  assert.strictEqual(d.count(), 1);
  assert.strictEqual(d.find(0), 8);
  assert.ok(d.depth(0) <= 2);
  assert.strictEqual(d.find(0), 8);
  assert.strictEqual(d.depth(0), 1);
  let maxDepth = 0;
  for (let i = 0; i < 9; i++) maxDepth = Math.max(maxDepth, d.depth(i));
  assert.ok(maxDepth <= 2);

  const one = new Dsu(1);
  assert.strictEqual(one.count(), 1);
  assert.strictEqual(one.find(0), 0);
  assert.strictEqual(one.depth(0), 0);

  const flat = new Dsu(3);
  assert.strictEqual(flat.union(0, 1), true);
  assert.strictEqual(flat.union(1, 2), true);
  assert.strictEqual(flat.depth(0), 2);
  assert.strictEqual(flat.find(0), 2);
  assert.strictEqual(flat.depth(0), 1);
  assert.strictEqual(flat.depth(1), 1);
  console.log("PASS");
}
