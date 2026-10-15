/** Task 109 - Implement union-find / disjoint set (basic). (Tier: Beginner, CTCI Ch 4) */
"use strict";

class Dsu {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
    this.components = n;
  }

  find(x) {
    while (this.parent[x] !== x) x = this.parent[x]; // bare walk, no compression
    return x;
  }

  union(a, b) { // plain linking
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return false;
    this.parent[ra] = rb;
    this.components -= 1;
    return true;
  }

  unionByRank(a, b) {
    let ra = this.find(a);
    let rb = this.find(b);
    if (ra === rb) return false;
    if (this.rank[ra] < this.rank[rb]) [ra, rb] = [rb, ra];
    this.parent[rb] = ra;
    if (this.rank[ra] === this.rank[rb]) this.rank[ra] += 1;
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
  const d = new Dsu(6);
  assert.strictEqual(d.count(), 6);
  assert.strictEqual(d.find(3), 3);
  assert.strictEqual(d.connected(0, 0), true);
  assert.strictEqual(d.union(0, 1), true);
  assert.strictEqual(d.union(1, 2), true);
  assert.strictEqual(d.count(), 4);
  assert.strictEqual(d.connected(0, 2), true);
  assert.strictEqual(d.depth(0), 2);
  assert.strictEqual(d.union(0, 2), false);
  assert.strictEqual(d.count(), 4);
  assert.strictEqual(d.union(5, 5), false);

  assert.strictEqual(d.unionByRank(3, 4), true);
  assert.strictEqual(d.unionByRank(4, 5), true);
  assert.strictEqual(d.connected(3, 5), true);
  assert.strictEqual(d.connected(2, 3), false);
  assert.strictEqual(d.count(), 2);

  const r = new Dsu(8);
  for (let i = 0; i < 7; i++) assert.strictEqual(r.unionByRank(i, i + 1), true);
  assert.strictEqual(r.count(), 1);
  let maxDepth = 0;
  for (let i = 0; i < 8; i++) maxDepth = Math.max(maxDepth, r.depth(i));
  assert.ok(maxDepth <= 1);

  const one = new Dsu(1);
  assert.strictEqual(one.count(), 1);
  assert.strictEqual(one.find(0), 0);
  assert.strictEqual(one.depth(0), 0);
  console.log("PASS");
}
