/** Task 108 - Detect cycle in undirected graph (union-find). (Tier: Beginner, CTCI Ch 4) */
"use strict";

class Dsu {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x) {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]]; // path halving
      x = this.parent[x];
    }
    return x;
  }

  union(a, b) {
    let ra = this.find(a);
    let rb = this.find(b);
    if (ra === rb) return false;
    if (this.rank[ra] < this.rank[rb]) [ra, rb] = [rb, ra];
    this.parent[rb] = ra;
    if (this.rank[ra] === this.rank[rb]) this.rank[ra] += 1;
    return true;
  }
}

function hasCycleUndirected(n, edges) {
  const dsu = new Dsu(n);
  for (const [u, v] of edges) {
    if (!(u >= 0 && u < n && v >= 0 && v < n)) {
      throw new Error("edge endpoint out of range");
    }
    if (!dsu.union(u, v)) return true;
  }
  return false;
}

module.exports = { hasCycleUndirected, Dsu };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(hasCycleUndirected(5, [[0, 1], [0, 2], [1, 3], [1, 4]]), false);
  assert.strictEqual(hasCycleUndirected(3, [[0, 1], [1, 2], [2, 0]]), true);
  assert.strictEqual(hasCycleUndirected(6, [[0, 1], [2, 3], [3, 4], [4, 5]]), false);
  assert.strictEqual(hasCycleUndirected(6, [[0, 1], [2, 3], [3, 4], [4, 5], [5, 2]]), true);
  assert.strictEqual(hasCycleUndirected(1, []), false);
  assert.strictEqual(hasCycleUndirected(2, [[0, 0]]), true);
  assert.strictEqual(hasCycleUndirected(2, [[0, 1], [1, 0]]), true);
  assert.throws(() => hasCycleUndirected(2, [[0, 5]]), /out of range/);
  console.log("PASS");
}
