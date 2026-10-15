/** Task 111 - Kruskal's MST using union-find. (Tier: Beginner, CTCI Ch 4) */
"use strict";

class Dsu {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x) {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]];
      x = this.parent[x];
    }
    return x;
  }

  union(a, b) {
    let ra = this.find(a), rb = this.find(b);
    if (ra === rb) return false;
    if (this.rank[ra] < this.rank[rb]) [ra, rb] = [rb, ra];
    this.parent[rb] = ra;
    if (this.rank[ra] === this.rank[rb]) this.rank[ra]++;
    return true;
  }
}

function kruskal(n, edges) {
  const dsu = new Dsu(n);
  const sorted = [...edges].sort((a, b) => a[2] - b[2]);
  let total = 0;
  const chosen = [];
  for (const [u, v, w] of sorted) {
    if (dsu.union(u, v)) {
      chosen.push([u, v, w]);
      total += w;
      if (chosen.length === n - 1) break;
    }
  }
  return chosen.length === Math.max(0, n - 1) ? { total, chosen } : null;
}

module.exports = { Dsu, kruskal };

if (require.main === module) {
  const assert = require("node:assert");
  const edges = [
    [0, 1, 4], [0, 2, 3], [1, 2, 1],
    [1, 3, 2], [2, 3, 4], [3, 4, 2], [2, 4, 5],
  ];
  const result = kruskal(5, edges);
  assert.ok(result);
  assert.strictEqual(result.total, 8);
  assert.strictEqual(result.chosen.length, 4);
  assert.deepStrictEqual(kruskal(1, []), { total: 0, chosen: [] });
  assert.strictEqual(kruskal(3, [[0, 1, 1]]), null);
  console.log("PASS");
}
