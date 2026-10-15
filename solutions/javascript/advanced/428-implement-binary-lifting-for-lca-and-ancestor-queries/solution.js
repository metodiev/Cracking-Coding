/** Task 428 - Implement binary lifting for LCA and ancestor queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

class BinaryLifting {
  constructor(n, edges) {
    this.n = n;
    this.log = 1;
    while ((1 << this.log) <= n) this.log++;
    const adjacency = Array.from({ length: n }, () => []);
    for (const [a, b] of edges) {
      adjacency[a].push(b);
      adjacency[b].push(a);
    }
    this.depth = new Array(n).fill(0);
    this.parent = new Array(n).fill(0);
    const visited = new Array(n).fill(false);
    visited[0] = true;
    const order = [0];
    for (let i = 0; i < order.length; i++) {
      const u = order[i];
      for (const v of adjacency[u]) {
        if (!visited[v]) {
          visited[v] = true;
          this.parent[v] = u;
          this.depth[v] = this.depth[u] + 1;
          order.push(v);
        }
      }
    }
    this.up = [this.parent.slice()];
    for (let j = 1; j < this.log; j++) {
      const previous = this.up[j - 1];
      const current = new Array(n);
      for (let u = 0; u < n; u++) current[u] = previous[previous[u]];
      this.up.push(current);
    }
  }

  kthAncestor(node, k) {
    if (k > this.depth[node]) return -1;
    let bit = 0;
    while (k) {
      if (k & 1) node = this.up[bit][node];
      k >>= 1;
      bit++;
    }
    return node;
  }

  lca(a, b) {
    if (this.depth[a] < this.depth[b]) [a, b] = [b, a];
    a = this.kthAncestor(a, this.depth[a] - this.depth[b]);
    if (a === b) return a;
    for (let j = this.log - 1; j >= 0; j--) {
      if (this.up[j][a] !== this.up[j][b]) {
        a = this.up[j][a];
        b = this.up[j][b];
      }
    }
    return this.parent[a];
  }
}

function rootedTree(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const parent = new Array(n).fill(0);
  const depth = new Array(n).fill(0);
  const visited = new Array(n).fill(false);
  visited[0] = true;
  const order = [0];
  for (let i = 0; i < order.length; i++) {
    const u = order[i];
    for (const v of adjacency[u]) {
      if (!visited[v]) {
        visited[v] = true;
        parent[v] = u;
        depth[v] = depth[u] + 1;
        order.push(v);
      }
    }
  }
  return { parent, depth };
}

function naiveLca(parent, depth, a, b) {
  while (depth[a] > depth[b]) a = parent[a];
  while (depth[b] > depth[a]) b = parent[b];
  while (a !== b) {
    a = parent[a];
    b = parent[b];
  }
  return a;
}

function naiveKthAncestor(parent, depth, node, k) {
  if (k > depth[node]) return -1;
  for (let i = 0; i < k; i++) node = parent[node];
  return node;
}

function lcgTrees(count, seed, maxN = 30) {
  const trees = [];
  for (let t = 0; t < count; t++) {
    const n = 2 + (seed % (maxN - 1));
    const edges = [];
    for (let node = 1; node < n; node++) {
      seed = (seed * 75 + 74) % 65537;
      edges.push([seed % node, node]);
    }
    trees.push([n, edges]);
  }
  return trees;
}

module.exports = { BinaryLifting, rootedTree, naiveLca, naiveKthAncestor, lcgTrees };

if (require.main === module) {
  const assert = require("node:assert");

  const path = new BinaryLifting(4, [[0, 1], [1, 2], [2, 3]]);
  assert.strictEqual(path.lca(0, 3), 0);
  assert.strictEqual(path.lca(2, 3), 2);
  assert.strictEqual(path.lca(1, 2), 1);
  assert.strictEqual(path.lca(3, 3), 3);
  assert.strictEqual(path.lca(0, 0), 0);
  assert.strictEqual(path.kthAncestor(3, 0), 3);
  assert.strictEqual(path.kthAncestor(3, 1), 2);
  assert.strictEqual(path.kthAncestor(3, 2), 1);
  assert.strictEqual(path.kthAncestor(3, 3), 0);
  assert.strictEqual(path.kthAncestor(3, 4), -1);
  assert.strictEqual(path.kthAncestor(0, 1), -1);

  const star = new BinaryLifting(4, [[0, 1], [0, 2], [0, 3]]);
  assert.strictEqual(star.lca(1, 2), 0);
  assert.strictEqual(star.lca(1, 3), 0);
  assert.strictEqual(star.lca(0, 3), 0);
  assert.strictEqual(star.lca(2, 2), 2);
  assert.strictEqual(star.kthAncestor(2, 1), 0);
  assert.strictEqual(star.kthAncestor(2, 2), -1);

  const single = new BinaryLifting(1, []);
  assert.strictEqual(single.lca(0, 0), 0);
  assert.strictEqual(single.kthAncestor(0, 0), 0);
  assert.strictEqual(single.kthAncestor(0, 1), -1);

  for (const [n, edges] of lcgTrees(25, 428)) {
    const bl = new BinaryLifting(n, edges);
    const { parent, depth } = rootedTree(n, edges);
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        const expected = naiveLca(parent, depth, a, b);
        assert.strictEqual(bl.lca(a, b), expected, `lca ${a}/${b}`);
        assert.strictEqual(bl.lca(a, b), bl.lca(b, a), `symmetry ${a}/${b}`);
      }
    }
    for (let u = 0; u < n; u++) {
      for (let k = 0; k <= depth[u] + 2; k++) {
        const expected = naiveKthAncestor(parent, depth, u, k);
        assert.strictEqual(bl.kthAncestor(u, k), expected, `kth ${u}/${k}`);
      }
    }
  }

  console.log("PASS");
}
