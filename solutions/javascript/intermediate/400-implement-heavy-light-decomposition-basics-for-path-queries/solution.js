/** Task 400 - Implement heavy-light decomposition basics for path queries. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class Fenwick {
  constructor(size) {
    this.tree = new Array(size + 1).fill(0);
  }

  add(index, delta) {
    let i = index + 1;
    while (i < this.tree.length) {
      this.tree[i] += delta;
      i += i & -i;
    }
  }

  prefix(count) {
    let total = 0;
    let i = count;
    while (i > 0) {
      total += this.tree[i];
      i -= i & -i;
    }
    return total;
  }

  rangeSum(left, right) {
    return this.prefix(right + 1) - this.prefix(left);
  }
}

class HeavyLight {
  constructor(n, edges, values) {
    const adjacency = Array.from({ length: n }, () => []);
    for (const [a, b] of edges) {
      adjacency[a].push(b);
      adjacency[b].push(a);
    }
    this.parent = new Array(n).fill(-1);
    this.depth = new Array(n).fill(0);
    const order = [0];
    for (let i = 0; i < order.length; i++) {
      const u = order[i];
      for (const v of adjacency[u]) {
        if (v !== this.parent[u]) {
          this.parent[v] = u;
          this.depth[v] = this.depth[u] + 1;
          order.push(v);
        }
      }
    }
    this.size = new Array(n).fill(1);
    this.heavy = new Array(n).fill(-1);
    for (let i = order.length - 1; i >= 0; i--) {
      const u = order[i];
      let best = 0;
      for (const v of adjacency[u]) {
        if (v !== this.parent[u]) {
          this.size[u] += this.size[v];
          if (this.size[v] > best) {
            best = this.size[v];
            this.heavy[u] = v;
          }
        }
      }
    }
    this.head = new Array(n).fill(0);
    this.pos = new Array(n).fill(0);
    let position = 0;
    for (const u of order) {
      if (u === 0 || this.heavy[this.parent[u]] !== u) {       // starts a new chain
        let x = u;
        while (x !== -1) {
          this.head[x] = u;
          this.pos[x] = position++;
          x = this.heavy[x];
        }
      }
    }
    this.values = values.slice();
    this.fenwick = new Fenwick(n);
    for (let node = 0; node < n; node++) this.fenwick.add(this.pos[node], values[node]);
  }

  update(node, value) {
    const delta = value - this.values[node];
    this.values[node] = value;
    this.fenwick.add(this.pos[node], delta);
  }

  pathSum(u, v) {
    let total = 0;
    while (this.head[u] !== this.head[v]) {
      if (this.depth[this.head[u]] < this.depth[this.head[v]]) [u, v] = [v, u];
      total += this.fenwick.rangeSum(this.pos[this.head[u]], this.pos[u]);
      u = this.parent[this.head[u]];
    }
    if (this.depth[u] > this.depth[v]) [u, v] = [v, u];
    total += this.fenwick.rangeSum(this.pos[u], this.pos[v]);
    return total;
  }
}

function brutePathSum(n, edges, values, u, v) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const parent = new Array(n).fill(-1);
  parent[u] = u;
  const queue = [u];
  for (let i = 0; i < queue.length; i++) {
    const x = queue[i];
    if (x === v) break;
    for (const y of adjacency[x]) {
      if (parent[y] === -1) {
        parent[y] = x;
        queue.push(y);
      }
    }
  }
  let total = 0;
  let x = v;
  while (x !== u) {
    total += values[x];
    x = parent[x];
  }
  return total + values[u];
}

function lcgTrees(count, seed = 400, maxN = 50) {
  const trees = [];
  for (let t = 0; t < count; t++) {
    const n = 2 + (seed % (maxN - 1));
    const edges = [];
    const values = [];
    for (let node = 0; node < n; node++) {
      seed = (seed * 75 + 74) % 65537;
      values.push((seed % 20) - 5);
      if (node > 0) {
        seed = (seed * 75 + 74) % 65537;
        edges.push([seed % node, node]);
      }
    }
    trees.push([n, edges, values]);
  }
  return trees;
}

module.exports = { Fenwick, HeavyLight, brutePathSum, lcgTrees };

if (require.main === module) {
  const assert = require("node:assert");
  const hld = new HeavyLight(4, [[0, 1], [1, 2], [2, 3]], [1, 2, 3, 4]);
  assert.strictEqual(hld.pathSum(0, 3), 10);
  assert.strictEqual(hld.pathSum(0, 0), 1);
  assert.strictEqual(hld.pathSum(1, 2), 5);
  hld.update(2, 10);
  assert.strictEqual(hld.pathSum(0, 3), 17);
  assert.strictEqual(hld.pathSum(1, 2), 12);
  assert.strictEqual(hld.pathSum(2, 2), 10);

  const star = new HeavyLight(3, [[0, 1], [0, 2]], [5, 1, 1]);
  assert.strictEqual(star.pathSum(1, 2), 7);
  assert.strictEqual(star.pathSum(0, 0), 5);

  let trial = 0;
  for (const [n, edges, initial] of lcgTrees(25)) {
    const values = initial.slice();
    const tree = new HeavyLight(n, edges, values);
    let seed = n;
    for (let step = 0; step < 12; step++) {
      seed = (seed * 75 + 74) % 65537;
      const u = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const v = seed % n;
      assert.strictEqual(tree.pathSum(u, v), brutePathSum(n, edges, values, u, v), `trial ${trial}`);
      seed = (seed * 75 + 74) % 65537;
      const node = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const newValue = (seed % 40) - 20;
      values[node] = newValue;
      tree.update(node, newValue);
      assert.strictEqual(tree.pathSum(u, v), brutePathSum(n, edges, values, u, v), `trial ${trial} after update`);
    }
    trial++;
  }
  console.log("PASS");
}
