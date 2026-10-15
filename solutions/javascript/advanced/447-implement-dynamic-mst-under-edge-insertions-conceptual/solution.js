/** Task 447 - Implement dynamic MST under edge insertions (conceptual). (Tier: Advanced, CTCI Ch 4) */
"use strict";

class DynamicMST {
  constructor() {
    this.edges = [];
    this.treeEdges = [];
    this.swaps = 0;
  }

  addEdge(u, v, w) {
    this.edges.push([u, v, w]);
    const heaviest = this.heaviestOnPath(u, v);
    if (heaviest === null) {
      this.treeEdges.push([u, v, w]);
    } else if (heaviest[2] > w) {
      this.treeEdges.splice(this.treeEdges.indexOf(heaviest), 1);
      this.treeEdges.push([u, v, w]);
      this.swaps++;
    }
  }

  heaviestOnPath(u, v) {
    if (u === v) return null;
    const adjacency = new Map();
    for (let index = 0; index < this.treeEdges.length; index++) {
      const [a, b] = this.treeEdges[index];
      if (!adjacency.has(a)) adjacency.set(a, []);
      if (!adjacency.has(b)) adjacency.set(b, []);
      adjacency.get(a).push([b, index]);
      adjacency.get(b).push([a, index]);
    }
    if (!adjacency.has(u)) return null;
    const parent = new Map([[u, [-1, -1]]]);
    const stack = [u];
    while (stack.length > 0) {
      const x = stack.pop();
      for (const [y, edgeIndex] of adjacency.get(x)) {
        if (!parent.has(y)) {
          parent.set(y, [x, edgeIndex]);
          stack.push(y);
        }
      }
    }
    if (!parent.has(v)) return null;
    let heaviest = null;
    let node = v;
    while (parent.get(node)[0] !== -1) {
      const [previous, edgeIndex] = parent.get(node);
      const edge = this.treeEdges[edgeIndex];
      if (heaviest === null || edge[2] > heaviest[2]) heaviest = edge;
      node = previous;
    }
    return heaviest;
  }

  mstWeight() {
    return this.treeEdges.reduce((total, edge) => total + edge[2], 0);
  }
}

function kruskalWeight(n, edges) {
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(x) {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  let total = 0;
  for (const [u, v, w] of edges.slice().sort((a, b) => a[2] - b[2])) {
    const ru = find(u);
    const rv = find(v);
    if (ru !== rv) {
      parent[ru] = rv;
      total += w;
    }
  }
  return total;
}

function lcgEdgeStream(count, seed, n) {
  const edges = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    const u = seed % n;
    seed = (seed * 75 + 74) % 65537;
    let v = seed % n;
    if (u === v) v = (v + 1) % n;
    seed = (seed * 75 + 74) % 65537;
    const w = 1 + (seed % 30);
    edges.push([u, v, w]);
  }
  return edges;
}

module.exports = { DynamicMST, kruskalWeight, lcgEdgeStream };

if (require.main === module) {
  const assert = require("node:assert");

  const dynamic = new DynamicMST();
  dynamic.addEdge(0, 1, 5);
  assert.strictEqual(dynamic.mstWeight(), 5);
  dynamic.addEdge(1, 2, 3);
  assert.strictEqual(dynamic.mstWeight(), 8);
  dynamic.addEdge(2, 3, 7);
  assert.strictEqual(dynamic.mstWeight(), 15);
  dynamic.addEdge(0, 3, 1);
  assert.strictEqual(dynamic.mstWeight(), 9);
  assert.strictEqual(dynamic.swaps, 1);
  dynamic.addEdge(0, 2, 9);
  assert.strictEqual(dynamic.mstWeight(), 9);

  let swapsSeen = 0;
  for (let trial = 0; trial < 25; trial++) {
    const n = 4 + (trial % 5);
    const stream = lcgEdgeStream(2 * n, 900 + trial * 37, n);
    const runner = new DynamicMST();
    const inserted = [];
    for (const edge of stream) {
      runner.addEdge(edge[0], edge[1], edge[2]);
      inserted.push(edge);
      assert.strictEqual(runner.mstWeight(), kruskalWeight(n, inserted), `trial ${trial}`);
    }
    swapsSeen += runner.swaps;
  }
  assert.ok(swapsSeen > 0);

  const n = 6;
  const stream = lcgEdgeStream(3 * n, 321, n);
  const runner = new DynamicMST();
  for (const edge of stream) runner.addEdge(edge[0], edge[1], edge[2]);
  assert.strictEqual(runner.mstWeight(), kruskalWeight(n, stream));
  console.log("PASS");
}
