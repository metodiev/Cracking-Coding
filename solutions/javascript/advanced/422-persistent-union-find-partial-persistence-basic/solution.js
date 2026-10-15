/** Task 422 - Persistent union-find (partial persistence) basic. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

class PersistentUnionFind {
  constructor(n) {
    this.n = n;
    this.parentHistory = Array.from({ length: n }, () => []);  // (time, new parent)
    this.sizeHistory = Array.from({ length: n }, () => []);    // (time, new root size)
  }

  static valueAt(history, time) {
    let lo = 0;
    let hi = history.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (history[mid][0] <= time) lo = mid + 1;
      else hi = mid;
    }
    return lo === 0 ? null : history[lo - 1][1];
  }

  find(node, time) {
    for (;;) {
      const parent = PersistentUnionFind.valueAt(this.parentHistory[node], time);
      if (parent === null) return node;
      node = parent;
    }
  }

  size(node, time) {
    const recorded = PersistentUnionFind.valueAt(this.sizeHistory[node], time);
    return recorded === null ? 1 : recorded;
  }

  union(a, b, time) {
    let rootA = this.find(a, time - 1);
    let rootB = this.find(b, time - 1);
    if (rootA === rootB) return false;
    let sizeA = this.size(rootA, time - 1);
    let sizeB = this.size(rootB, time - 1);
    if (sizeA < sizeB) {
      [rootA, rootB] = [rootB, rootA];
      [sizeA, sizeB] = [sizeB, sizeA];
    }
    this.parentHistory[rootB].push([time, rootA]);
    this.sizeHistory[rootA].push([time, sizeA + sizeB]);
    return true;
  }

  componentCount(time) {
    const roots = new Set();
    for (let node = 0; node < this.n; node++) roots.add(this.find(node, time));
    return roots.size;
  }
}

function rootOf(parent, node) {
  while (parent[node] !== node) node = parent[node];
  return node;
}

function replay(n, operations, time) {
  const parent = Array.from({ length: n }, (_, i) => i);
  const size = new Array(n).fill(1);
  for (let step = 1; step <= time; step++) {
    const [a, b] = operations[step - 1];
    let rootA = rootOf(parent, a);
    let rootB = rootOf(parent, b);
    if (rootA !== rootB) {
      if (size[rootA] < size[rootB]) [rootA, rootB] = [rootB, rootA];
      parent[rootB] = rootA;
      size[rootA] += size[rootB];
    }
  }
  return parent;
}

function oracleRoots(n, operations, time) {
  const parent = replay(n, operations, time);
  return Array.from({ length: n }, (_, node) => rootOf(parent, node));
}

module.exports = { PersistentUnionFind, rootOf, replay, oracleRoots };

if (require.main === module) {
  const assert = require("node:assert");

  const operations = [[0, 1], [2, 3], [1, 3], [0, 2]];
  const uf = new PersistentUnionFind(4);
  const merges = operations.map(([a, b], index) => uf.union(a, b, index + 1));
  assert.deepStrictEqual(merges, [true, true, true, false]);

  assert.strictEqual(uf.componentCount(0), 4);
  assert.strictEqual(uf.componentCount(1), 3);
  assert.strictEqual(uf.componentCount(2), 2);
  assert.strictEqual(uf.componentCount(3), 1);
  assert.strictEqual(uf.componentCount(4), 1);

  assert.strictEqual(uf.find(1, 1), 0);
  assert.strictEqual(uf.find(0, 1), 0);
  assert.strictEqual(uf.find(3, 1), 3);
  assert.strictEqual(uf.find(3, 2), 2);
  assert.strictEqual(uf.find(3, 3), 0);
  assert.strictEqual(uf.find(3, 4), 0);
  assert.strictEqual(uf.find(2, 2), 2);
  assert.strictEqual(uf.find(1, 0), 1);
  assert.strictEqual(uf.find(2, 0), 2);

  for (let time = 0; time <= 4; time++) {
    const expected = oracleRoots(4, operations, time);
    assert.strictEqual(uf.componentCount(time), new Set(expected).size);
    for (let node = 0; node < 4; node++) {
      assert.strictEqual(uf.find(node, time), expected[node], `time=${time} node=${node}`);
    }
  }

  for (const seed of [422, 31415, 271828]) {
    const n = 2 + (seed % 9);
    const structure = new PersistentUnionFind(n);
    const ops = [];
    let local = seed;
    for (let i = 0; i < 40; i++) {
      local = (local * 75 + 74) % 65537;
      const a = local % n;
      local = (local * 75 + 74) % 65537;
      ops.push([a, local % n]);
    }
    let applied = 0;
    for (let time = 1; time <= 40; time++) {
      if (structure.union(ops[time - 1][0], ops[time - 1][1], time)) applied++;
      assert.strictEqual(structure.componentCount(time), n - applied, `seed=${seed} time=${time}`);
    }
    for (let step = 0; step < 60; step++) {
      local = (local * 75 + 74) % 65537;
      const time = local % 41;
      const expected = oracleRoots(n, ops, time);
      assert.strictEqual(structure.componentCount(time), new Set(expected).size);
      for (let node = 0; node < n; node++) {
        assert.strictEqual(structure.find(node, time), expected[node], `seed=${seed} time=${time}`);
      }
    }
  }

  console.log("PASS");
}
