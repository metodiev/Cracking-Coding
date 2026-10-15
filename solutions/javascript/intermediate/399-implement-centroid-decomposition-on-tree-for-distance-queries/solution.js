/** Task 399 - Implement centroid decomposition for distance queries. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function countPairsAtDistance(n, edges, k) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const removed = new Array(n).fill(false);
  const size = new Array(n).fill(0);
  let total = 0;

  function sizes(u, parent) {
    size[u] = 1;
    for (const v of adjacency[u]) {
      if (v !== parent && !removed[v]) {
        sizes(v, u);
        size[u] += size[v];
      }
    }
  }

  function findCentroid(u, parent, componentSize) {
    for (const v of adjacency[u]) {
      if (v !== parent && !removed[v] && size[v] > Math.floor(componentSize / 2)) {
        return findCentroid(v, u, componentSize);
      }
    }
    return u;
  }

  function collect(u, parent, distance, out) {
    out.push(distance);
    for (const v of adjacency[u]) {
      if (v !== parent && !removed[v]) collect(v, u, distance + 1, out);
    }
  }

  function decompose(start) {
    sizes(start, -1);
    const centroid = findCentroid(start, -1, size[start]);
    removed[centroid] = true;
    const seen = new Map([[0, 1]]);            // the centroid itself is at distance 0
    for (const v of adjacency[centroid]) {
      if (removed[v]) continue;
      const subtree = [];
      collect(v, centroid, 1, subtree);
      for (const d of subtree) total += seen.get(k - d) || 0;
      for (const d of subtree) seen.set(d, (seen.get(d) || 0) + 1);
    }
    for (const v of adjacency[centroid]) {
      if (!removed[v]) decompose(v);
    }
  }

  if (n > 0) decompose(0);
  return total;
}

function bruteForce(n, edges, k) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  let count = 0;
  for (let source = 0; source < n; source++) {
    const distance = new Array(n).fill(-1);
    distance[source] = 0;
    const queue = [source];
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      for (const v of adjacency[u]) {
        if (distance[v] === -1) {
          distance[v] = distance[u] + 1;
          queue.push(v);
        }
      }
    }
    for (let target = source + 1; target < n; target++) {
      if (distance[target] === k) count++;
    }
  }
  return count;
}

function lcgTrees(count, seed = 600, maxN = 60) {
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

module.exports = { countPairsAtDistance, bruteForce, lcgTrees };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countPairsAtDistance(3, [[0, 1], [1, 2]], 1), 2);
  assert.strictEqual(countPairsAtDistance(3, [[0, 1], [1, 2]], 2), 1);
  assert.strictEqual(countPairsAtDistance(3, [[0, 1], [1, 2]], 3), 0);
  assert.strictEqual(countPairsAtDistance(3, [[0, 1], [0, 2]], 2), 1);
  assert.strictEqual(countPairsAtDistance(3, [[0, 1], [0, 2]], 1), 2);
  assert.strictEqual(countPairsAtDistance(1, [], 0), 0);
  assert.strictEqual(countPairsAtDistance(2, [[0, 1]], 1), 1);
  assert.strictEqual(countPairsAtDistance(2, [[0, 1]], 2), 0);

  const path6 = [];
  for (let i = 0; i < 5; i++) path6.push([i, i + 1]);
  for (let k = 0; k <= 6; k++) {
    assert.strictEqual(countPairsAtDistance(6, path6, k), bruteForce(6, path6, k), `path6 k=${k}`);
  }

  for (const [n, edges] of lcgTrees(30)) {
    for (let k = 0; k < 6; k++) {
      assert.strictEqual(countPairsAtDistance(n, edges, k), bruteForce(n, edges, k), `n=${n} k=${k}`);
    }
  }
  console.log("PASS");
}
