/** Task 425 - Implement centroid decomposition for k-distance pairs. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function countPairsAtDistance(n, edges, k) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const removed = new Array(n).fill(false);
  const size = new Array(n).fill(0);

  function sizes(u, parent) {
    size[u] = 1;
    for (const v of adjacency[u]) {
      if (v !== parent && !removed[v]) {
        sizes(v, u);
        size[u] += size[v];
      }
    }
  }

  function findCentroid(u, parent, component) {
    for (const v of adjacency[u]) {
      if (v !== parent && !removed[v] && size[v] > Math.floor(component / 2)) {
        return findCentroid(v, u, component);
      }
    }
    return u;
  }

  function collect(u, parent, distance, distances) {
    distances.push(distance);
    for (const v of adjacency[u]) {
      if (v !== parent && !removed[v]) collect(v, u, distance + 1, distances);
    }
  }

  function pairsWithSum(distances) {
    const seen = new Map();
    let pairs = 0;
    for (const d of distances) {
      pairs += seen.get(k - d) || 0;
      seen.set(d, (seen.get(d) || 0) + 1);
    }
    return pairs;
  }

  let total = 0;

  function decompose(start) {
    sizes(start, -1);
    const centroid = findCentroid(start, -1, size[start]);
    removed[centroid] = true;
    const allDistances = [0];                    // the centroid itself
    const branches = [];
    for (const v of adjacency[centroid]) {
      if (removed[v]) continue;
      const distances = [];
      collect(v, centroid, 1, distances);
      branches.push(distances);
      allDistances.push(...distances);
    }
    total += pairsWithSum(allDistances);
    for (const distances of branches) {          // remove pairs inside one branch
      total -= pairsWithSum(distances);
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
  let total = 0;
  for (let source = 0; source < n; source++) {
    const distance = new Array(n).fill(-1);
    distance[source] = 0;
    const queue = [source];
    for (let i = 0; i < queue.length; i++) {
      const u = queue[i];
      for (const v of adjacency[u]) {
        if (distance[v] === -1) {
          distance[v] = distance[u] + 1;
          queue.push(v);
        }
      }
    }
    for (let other = source + 1; other < n; other++) {
      if (distance[other] === k) total++;
    }
  }
  return total;
}

function lcgTrees(count, seed, maxN = 60) {
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

  assert.strictEqual(countPairsAtDistance(0, [], 0), 0);
  assert.strictEqual(countPairsAtDistance(1, [], 0), 0);
  assert.strictEqual(countPairsAtDistance(1, [], 1), 0);
  assert.strictEqual(countPairsAtDistance(2, [[0, 1]], 1), 1);
  assert.strictEqual(countPairsAtDistance(2, [[0, 1]], 0), 0);
  assert.strictEqual(countPairsAtDistance(2, [[0, 1]], 2), 0);

  const path3 = [[0, 1], [1, 2]];
  assert.strictEqual(countPairsAtDistance(3, path3, 0), 0);
  assert.strictEqual(countPairsAtDistance(3, path3, 1), 2);   // {0-1, 1-2}
  assert.strictEqual(countPairsAtDistance(3, path3, 2), 1);   // {0-2}
  assert.strictEqual(countPairsAtDistance(3, path3, 3), 0);

  const star = [[0, 1], [0, 2], [0, 3]];
  assert.strictEqual(countPairsAtDistance(4, star, 1), 3);    // centre to each leaf
  assert.strictEqual(countPairsAtDistance(4, star, 2), 3);    // C(3,2) leaf pairs
  assert.strictEqual(countPairsAtDistance(4, star, 3), 0);

  const path4 = [[0, 1], [1, 2], [2, 3]];
  assert.strictEqual(countPairsAtDistance(4, path4, 1), 3);
  assert.strictEqual(countPairsAtDistance(4, path4, 2), 2);   // {0-2, 1-3}
  assert.strictEqual(countPairsAtDistance(4, path4, 3), 1);   // {0-3}
  assert.strictEqual(countPairsAtDistance(4, path4, 4), 0);

  const path6 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]];
  const expected6 = [0, 5, 4, 3, 2, 1, 0];
  expected6.forEach((expected, k) => {
    assert.strictEqual(countPairsAtDistance(6, path6, k), expected, `path6 k=${k}`);
  });

  for (const [n, edges] of lcgTrees(30, 425)) {
    for (let k = 0; k < 13; k++) {
      const got = countPairsAtDistance(n, edges, k);
      const expected = bruteForce(n, edges, k);
      assert.strictEqual(got, expected, `n=${n} k=${k} got=${got} want=${expected}`);
    }
  }

  console.log("PASS");
}
