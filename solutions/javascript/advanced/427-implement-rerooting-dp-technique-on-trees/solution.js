/** Task 427 - Implement rerooting DP technique on trees. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function sumOfDistancesFromEachNode(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const parent = new Array(n).fill(-1);
  parent[0] = 0;
  const order = [0];
  for (let i = 0; i < order.length; i++) {
    const u = order[i];
    for (const v of adjacency[u]) {
      if (v !== parent[u]) {
        parent[v] = u;
        order.push(v);
      }
    }
  }
  const size = new Array(n).fill(1);
  const down = new Array(n).fill(0);        // distances from u into its subtree
  for (let index = order.length - 1; index >= 0; index--) {
    const u = order[index];
    for (const v of adjacency[u]) {
      if (v !== parent[u]) {
        size[u] += size[v];
        down[u] += down[v] + size[v];
      }
    }
  }
  const result = new Array(n).fill(0);
  result[0] = down[0];
  for (const u of order) {
    for (const v of adjacency[u]) {
      if (v !== parent[u]) result[v] = result[u] + n - 2 * size[v];
    }
  }
  return result;
}

function bruteForce(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const totals = [];
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
    totals.push(distance.reduce((a, b) => a + b, 0));
  }
  return totals;
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

module.exports = { sumOfDistancesFromEachNode, bruteForce, lcgTrees };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(sumOfDistancesFromEachNode(1, []), [0]);
  assert.deepStrictEqual(sumOfDistancesFromEachNode(2, [[0, 1]]), [1, 1]);
  assert.deepStrictEqual(sumOfDistancesFromEachNode(3, [[0, 1], [1, 2]]), [3, 2, 3]);
  assert.deepStrictEqual(sumOfDistancesFromEachNode(4, [[0, 1], [1, 2], [2, 3]]), [6, 4, 4, 6]);
  assert.deepStrictEqual(sumOfDistancesFromEachNode(4, [[0, 1], [0, 2], [0, 3]]), [3, 5, 5, 5]);
  assert.deepStrictEqual(sumOfDistancesFromEachNode(4, [[0, 1], [0, 2], [1, 3]]), [4, 4, 6, 6]);

  for (const [n, edges] of lcgTrees(30, 427)) {
    const got = sumOfDistancesFromEachNode(n, edges);
    const expected = bruteForce(n, edges);
    assert.deepStrictEqual(got, expected, `n=${n}`);
  }

  console.log("PASS");
}
