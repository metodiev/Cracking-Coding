/** Task 271 - Minimum height trees (tree centers). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function minHeightTrees(n, edges) {
  if (n === 1) return [0];
  const adj = Array.from({ length: n }, () => []);
  const degree = new Array(n).fill(0);
  for (const [a, b] of edges) {
    adj[a].push(b);
    adj[b].push(a);
    degree[a]++;
    degree[b]++;
  }

  let leaves = [];
  for (let i = 0; i < n; i++) if (degree[i] === 1) leaves.push(i);
  let remaining = n;
  while (remaining > 2) {
    remaining -= leaves.length;
    const next = [];
    for (const leaf of leaves) {
      for (const neighbour of adj[leaf]) {
        if (--degree[neighbour] === 1) next.push(neighbour);
      }
    }
    leaves = next;
  }
  return leaves.sort((a, b) => a - b);
}

function bruteForceCenters(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adj[a].push(b);
    adj[b].push(a);
  }
  const height = (root) => {
    const seen = new Set([root]);
    let frontier = [root];
    let levels = -1;
    while (frontier.length) {
      levels++;
      const next = [];
      for (const u of frontier) {
        for (const v of adj[u]) if (!seen.has(v)) { seen.add(v); next.push(v); }
      }
      frontier = next;
    }
    return levels;
  };
  const heights = Array.from({ length: n }, (_, r) => height(r));
  const best = Math.min(...heights);
  return { centers: heights.map((h, i) => (h === best ? i : -1)).filter((i) => i >= 0), best };
}

module.exports = { minHeightTrees, bruteForceCenters };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(minHeightTrees(6, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]), [2, 3]);
  assert.deepStrictEqual(minHeightTrees(4, [[0, 1], [0, 2], [0, 3]]), [0]);
  assert.deepStrictEqual(minHeightTrees(1, []), [0]);
  assert.deepStrictEqual(minHeightTrees(2, [[0, 1]]), [0, 1]);

  const sample = [[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [5, 6]];
  const oracle = bruteForceCenters(7, sample);
  assert.deepStrictEqual(minHeightTrees(7, sample), oracle.centers);
  assert.strictEqual(oracle.best, 3);
  assert.ok(minHeightTrees(7, sample).length <= 2);
  console.log("PASS");
}
