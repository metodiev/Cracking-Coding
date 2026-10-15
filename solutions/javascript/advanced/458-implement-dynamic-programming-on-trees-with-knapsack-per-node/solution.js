/** Task 458 - Implement dynamic programming on trees with knapsack per node. (Tier: Advanced, CTCI Ch 4) */
"use strict";

const NEG_INF = -1_000_000_000;

/** Best sum of <= k chosen vertices with the parent-before-child closure rule. */
function treeKnapsack(n, parent, weight, k) {
  if (k <= 0 || n === 0) return 0;
  const children = Array.from({ length: n }, () => []);
  for (let v = 1; v < n; v++) children[parent[v]].push(v);

  function dfs(u) {
    let subtreeSize = 1;
    let dp = new Array(k + 1).fill(NEG_INF);
    dp[1] = weight[u];
    for (const child of children[u]) {
      const childResult = dfs(child);
      const cdp = childResult.dp;
      const childSize = childResult.size;
      const newDp = dp.slice();
      for (let i = 1; i <= Math.min(k, subtreeSize); i++) {
        if (dp[i] === NEG_INF) continue;
        for (let j = 1; j <= Math.min(childSize, k - i); j++) {
          if (cdp[j] === NEG_INF) continue;
          newDp[i + j] = Math.max(newDp[i + j], dp[i] + cdp[j]);
        }
      }
      dp = newDp;
      subtreeSize += childSize;
    }
    return { dp, size: subtreeSize };
  }

  const result = dfs(0);
  let best = 0;
  for (let j = 1; j <= k; j++) best = Math.max(best, result.dp[j]);
  return best;
}

/** Exhaustive over all parent-closed subsets of size <= k (empty set allowed). */
function bruteTreeKnapsack(n, parent, weight, k) {
  let best = 0;
  for (let mask = 0; mask < (1 << n); mask++) {
    let size = 0;
    for (let v = 0; v < n; v++) size += (mask >> v) & 1;
    if (size > k) continue;
    let ok = true;
    for (let v = 1; v < n; v++) {
      if (((mask >> v) & 1) === 1 && ((mask >> parent[v]) & 1) === 0) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    let total = 0;
    for (let v = 0; v < n; v++) if (((mask >> v) & 1) === 1) total += weight[v];
    best = Math.max(best, total);
  }
  return best;
}

function lcgTrees(count, seed) {
  const trees = [];
  for (let g = 0; g < count; g++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 6 + (seed % 4);
    const parent = new Array(n).fill(-1);
    const weight = new Array(n).fill(0);
    seed = (seed * 75 + 74) % 65537;
    weight[0] = (seed % 31) - 15;
    for (let v = 1; v < n; v++) {
      seed = (seed * 75 + 74) % 65537;
      parent[v] = seed % v;
      seed = (seed * 75 + 74) % 65537;
      weight[v] = (seed % 31) - 15;
    }
    trees.push({ n, parent, weight });
  }
  return trees;
}

module.exports = { treeKnapsack, bruteTreeKnapsack, lcgTrees };

if (require.main === module) {
  const assert = require("node:assert");
  const p1 = [-1, 0, 0, 1];
  const w1 = [5, 3, 4, -10];
  assert.strictEqual(treeKnapsack(4, p1, w1, 1), 5);
  assert.strictEqual(treeKnapsack(4, p1, w1, 2), 9);
  assert.strictEqual(treeKnapsack(4, p1, w1, 3), 12);
  assert.strictEqual(treeKnapsack(4, p1, w1, 4), 12);
  assert.strictEqual(treeKnapsack(4, p1, w1, 0), 0);
  for (let k = 0; k <= 4; k++) {
    assert.strictEqual(treeKnapsack(4, p1, w1, k), bruteTreeKnapsack(4, p1, w1, k), `tree1 k=${k}`);
  }

  const p2 = [-1, 0, 0, 1, 1];
  const w2 = [10, 8, 5, 7, 6];
  assert.strictEqual(treeKnapsack(5, p2, w2, 3), 25);
  for (let k = 0; k <= 5; k++) {
    assert.strictEqual(treeKnapsack(5, p2, w2, k), bruteTreeKnapsack(5, p2, w2, k), `tree2 k=${k}`);
  }

  const p3 = [-1, 0];
  const w3 = [-2, -3];
  assert.strictEqual(treeKnapsack(2, p3, w3, 2), 0);
  assert.strictEqual(treeKnapsack(2, p3, w3, 1), 0);

  const p4 = [-1, 0, 1];
  const w4 = [1, -5, 10];
  assert.strictEqual(treeKnapsack(3, p4, w4, 1), 1);
  assert.strictEqual(treeKnapsack(3, p4, w4, 2), 1);
  assert.strictEqual(treeKnapsack(3, p4, w4, 3), 6);
  for (let k = 0; k <= 3; k++) {
    assert.strictEqual(treeKnapsack(3, p4, w4, k), bruteTreeKnapsack(3, p4, w4, k), `chain k=${k}`);
  }

  let checked = 0;
  let comparisons = 0;
  let expectedComparisons = 0;
  for (const seed of [4581, 4582, 4583]) {
    for (const { n, parent, weight } of lcgTrees(5, seed)) {
      let previous = 0;
      for (let k = 0; k <= n; k++) {
        const got = treeKnapsack(n, parent, weight, k);
        const want = bruteTreeKnapsack(n, parent, weight, k);
        assert.strictEqual(got, want, `LCG tree k=${k}`);
        assert.ok(got >= previous, "monotone in K");
        previous = got;
        comparisons++;
      }
      expectedComparisons += n + 1;
      checked++;
    }
  }
  assert.strictEqual(checked, 15);
  assert.strictEqual(comparisons, expectedComparisons);
  console.log("PASS");
}
