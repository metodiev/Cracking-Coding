/** Task 397 - Implement Knuth optimization (DP speedup demonstration). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

const INF = Number.POSITIVE_INFINITY;

function obstPlain(weights) {
  const n = weights.length;
  if (n === 0) return 0;
  const prefix = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + weights[i - 1];
  const e = Array.from({ length: n + 2 }, () => new Array(n + 2).fill(0));
  for (let i = 1; i <= n; i++) e[i][i] = weights[i - 1];
  for (let length = 2; length <= n; length++) {
    for (let i = 1; i + length - 1 <= n; i++) {
      const j = i + length - 1;
      const total = prefix[j] - prefix[i - 1];
      let best = INF;
      for (let k = i; k <= j; k++) {
        const left = k - 1 >= i ? e[i][k - 1] : 0;
        const right = k + 1 <= j ? e[k + 1][j] : 0;
        if (left + right + total < best) best = left + right + total;
      }
      e[i][j] = best;
    }
  }
  return e[1][n];
}

function obstKnuth(weights) {
  const n = weights.length;
  const e = Array.from({ length: n + 2 }, () => new Array(n + 2).fill(0));
  const opt = Array.from({ length: n + 2 }, () => new Array(n + 2).fill(0));
  if (n === 0) return { cost: 0, opt };
  const prefix = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + weights[i - 1];
  for (let i = 1; i <= n; i++) {
    e[i][i] = weights[i - 1];
    opt[i][i] = i;
  }
  for (let length = 2; length <= n; length++) {
    for (let i = 1; i + length - 1 <= n; i++) {
      const j = i + length - 1;
      const total = prefix[j] - prefix[i - 1];
      const lo = Math.max(opt[i][j - 1], i);
      const hi = Math.min(opt[i + 1][j], j);
      let best = INF;
      let bestK = lo;
      for (let k = lo; k <= hi; k++) {
        const left = k - 1 >= i ? e[i][k - 1] : 0;
        const right = k + 1 <= j ? e[k + 1][j] : 0;
        const candidate = left + right + total;
        if (candidate < best) {
          best = candidate;
          bestK = k;
        }
      }
      e[i][j] = best;
      opt[i][j] = bestK;
    }
  }
  return { cost: e[1][n], opt };
}

function optIsMonotone(opt, n) {
  for (let length = 2; length <= n; length++) {
    for (let i = 1; i + length - 1 <= n; i++) {
      const j = i + length - 1;
      if (opt[i][j] < opt[i][j - 1] || opt[i][j] > opt[i + 1][j]) return false;
    }
  }
  return true;
}

function rebuildCost(weights, opt, i, j, depth = 0) {
  if (i > j) return 0;
  const root = opt[i][j];
  return (
    weights[root - 1] * (depth + 1) +
    rebuildCost(weights, opt, i, root - 1, depth + 1) +
    rebuildCost(weights, opt, root + 1, j, depth + 1)
  );
}

function rebuildInorder(opt, i, j) {
  if (i > j) return [];
  const root = opt[i][j];
  return [...rebuildInorder(opt, i, root - 1), root, ...rebuildInorder(opt, root + 1, j)];
}

function lcgWeightSets(count, seed = 2026, maxN = 40) {
  const sets = [];
  for (let s = 0; s < count; s++) {
    const n = 2 + (seed % (maxN - 1));
    const weights = [];
    for (let i = 0; i < n; i++) {
      seed = (seed * 75 + 74) % 65537;
      weights.push(1 + (seed % 50));
    }
    sets.push(weights);
  }
  return sets;
}

module.exports = { obstPlain, obstKnuth, optIsMonotone, rebuildCost, rebuildInorder, lcgWeightSets };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(obstPlain([5]), 5);
  assert.strictEqual(obstKnuth([5]).cost, 5);
  assert.strictEqual(obstPlain([1, 1]), 3);
  assert.strictEqual(obstKnuth([1, 1]).cost, 3);
  assert.strictEqual(obstPlain([3, 2, 1]), 10);
  assert.strictEqual(obstKnuth([3, 2, 1]).cost, 10);
  assert.strictEqual(obstPlain([1, 1, 1, 1]), 8);
  assert.strictEqual(obstKnuth([1, 1, 1, 1]).cost, 8);

  for (const weights of lcgWeightSets(120)) {
    const plain = obstPlain(weights);
    const { cost, opt } = obstKnuth(weights);
    const n = weights.length;
    assert.strictEqual(cost, plain, weights.join(","));
    assert.ok(optIsMonotone(opt, n), weights.join(","));
    assert.strictEqual(rebuildCost(weights, opt, 1, n), cost);
    assert.deepStrictEqual(rebuildInorder(opt, 1, n), Array.from({ length: n }, (_, i) => i + 1));
  }

  const sample = lcgWeightSets(1, 31415)[0];
  const n = sample.length;
  const { opt } = obstKnuth(sample);
  let scanned = 0;
  for (let length = 2; length <= n; length++) {
    for (let i = 1; i + length - 1 <= n; i++) {
      const j = i + length - 1;
      scanned += opt[i][j] - opt[i][j - 1] + 1;
    }
  }
  assert.ok(scanned < n * n, "windows stay quadratic");
  console.log("PASS");
}
