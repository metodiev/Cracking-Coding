/** Task 398 - Implement a divide-and-conquer DP optimization example. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

const INF = Number.POSITIVE_INFINITY;

function partitionNaive(values, k) {
  const n = values.length;
  if (k > n) return INF;
  const prefix = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + values[i - 1];
  let previous = [0, ...new Array(n).fill(INF)];
  for (let t = 1; t <= k; t++) {
    const current = new Array(n + 1).fill(INF);
    for (let i = t; i <= n; i++) {
      for (let j = t - 1; j < i; j++) {
        if (previous[j] === INF) continue;
        const group = prefix[i] - prefix[j];
        const cost = previous[j] + group * group;
        if (cost < current[i]) current[i] = cost;
      }
    }
    previous = current;
  }
  return previous[n];
}

function partitionDc(values, k) {
  const n = values.length;
  if (k > n) return { answer: INF, allOpts: [] };
  const prefix = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + values[i - 1];
  let previous = [0, ...new Array(n).fill(INF)];
  const allOpts = [];
  for (let t = 1; t <= k; t++) {
    const current = new Array(n + 1).fill(INF);
    const opt = new Array(n + 1).fill(0);
    // i == t: the only split is j == t-1; the last singleton group costs values[t-1]^2
    current[t] = previous[t - 1] + values[t - 1] * values[t - 1];
    opt[t] = t - 1;
    if (t + 1 <= n) solve(previous, current, opt, prefix, t + 1, n, t - 1, n - 1);
    allOpts.push(opt);
    previous = current;
  }
  return { answer: previous[n], allOpts };
}

function solve(previous, current, opt, prefix, l, r, optl, optr) {
  if (l > r) return;
  const mid = (l + r) >> 1;
  let best = INF;
  let bestJ = optl;
  const upper = Math.min(optr, mid - 1);
  for (let j = optl; j <= upper; j++) {
    if (previous[j] === INF) continue;
    const group = prefix[mid] - prefix[j];
    const cost = previous[j] + group * group;
    if (cost < best) {
      best = cost;
      bestJ = j;
    }
  }
  current[mid] = best;
  opt[mid] = bestJ;
  solve(previous, current, opt, prefix, l, mid - 1, optl, bestJ);
  solve(previous, current, opt, prefix, mid + 1, r, bestJ, optr);
}

function optIsMonotone(opt, t, n) {
  let last = -1;
  for (let i = t; i <= n; i++) {
    if (opt[i] < last) return false;
    last = opt[i];
  }
  return true;
}

function lcgArrays(count, seed = 804, maxLen = 60) {
  const arrays = [];
  for (let a = 0; a < count; a++) {
    const n = 2 + (seed % (maxLen - 1));
    const values = [];
    for (let i = 0; i < n; i++) {
      seed = (seed * 75 + 74) % 65537;
      values.push(1 + (seed % 9));
    }
    arrays.push(values);
  }
  return arrays;
}

module.exports = { partitionNaive, partitionDc, optIsMonotone, lcgArrays };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(partitionNaive([1, 2, 3, 4], 2), 52);
  assert.strictEqual(partitionDc([1, 2, 3, 4], 2).answer, 52);
  assert.strictEqual(partitionNaive([1, 2, 3, 4], 1), 100);
  assert.strictEqual(partitionNaive([1, 2, 3, 4], 4), 30);
  assert.strictEqual(partitionNaive([5], 1), 25);
  assert.strictEqual(partitionNaive([1, 2], 3), INF);
  assert.strictEqual(partitionDc([1, 2, 3, 4], 4).answer, 30);

  for (const values of lcgArrays(40)) {
    const n = values.length;
    for (let k = 1; k <= Math.min(n, 8); k++) {
      const naive = partitionNaive(values, k);
      const { answer, allOpts } = partitionDc(values, k);
      assert.strictEqual(answer, naive, `${values.join(",")} k=${k}`);
      allOpts.forEach((opt, index) => {
        assert.ok(optIsMonotone(opt, index + 1, n), `${values.join(",")} k=${k} t=${index + 1}`);
      });
    }
  }
  console.log("PASS");
}
