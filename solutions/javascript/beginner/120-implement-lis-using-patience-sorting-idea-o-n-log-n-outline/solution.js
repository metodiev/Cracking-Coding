/** Task 120 - Implement LIS using patience sorting idea (O(n log n) outline). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function lowerBound(arr, x, size) {
  let lo = 0, hi = size;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function lisLength(a) {
  const tails = [];
  for (const x of a) {
    const i = lowerBound(tails, x, tails.length);
    if (i === tails.length) tails.push(x);
    else tails[i] = x;
  }
  return tails.length;
}

function lisReconstructed(a) {
  if (a.length === 0) return [];
  const tails = [], tailIdx = [], parent = new Array(a.length).fill(-1);
  for (let i = 0; i < a.length; i++) {
    const j = lowerBound(tails, a[i], tails.length);
    if (j === tails.length) {
      tails.push(a[i]);
      tailIdx.push(i);
    } else {
      tails[j] = a[i];
      tailIdx[j] = i;
    }
    parent[i] = j > 0 ? tailIdx[j - 1] : -1;
  }
  const seq = [];
  for (let node = tailIdx[tailIdx.length - 1]; node !== -1; node = parent[node]) seq.push(a[node]);
  return seq.reverse();
}

module.exports = { lisLength, lisReconstructed };

if (require.main === module) {
  const assert = require("node:assert");
  const a = [10, 9, 2, 5, 3, 7, 101, 18];
  assert.strictEqual(lisLength(a), 4);
  assert.strictEqual(lisLength([]), 0);
  assert.strictEqual(lisLength([3, 2, 1]), 1);
  assert.deepStrictEqual(lisReconstructed(a), [2, 3, 7, 18]);
  console.log("PASS");
}
