/** Task 119 - Longest increasing subsequence (naive DP O(n^2)). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function lisLength(a) {
  if (a.length === 0) return 0;
  const dp = new Array(a.length).fill(1);
  let best = 1;
  for (let i = 1; i < a.length; i++) {
    for (let j = 0; j < i; j++) {
      if (a[j] < a[i] && dp[j] + 1 > dp[i]) dp[i] = dp[j] + 1;
    }
    best = Math.max(best, dp[i]);
  }
  return best;
}

function lisWithSequence(a) {
  if (a.length === 0) return [];
  const dp = new Array(a.length).fill(1);
  const parent = new Array(a.length).fill(-1);
  let best = 0;
  for (let i = 1; i < a.length; i++) {
    for (let j = 0; j < i; j++) {
      if (a[j] < a[i] && dp[j] + 1 > dp[i]) {
        dp[i] = dp[j] + 1;
        parent[i] = j;
      }
    }
    if (dp[i] >= dp[best]) best = i;   // >= keeps the LAST index
  }
  const seq = [];
  for (let node = best; node !== -1; node = parent[node]) seq.push(a[node]);
  return seq.reverse();
}

module.exports = { lisLength, lisWithSequence };

if (require.main === module) {
  const assert = require("node:assert");
  const a = [10, 9, 2, 5, 3, 7, 101, 18];
  assert.strictEqual(lisLength(a), 4);
  assert.strictEqual(lisLength([]), 0);
  assert.strictEqual(lisLength([3, 2, 1]), 1);
  assert.deepStrictEqual(lisWithSequence(a), [2, 5, 7, 18]);
  console.log("PASS");
}
