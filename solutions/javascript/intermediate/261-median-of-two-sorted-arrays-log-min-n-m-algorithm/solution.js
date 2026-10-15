/** Task 261 - Median of two sorted arrays (log min(n,m) algorithm). (Tier: Intermediate, CTCI Ch 10) */
"use strict";

/**
 * Partition-based binary search on the SHORTER array: i = number of a-elements on the
 * left side, j = (m+n+1)>>1 - i follows. Valid when a[i-1] <= b[j] and b[j-1] <= a[i];
 * +-Infinity sentinels cover empty arrays and the fully-left/fully-right cases.
 * O(log min(m, n)); two empty arrays throw (documented boundary).
 */
function medianOfTwoSorted(a, b) {
  if (a.length === 0 && b.length === 0) throw new Error("at least one array must be non-empty");
  if (a.length > b.length) { const t = a; a = b; b = t; }   // binary search the shorter array
  const m = a.length, n = b.length;
  const half = (m + n + 1) >> 1;
  let lo = 0, hi = m;
  while (lo <= hi) {
    const i = (lo + hi) >> 1;
    const j = half - i;
    const aLeft = i > 0 ? a[i - 1] : -Infinity;
    const aRight = i < m ? a[i] : Infinity;
    const bLeft = j > 0 ? b[j - 1] : -Infinity;
    const bRight = j < n ? b[j] : Infinity;
    if (aLeft <= bRight && bLeft <= aRight) {               // valid partition found
      if ((m + n) % 2 === 1) return Math.max(aLeft, bLeft);
      return (Math.max(aLeft, bLeft) + Math.min(aRight, bRight)) / 2;
    }
    if (aLeft > bRight) hi = i - 1;                         // too many a on the left
    else lo = i + 1;                                        // too few a on the left
  }
  throw new Error("inputs are not sorted");
}

function mergeSorted(a, b) {
  const out = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) out.push(a[i] <= b[j] ? a[i++] : b[j++]);
  while (i < a.length) out.push(a[i++]);
  while (j < b.length) out.push(b[j++]);
  return out;
}

/** O(m+n) merged-middle reference used only for cross-checks. */
function medianOracle(a, b) {
  const merged = mergeSorted(a, b);
  const n = merged.length;
  return n % 2 === 1 ? merged[(n - 1) >> 1] : (merged[n / 2 - 1] + merged[n / 2]) / 2;
}

module.exports = { medianOfTwoSorted, mergeSorted, medianOracle };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(medianOfTwoSorted([1, 3], [2]), 2.0);
  assert.strictEqual(medianOfTwoSorted([1, 2], [3, 4]), 2.5);
  assert.strictEqual(medianOfTwoSorted([], [1]), 1.0);
  assert.strictEqual(medianOfTwoSorted([1, 2, 3, 4, 5], []), 3.0);
  assert.strictEqual(medianOfTwoSorted([0, 0], [0, 0]), 0.0);
  assert.strictEqual(medianOfTwoSorted([2], [1, 3]), 2.0);
  assert.strictEqual(medianOfTwoSorted([1, 3, 8], [7, 9, 10, 11]), 8.0);   // classic case
  assert.strictEqual(medianOfTwoSorted([1, 2], [3, 4, 5, 6]), 3.5);        // fully left
  assert.strictEqual(medianOfTwoSorted([5, 6, 7, 8], [1, 2, 3, 4]), 4.5);  // fully right
  assert.strictEqual(medianOfTwoSorted([1, 1], [1, 1]), 1.0);
  assert.strictEqual(medianOfTwoSorted([-5, -3, -1], [-2]), -2.5);
  assert.strictEqual(medianOfTwoSorted([1, 2, 3], [4, 5, 6]), 3.5);
  assert.throws(() => medianOfTwoSorted([], []), /non-empty/);

  // deterministic LCG so the cross-check is reproducible
  let seed = 261;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  for (let test = 0; test < 300; test++) {      // cross-check vs the merge oracle
    const a = Array.from({ length: randInt(0, 8) }, () => randInt(-20, 20)).sort((x, y) => x - y);
    const b = Array.from({ length: randInt(0, 8) }, () => randInt(-20, 20)).sort((x, y) => x - y);
    if (a.length === 0 && b.length === 0) continue;
    assert.ok(Math.abs(medianOfTwoSorted(a, b) - medianOracle(a, b)) < 1e-9, `random agrees: ${a} / ${b}`);
  }
  console.log("PASS");
}
