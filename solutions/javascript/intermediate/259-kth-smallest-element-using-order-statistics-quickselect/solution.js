/** Task 259 - Kth smallest element using order statistics / quickselect. (Tier: Intermediate, CTCI Beyond — Numerical & Scientific Computing) */
"use strict";

const LCG_MOD = 2147483647;
const LCG_MUL = 48271;

/** Lomuto partition with a[hi] as the deterministic pivot; returns its final index. */
function partition(a, lo, hi) {
  const pivot = a[hi];
  let store = lo;
  for (let i = lo; i < hi; i += 1) {
    if (a[i] <= pivot) {
      [a[store], a[i]] = [a[i], a[store]];
      store += 1;
    }
  }
  [a[store], a[hi]] = [a[hi], a[store]];
  return store;
}

/** The k-th smallest value (1-based k); the caller's array is copied first. */
function kthSmallest(values, k) {
  if (k < 1 || k > values.length) throw new RangeError("k out of range");
  const a = values.slice();
  let lo = 0;
  let hi = a.length - 1;
  const target = k - 1;
  for (;;) {
    const p = partition(a, lo, hi);            // iterative: a bad pivot never grows a stack
    if (p === target) return a[p];
    if (p < target) lo = p + 1;
    else hi = p - 1;
  }
}

/** Park-Miller minimal standard generator: identical values in every language. */
function lcgValues(seed, count, bound) {
  const out = [];
  let state = seed;
  for (let i = 0; i < count; i += 1) {
    state = (state * LCG_MUL) % LCG_MOD;
    out.push(state % bound);
  }
  return out;
}

/** Every k agrees with the sorted() oracle. */
function caseAgainstOracle(values) {
  const ordered = values.slice().sort((a, b) => a - b);
  for (let k = 1; k <= values.length; k += 1) {
    if (kthSmallest(values, k) !== ordered[k - 1]) return false;
  }
  return true;
}

module.exports = { partition, kthSmallest, lcgValues, caseAgainstOracle };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(kthSmallest([7], 1), 7);
  assert.strictEqual(kthSmallest([3, 1, 2], 1), 1);
  assert.strictEqual(kthSmallest([3, 1, 2], 2), 2);
  assert.strictEqual(kthSmallest([3, 1, 2], 3), 3);
  assert.strictEqual(kthSmallest([5, 5, 5, 5], 3), 5);
  assert.strictEqual(kthSmallest([2, 1, 1, 3], 2), 1);       // sorted: [1, 1, 2, 3]

  const original = [9, 4, 7, 1, 8, 2, 6, 3, 5];
  assert.strictEqual(kthSmallest(original, 4), 4);
  assert.deepStrictEqual(original, [9, 4, 7, 1, 8, 2, 6, 3, 5]);   // input untouched

  for (const bad of [0, -1, 10]) {
    assert.throws(() => kthSmallest([1, 2, 3], bad), RangeError);
  }

  const range = (n) => Array.from({ length: n }, (_, i) => i);
  const reverseRange = (n) => Array.from({ length: n }, (_, i) => n - 1 - i);
  const cases = [
    lcgValues(259001, 1, 5),
    lcgValues(259002, 2, 5),
    lcgValues(259003, 3, 7),
    lcgValues(259004, 5, 20),
    lcgValues(259005, 8, 3),
    lcgValues(259006, 16, 1000),
    lcgValues(259007, 33, 10),
    lcgValues(259008, 50, 50),
    range(64),
    reverseRange(64),
    new Array(32).fill(7),
    [5, 5, 4, 4, 3, 3, 2, 2, 1, 1],
    lcgValues(259009, 200, 10000),
  ];
  for (const c of cases) assert.ok(caseAgainstOracle(c), `oracle mismatch for n=${c.length}`);
  console.log("PASS");
}
