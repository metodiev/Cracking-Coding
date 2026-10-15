/** Task 260 - Quickselect randomized implementation. (Tier: Intermediate, CTCI Ch 10) */
"use strict";

const LCG_MOD = 2147483647;
const LCG_MUL = 48271;
const SEED = 260260;

/** Dutch flag; returns {lt, gt}: a[lo..lt-1] < pivot == a[lt..gt], a[gt+1..hi] > pivot. */
function partition3(a, lo, hi, pivot) {
  let lt = lo;
  let i = lo;
  let gt = hi;
  while (i <= gt) {
    if (a[i] < pivot) {
      [a[lt], a[i]] = [a[i], a[lt]];
      lt += 1;
      i += 1;
    } else if (a[i] > pivot) {
      [a[i], a[gt]] = [a[gt], a[i]];
      gt -= 1;
    } else {
      i += 1;
    }
  }
  return { lt, gt };
}

/**
 * Randomized quickselect with a fixed-seed LCG (reproducible runs); the returned value
 * never depends on the seed. Duplicates collapse into one equal block, so all-equal input
 * finishes after a single partition; the loop is iterative, so no stack blow-up. Expected
 * O(n), worst case O(n^2). k is 1-based; the caller's array is copied.
 */
function kthSmallest(values, k, seed = SEED) {
  if (k < 1 || k > values.length) throw new RangeError("k out of range");
  const a = values.slice();
  let state = seed % LCG_MOD;
  if (state === 0) state = 1;
  let lo = 0;
  let hi = a.length - 1;
  const target = k - 1;
  for (;;) {
    state = (state * LCG_MUL) % LCG_MOD;
    const pivot = a[lo + (state % (hi - lo + 1))];
    const { lt, gt } = partition3(a, lo, hi, pivot);
    if (target < lt) hi = lt - 1;
    else if (target > gt) lo = gt + 1;
    else return a[target];
  }
}

function lcgValues(seed, count, bound) {
  const out = [];
  let state = seed;
  for (let i = 0; i < count; i += 1) {
    state = (state * LCG_MUL) % LCG_MOD;
    out.push(state % bound);
  }
  return out;
}

/** Every k agrees with sorted order, and the value is seed-independent. */
function agreesWithOracle(values) {
  const ordered = values.slice().sort((a, b) => a - b);
  for (let k = 1; k <= values.length; k += 1) {
    if (kthSmallest(values, k) !== ordered[k - 1]) return false;
    if (kthSmallest(values, k, 999) !== ordered[k - 1]) return false;
  }
  return true;
}

module.exports = { partition3, kthSmallest, lcgValues, agreesWithOracle };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(kthSmallest([7], 1), 7);
  assert.strictEqual(kthSmallest([3, 1, 2], 2), 2);
  assert.strictEqual(kthSmallest([5, 5, 5, 5], 3), 5);
  assert.strictEqual(kthSmallest([2, 1, 1, 3], 2), 1);

  const original = [9, 4, 7, 1, 8, 2, 6, 3, 5];
  assert.strictEqual(kthSmallest(original, 4), 4);
  assert.deepStrictEqual(original, [9, 4, 7, 1, 8, 2, 6, 3, 5]);   // input untouched

  for (const bad of [0, -1, 10]) {
    assert.throws(() => kthSmallest([1, 2, 3], bad), RangeError);
  }

  const range = (n) => Array.from({ length: n }, (_, i) => i);
  const reverseRange = (n) => Array.from({ length: n }, (_, i) => n - 1 - i);
  const cases = [
    lcgValues(260001, 1, 5),
    lcgValues(260002, 2, 5),
    lcgValues(260003, 3, 7),
    lcgValues(260004, 5, 20),
    lcgValues(260005, 8, 3),
    lcgValues(260006, 16, 1000),
    lcgValues(260007, 33, 10),
    lcgValues(260008, 50, 50),
    range(64),
    reverseRange(64),
    new Array(17).fill(9),
    Array.from({ length: 200 }, (_, i) => 1 + (i % 2)),
    lcgValues(260009, 200, 10000),
  ];
  for (const c of cases) assert.ok(agreesWithOracle(c), `oracle mismatch for n=${c.length}`);

  const bigEqual = new Array(5000).fill(7);
  for (const k of [1, 2, 2500, 4999, 5000]) {
    assert.strictEqual(kthSmallest(bigEqual, k), 7);
  }
  const ascending = range(4000);
  const descending = reverseRange(4000);
  const sawtooth = Array.from({ length: 1000 }, (_, i) => i % 17);
  const sawOrdered = sawtooth.slice().sort((a, b) => a - b);
  for (const k of [1, 2000, 4000]) {
    assert.strictEqual(kthSmallest(ascending, k), k - 1);
    assert.strictEqual(kthSmallest(descending, k), k - 1);
  }
  for (const k of [1, 500, 1000]) {
    assert.strictEqual(kthSmallest(sawtooth, k), sawOrdered[k - 1]);
  }
  console.log("PASS");
}
