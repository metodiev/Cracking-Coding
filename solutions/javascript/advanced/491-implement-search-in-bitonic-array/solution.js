/**
 * Task 491 - Implement search in bitonic array. (Tier: Advanced, CTCI Ch 10)
 *
 * A bitonic (mountain) array strictly rises to a single peak and then strictly falls; the peak
 * may sit at either end, so purely increasing and purely decreasing arrays are the degenerate
 * cases. findPeak is the standard O(log n) binary search that compares values[mid] with its
 * neighbour; search then binary searches the ascending side [0, peak] and the descending side
 * (peak, n-1] separately, for O(log n) total. An absent target returns the documented sentinel
 * -1 (indices are non-negative, so -1 is unambiguous). Verified against a linear-scan oracle
 * on every element and on absent probes over LCG-built bitonic arrays of many sizes, including
 * single-element arrays and arrays whose peak is at a boundary.
 */
"use strict";

const SENTINEL = -1;

/** Index of the unique maximum (the peak) of a strictly bitonic array. */
function findPeak(values) {
  let lo = 0;
  let hi = values.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] < values[mid + 1]) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/** Index of target in a strictly bitonic array, or SENTINEL (-1) when absent. */
function search(values, target) {
  const n = values.length;
  if (n === 0) return SENTINEL;
  const peak = findPeak(values);

  let lo = 0;
  let hi = peak;                           // ascending side
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] === target) return mid;
    if (values[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  lo = peak + 1;
  hi = n - 1;                              // descending side
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] === target) return mid;
    if (values[mid] > target) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  return SENTINEL;
}

/** Simple oracle: first index holding target, or SENTINEL. */
function linearSearch(values, target) {
  for (let i = 0; i < values.length; i++) {
    if (values[i] === target) return i;
  }
  return SENTINEL;
}

/** Strictly bitonic array of n distinct LCG values with the peak at `peak`. */
function lcgBitonic(n, seed, peak) {
  const distinct = [];
  const seen = new Set();
  while (distinct.length < n) {
    seed = (seed * 75 + 74) % 65537;
    const value = seed % 1000;
    if (!seen.has(value)) {
      seen.add(value);
      distinct.push(value);
    }
  }
  distinct.sort((a, b) => a - b);
  const tailLen = n - 1 - peak;            // values strictly below the peak, descending
  const descending = [];
  for (let i = tailLen - 1; i >= 0; i--) descending.push(distinct[i]);
  return distinct.slice(tailLen).concat(descending);
}

function main() {
  const assert = require("node:assert");

  const sample = [1, 3, 5, 9, 7, 4, 2];
  assert.strictEqual(findPeak(sample), 3);
  sample.forEach((value, i) => assert.strictEqual(search(sample, value), i));
  for (const absent of [0, 6, 8, 10, -5]) assert.strictEqual(search(sample, absent), SENTINEL);

  assert.strictEqual(findPeak([5]), 0);
  assert.strictEqual(search([5], 5), 0);
  assert.strictEqual(search([5], 4), SENTINEL);
  assert.strictEqual(findPeak([1, 2, 3, 4]), 3);
  assert.strictEqual(search([1, 2, 3, 4], 4), 3);
  assert.strictEqual(search([1, 2, 3, 4], 0), SENTINEL);
  assert.strictEqual(findPeak([4, 3, 2, 1]), 0);
  assert.strictEqual(search([4, 3, 2, 1], 1), 3);
  assert.strictEqual(search([4, 3, 2, 1], 5), SENTINEL);

  let checkedArrays = 0;
  for (let n = 1; n <= 17; n++) {
    for (let peak = 0; peak < n; peak++) {
      const values = lcgBitonic(n, 1000 + 13 * n + peak, peak);
      checkedArrays++;
      assert.strictEqual(findPeak(values), peak, `${n}/${peak}`);
      for (let i = 0; i < n; i++) {
        assert.strictEqual(search(values, values[i]), i);
        assert.strictEqual(search(values, values[i]), linearSearch(values, values[i]));
      }
      for (const absent of [values[0] - 1, values[peak] + 1, 1000, -1, 65537]) {
        assert.strictEqual(search(values, absent), SENTINEL, `${n}/${peak}/${absent}`);
        assert.strictEqual(linearSearch(values, absent), SENTINEL);
      }
    }
  }
  assert.strictEqual(checkedArrays, (17 * 18) / 2, checkedArrays);

  const larger = [[256, 4242, 100], [257, 777, 0], [255, 31337, 254], [300, 5, 150]];
  for (const [n, seed, peak] of larger) {
    const values = lcgBitonic(n, seed, peak);
    assert.strictEqual(findPeak(values), peak);
    for (const i of [0, 1, n >> 1, peak - 1, peak, peak + 1, n - 2, n - 1]) {
      if (i >= 0 && i < n) assert.strictEqual(search(values, values[i]), i);
    }
    assert.strictEqual(search(values, values[peak] + 7), SENTINEL);
  }
}

module.exports = { SENTINEL, findPeak, search, linearSearch, lcgBitonic };

if (require.main === module) {
  main();
  console.log("PASS");
}
