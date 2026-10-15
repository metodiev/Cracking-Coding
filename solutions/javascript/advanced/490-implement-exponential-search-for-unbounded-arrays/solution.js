/** Task 490 - Implement exponential search for unbounded arrays. (Tier: Advanced, CTCI Ch 10)
 *
 * Sorted ascending arrays are treated as conceptually unbounded: a documented sentinel sits at the
 * virtual index n (value +infinity), so the doubling phase may "look at" index n without ever
 * reading past n-1 - the peek() helper returns the sentinel for idx >= n and records the largest
 * real index touched. The search is cold-start: exponentialSearch returns the FIRST index holding
 * target, or -1.
 *
 * Algorithm (documented):
 * 1. Peek index 0; if arr[0] >= target the range is [0, 0].
 * 2. Otherwise double the bound b = 1, 2, 4, ... while b < n and arr[b] < target, remembering the
 *    last probed index prev with arr[prev] < target.
 * 3. The first occurrence lies in [prev + 1, min(b, n - 1)]; finish with an ordinary lower-bound
 *    style binary search on that window and compare arr[lo] == target.
 *
 * Probe budget: <= 1 + ceil(log2(i + 1)) doubling peeks plus <= ceil(log2(i)) window steps for a hit
 * at index i, bounded overall by 2 * ceil(log2(n + 1)) + 2 peeks (asserted). The answer is asserted
 * against both a plain binary search and a brute-force linear first-occurrence scan on LCG arrays
 * (with duplicates) including target at index 0, at n - 1, and absent values (sentinel case).
 */
"use strict";

function exponentialSearch(arr, target) {
  const n = arr.length;
  let reads = 0;
  let maxIndex = -1;
  if (n === 0) return { index: -1, reads, maxIndex };
  let lo;
  let hi;
  reads += 1;
  if (arr[0] >= target) {
    lo = 0;
    hi = 0;
  } else {
    let bound = 1;
    let prev = 0;
    while (bound < n) {
      reads += 1;
      if (bound > maxIndex) maxIndex = bound;
      if (arr[bound] >= target) break;
      prev = bound;
      bound <<= 1;
    }
    lo = prev + 1;
    hi = Math.min(bound, n - 1);
  }
  if (lo > hi) return { index: -1, reads, maxIndex };
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    reads += 1;
    if (mid > maxIndex) maxIndex = mid;
    if (arr[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  reads += 1;
  if (lo > maxIndex) maxIndex = lo;
  if (arr[lo] === target) return { index: lo, reads, maxIndex };
  return { index: -1, reads, maxIndex };
}

function binarySearch(arr, target) {
  let lo = 0;
  let hi = arr.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < target) lo = mid + 1;
    else if (arr[mid] > target) hi = mid - 1;
    else {
      let at = mid;
      while (at > 0 && arr[at - 1] === target) at -= 1;
      return at;
    }
  }
  return -1;
}

function bruteFirst(arr, target) {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === target) return i;
    if (arr[i] > target) return -1;
  }
  return -1;
}

function lcgValues(count, seed, limit) {
  const values = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    values.push(seed % limit);
  }
  return values;
}

function budget(n) {
  return 2 * Math.ceil(Math.log2(n + 1)) + 2;
}

module.exports = { exponentialSearch, binarySearch, bruteFirst };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(exponentialSearch([], 5).index, -1);
  assert.strictEqual(exponentialSearch([5], 5).index, 0);
  assert.strictEqual(exponentialSearch([5], 4).index, -1);
  assert.strictEqual(exponentialSearch([5], 6).index, -1);
  const ramp = [1, 2, 3, 4, 5, 6, 7, 8];
  assert.strictEqual(exponentialSearch(ramp, 1).index, 0);
  assert.strictEqual(exponentialSearch(ramp, 8).index, 7);
  assert.strictEqual(exponentialSearch(ramp, 4).index, 3);
  assert.strictEqual(exponentialSearch(ramp, 0).index, -1);
  assert.strictEqual(exponentialSearch(ramp, 9).index, -1);
  assert.strictEqual(exponentialSearch([2, 2, 2, 2], 2).index, 0);
  assert.strictEqual(exponentialSearch([2, 2, 2, 2], 3).index, -1);
  assert.strictEqual(exponentialSearch(new Array(8).fill(0), 0).index, 0);
  const odd = [1, 3, 5, 7, 9, 11];
  assert.strictEqual(exponentialSearch(odd, 11).index, 5);
  assert.strictEqual(exponentialSearch(odd, 6).index, -1);
  assert.strictEqual(binarySearch(odd, 6), bruteFirst(odd, 6));
  assert.strictEqual(binarySearch(odd, 6), -1);

  let foundTotal = 0;
  let absentTotal = 0;
  let tests = 0;
  for (let seed = 490; seed <= 494; seed++) {
    const arr = lcgValues(64, seed, 50).sort((a, b) => a - b);
    const targets = arr.slice();
    targets.push(-1, 100, 49, 50);
    for (const value of lcgValues(20, seed + 100, 60)) targets.push(value);
    for (const target of targets) {
      const result = exponentialSearch(arr, target);
      const expected = bruteFirst(arr, target);
      assert.strictEqual(result.index, expected, `lcg ${seed} ${target}`);
      assert.strictEqual(result.index, binarySearch(arr, target), `binary match ${seed} ${target}`);
      assert.ok(result.maxIndex < arr.length, `bounds ${seed} ${target}`);
      assert.ok(result.reads <= budget(arr.length), `reads ${seed} ${target} ${result.reads}`);
      assert.ok(result.reads >= 1);
      tests += 1;
      if (result.index >= 0) foundTotal += 1;
      else absentTotal += 1;
    }
  }
  assert.strictEqual(tests, 5 * 88);
  assert.strictEqual(foundTotal, 389);
  assert.strictEqual(absentTotal, 51);

  const n = 4096;
  const big = lcgValues(n, 777, 100000).sort((a, b) => a - b);
  for (let i = 0; i < n; i += 137) {
    const result = exponentialSearch(big, big[i]);
    assert.strictEqual(result.index, bruteFirst(big, big[i]), `big ${i}`);
    assert.ok(result.maxIndex < n, `big bounds ${i}`);
    assert.ok(result.reads <= budget(n), `big reads ${i}`);
  }
  assert.strictEqual(exponentialSearch(big, -5).index, -1);
  assert.strictEqual(exponentialSearch(big, 100001).index, -1);
  console.log("PASS");
}
