/** Task 337 - Design majority vote generalized (> n/k) using count map. (Tier: Intermediate, CTCI Ch 7) */
"use strict";

/**
 * Generalised Boyer-Moore: keep at most k - 1 candidate counters; a new value with no free
 * slot cancels one vote from every counter (dropping zeros). A verification pass then keeps
 * exactly the values with count * k > n. O(n * k) time, O(k) space. Cross-checked against a
 * brute-force counting answer on deterministic LCG arrays.
 */
function majorityElements(values, k) {
  if (k < 2) throw new Error("k must be >= 2");
  const n = values.length;
  if (n === 0) return [];

  const slots = new Map();
  for (const value of values) {
    if (slots.has(value)) {
      slots.set(value, slots.get(value) + 1);
    } else if (slots.size < k - 1) {
      slots.set(value, 1);
    } else {
      for (const candidate of [...slots.keys()]) {
        slots.set(candidate, slots.get(candidate) - 1);
        if (slots.get(candidate) === 0) slots.delete(candidate);
      }
    }
  }

  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
  return [...slots.keys()].filter((value) => counts.get(value) * k > n).sort((a, b) => a - b);
}

function bruteMajority(values, k) {
  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
  return [...counts.keys()].filter((value) => counts.get(value) * k > values.length).sort((a, b) => a - b);
}

module.exports = { majorityElements, bruteMajority };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-checked with n = 6, k = 4: the bar is count * 4 > 6, so 1 (3 times) and 2 (twice)
  // qualify while 3 (once) does not.
  assert.deepStrictEqual(majorityElements([1, 1, 1, 2, 2, 3], 4), [1, 2]);
  assert.deepStrictEqual(bruteMajority([1, 1, 1, 2, 2, 3], 4), [1, 2]);

  // Classic majority (k = 2): the bar is count * 2 > n.
  assert.deepStrictEqual(majorityElements([3, 2, 3], 2), [3]);
  assert.deepStrictEqual(majorityElements([2, 2, 1, 1, 1, 2, 2], 2), [2]);

  // No element passes the bar.
  assert.deepStrictEqual(majorityElements([1, 2, 3, 4], 4), []);
  assert.deepStrictEqual(majorityElements([1, 2, 3, 4], 2), []);
  assert.deepStrictEqual(majorityElements([1, 2, 2, 3, 3, 4], 3), []);
  assert.deepStrictEqual(majorityElements([], 3), []);

  // Exactly on the boundary: count * k == n is not "more than n/k".
  assert.deepStrictEqual(majorityElements([5, 5, 6, 6], 2), []);
  assert.deepStrictEqual(majorityElements([5, 5, 6, 6], 3), [5, 6]);

  // Every distinct value can pass when k is large enough.
  assert.deepStrictEqual(majorityElements([7, 8, 9, 7, 8, 9], 4), [7, 8, 9]);

  let seed = 6060;
  for (let trial = 0; trial < 400; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const n = seed % 60;
    seed = (seed * 75 + 74) % 65537;
    const k = 2 + (seed % 5);
    seed = (seed * 75 + 74) % 65537;
    const alphabet = 1 + (seed % 9);
    const values = [];
    for (let i = 0; i < n; i++) {
      seed = (seed * 75 + 74) % 65537;
      values.push(seed % alphabet);
    }
    const found = majorityElements(values, k);
    assert.deepStrictEqual(found, bruteMajority(values, k), `random set ${trial}`);
    assert.ok(found.length <= k - 1, `bounded candidates ${trial}`);
    const counts = new Map();
    for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
    for (const value of found) {
      assert.ok(counts.get(value) * k > n, `every report exceeds n/k ${trial}`);
    }
  }
  console.log("PASS");
}
