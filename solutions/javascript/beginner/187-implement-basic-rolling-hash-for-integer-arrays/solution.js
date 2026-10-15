/** Task 187 - Implement basic rolling hash for integer arrays. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

const DEFAULT_BASE = 257;
const DEFAULT_MOD = 1000000007;

function powMod(base, exponent, mod) {
  let result = 1;
  for (let i = 0; i < exponent; i++) result = (result * base) % mod;
  return result;
}

const normalize = (value, mod) => ((value % mod) + mod) % mod;
const windowKey = (values, start, k) => values.slice(start, start + k).join(",");

/**
 * Map each repeated window's first start index to its later occurrences. Each length-k
 * window hash slides in O(1); equal hashes are only candidates, so every bucket is
 * confirmed by comparing the windows themselves (a collision cannot fake a duplicate).
 */
function findDuplicateWindows(values, k, base = DEFAULT_BASE, mod = DEFAULT_MOD) {
  if (k <= 0) throw new RangeError("window length must be positive");
  const n = values.length;
  if (k > n) return {};

  let highest = powMod(base, k - 1, mod);
  let windowHash = 0;
  for (let i = 0; i < k; i++) windowHash = (windowHash * base + normalize(values[i], mod)) % mod;

  const buckets = new Map();
  const addToBucket = (hash, start) => {
    if (!buckets.has(hash)) buckets.set(hash, []);
    buckets.get(hash).push(start);
  };
  addToBucket(windowHash, 0);
  for (let start = 1; start + k <= n; start++) {
    windowHash = (windowHash - normalize(values[start - 1], mod) * highest) % mod;
    if (windowHash < 0) windowHash += mod;
    windowHash = (windowHash * base + normalize(values[start + k - 1], mod)) % mod;
    addToBucket(windowHash, start);
  }

  // equal hashes are only candidates: compare the windows themselves
  const duplicates = {};
  for (const indices of buckets.values()) {
    if (indices.length < 2) continue;
    const groups = new Map();
    for (const index of indices) {
      const key = windowKey(values, index, k);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(index);
    }
    for (const positions of groups.values()) {
      if (positions.length > 1) duplicates[positions[0]] = positions.slice(1);
    }
  }
  return duplicates;
}

/** Reference implementation: group windows directly, no hashing. */
function findDuplicateWindowsNaive(values, k) {
  const duplicates = {};
  if (k <= 0 || k > values.length) return duplicates;
  const groups = new Map();
  for (let start = 0; start + k <= values.length; start++) {
    const key = windowKey(values, start, k);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(start);
  }
  for (const positions of groups.values()) {
    if (positions.length > 1) duplicates[positions[0]] = positions.slice(1);
  }
  return duplicates;
}

/** Rendering such as "0:[3, 6];1:[4];2:[5]" so tests read the whole answer. */
const describe = (duplicates) =>
  Object.keys(duplicates)
    .map(Number)
    .sort((a, b) => a - b)
    .map((first) => `${first}:[${duplicates[first].join(", ")}]`)
    .join(";");

module.exports = { findDuplicateWindows, findDuplicateWindowsNaive };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(describe(findDuplicateWindows([1, 2, 3, 1, 2, 3, 1, 2, 3], 3)), "0:[3, 6];1:[4];2:[5]");
  assert.strictEqual(describe(findDuplicateWindows([2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], 3)), "0:[3, 6];1:[4, 7];2:[5, 8]");
  assert.strictEqual(describe(findDuplicateWindows([5, 5, 5, 5], 2)), "0:[1, 2]");
  assert.strictEqual(describe(findDuplicateWindows([4, 7, 4, 7, 4], 2)), "0:[2];1:[3]");
  assert.strictEqual(describe(findDuplicateWindows([7, 7, 8, 7, 7], 1)), "0:[1, 3, 4]");
  assert.strictEqual(describe(findDuplicateWindows([1, 2, 3, 1, 2, 4], 3)), "");
  assert.strictEqual(describe(findDuplicateWindows([1, 2, 3, 4], 4)), "");
  assert.strictEqual(describe(findDuplicateWindows([1, 2, 3, 4], 5)), "");
  assert.strictEqual(describe(findDuplicateWindows([], 2)), "");
  assert.strictEqual(describe(findDuplicateWindows([-5, -5, 3, -5, -5], 2)), "0:[3]");
  assert.throws(() => findDuplicateWindows([1, 2], 0), RangeError);

  // agreement with the direct grouping on pseudo-random arrays
  let state = 23;
  const nextInt = (bound) => {
    state = (state * 48271) % 2147483647;
    return state % bound;
  };
  const randomArray = (size, bound) => Array.from({ length: size }, () => nextInt(bound));
  for (let t = 0; t < 200; t++) {
    const values = randomArray(nextInt(31), 4);
    const k = 1 + nextInt(4);
    assert.strictEqual(
      describe(findDuplicateWindows(values, k)),
      describe(findDuplicateWindowsNaive(values, k)),
      `cross-check ${values} / ${k}`,
    );
  }

  // a tiny modulus makes collisions common; results must still be exact
  for (let t = 0; t < 200; t++) {
    const values = randomArray(nextInt(31), 200);
    const k = 1 + nextInt(3);
    assert.strictEqual(
      describe(findDuplicateWindows(values, k, 257, 97)),
      describe(findDuplicateWindowsNaive(values, k)),
      `tiny modulus cross-check ${values} / ${k}`,
    );
  }

  // concrete collision: the windows (5, 63) and (6, 0) share one hash under mod 97
  assert.strictEqual((5 * 257 + 63) % 97, (6 * 257 + 0) % 97);
  assert.strictEqual(describe(findDuplicateWindows([5, 63, 4, 6, 0], 2, 257, 97)), "");
  assert.strictEqual(describe(findDuplicateWindows([5, 63, 4, 6, 0, 5, 63], 2, 257, 97)), "0:[5]");
  console.log("PASS");
}
