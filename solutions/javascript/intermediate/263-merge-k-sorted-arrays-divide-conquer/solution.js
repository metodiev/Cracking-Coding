/** Task 263 - Merge K sorted arrays (divide & conquer). (Tier: Intermediate, CTCI Ch 10) */
"use strict";

/** Small binary min-heap driven by a comparator (used only for the heap cross-check). */
class MinHeap {
  constructor(cmp) {
    this.cmp = cmp;
    this.data = [];
  }

  push(value) {
    this.data.push(value);
    this.siftUp(this.data.length - 1);
  }

  pop() {
    if (this.data.length === 0) throw new Error("heap empty");
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) {
      this.data[0] = last;
      this.siftDown(0);
    }
    return top;
  }

  siftUp(i) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.cmp(this.data[parent], this.data[i]) <= 0) break;
      [this.data[i], this.data[parent]] = [this.data[parent], this.data[i]];
      i = parent;
    }
  }

  siftDown(i) {
    const n = this.data.length;
    for (;;) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.cmp(this.data[l], this.data[smallest]) < 0) smallest = l;
      if (r < n && this.cmp(this.data[r], this.data[smallest]) < 0) smallest = r;
      if (smallest === i) break;
      [this.data[i], this.data[smallest]] = [this.data[smallest], this.data[i]];
      i = smallest;
    }
  }

  get size() { return this.data.length; }
}

function mergeTwo(a, b) {
  const out = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) out.push(a[i] <= b[j] ? a[i++] : b[j++]);
  while (i < a.length) out.push(a[i++]);
  while (j < b.length) out.push(b[j++]);
  return out;
}

/**
 * Pairwise divide-and-conquer merge with a plain 2-way two-pointer merge (NO heap):
 * log k levels of O(n) work -> O(n log k). k = 0 -> []; k = 1 returns a fresh copy;
 * the input arrays are never modified.
 */
function mergeKDivide(arrays) {
  const divide = (lo, hi) => {
    if (hi - lo === 1) return arrays[lo].slice();
    const mid = (lo + hi) >> 1;
    return mergeTwo(divide(lo, mid), divide(mid, hi));
  };
  return arrays.length === 0 ? [] : divide(0, arrays.length);
}

/** Independent heap-based merge used only for cross-checks (same result expected). */
function mergeKHeap(arrays) {
  const heap = new MinHeap((a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]));
  arrays.forEach((arr, idx) => { if (arr.length > 0) heap.push([arr[0], idx, 0]); });
  const out = [];
  while (heap.size > 0) {
    const [val, idx, pos] = heap.pop();
    out.push(val);
    if (pos + 1 < arrays[idx].length) heap.push([arrays[idx][pos + 1], idx, pos + 1]);
  }
  return out;
}

/** Flat sort reference used only for cross-checks. */
function mergeOracle(arrays) {
  const values = [];
  for (const arr of arrays) values.push(...arr);
  values.sort((a, b) => a - b);
  return values;
}

module.exports = { MinHeap, mergeTwo, mergeKDivide, mergeKHeap, mergeOracle };

if (require.main === module) {
  const assert = require("node:assert");
  const classic = [[1, 4, 5], [1, 3, 4], [2, 6]];
  assert.deepStrictEqual(mergeKDivide(classic), [1, 1, 2, 3, 4, 4, 5, 6]);
  assert.deepStrictEqual(mergeKDivide([]), []);                            // k = 0
  assert.deepStrictEqual(mergeKDivide([[3, 4]]), [3, 4]);                  // k = 1
  assert.deepStrictEqual(mergeKDivide([[]]), []);
  assert.deepStrictEqual(mergeKDivide([[1], [2, 3, 4, 5], [6]]), [1, 2, 3, 4, 5, 6]);   // unequal lengths
  assert.deepStrictEqual(mergeKDivide([[2, 2], [2, 2], [2]]), [2, 2, 2, 2, 2]);         // duplicates
  assert.deepStrictEqual(mergeKDivide([[], [1], [], [2, 3]]), [1, 2, 3]);               // empties inside
  assert.deepStrictEqual(mergeKDivide([[-3, -1], [-2, 0]]), [-3, -2, -1, 0]);

  const original = [3, 4];
  const copy = mergeKDivide([original]);
  assert.deepStrictEqual(copy, original);
  assert.notStrictEqual(copy, original, "k = 1 returns a fresh copy");

  const fixed = [classic, [], [[3, 4]], [[]], [[1], [2, 3, 4, 5], [6]], [[2, 2], [2, 2], [2]], [[-3, -1], [-2, 0]]];
  for (const arrays of fixed) {              // divide & conquer == heap == flat sort
    assert.deepStrictEqual(mergeKDivide(arrays), mergeKHeap(arrays), `divide == heap: ${JSON.stringify(arrays)}`);
    assert.deepStrictEqual(mergeKDivide(arrays), mergeOracle(arrays), `divide == oracle: ${JSON.stringify(arrays)}`);
  }

  // deterministic LCG so the cross-check is reproducible
  let seed = 263;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  for (let test = 0; test < 300; test++) {
    const k = randInt(0, 6);
    const arrays = Array.from({ length: k }, () =>
      Array.from({ length: randInt(0, 6) }, () => randInt(-6, 6)).sort((a, b) => a - b));
    assert.deepStrictEqual(mergeKDivide(arrays), mergeKHeap(arrays), `random divide == heap #${test}`);
    assert.deepStrictEqual(mergeKDivide(arrays), mergeOracle(arrays), `random divide == oracle #${test}`);
  }
  console.log("PASS");
}
