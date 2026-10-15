/** Task 256 - Top K frequent elements in array (heap). (Tier: Intermediate, CTCI Ch 10) */
"use strict";

/** Tiny binary heap; compare(a, b) < 0 means a ranks nearer the root. */
class MinHeap {
  constructor(compare) {
    this.data = [];
    this.compare = compare;
  }

  get size() {
    return this.data.length;
  }

  peek() {
    return this.data[0];
  }

  push(value) {
    this.data.push(value);
    let i = this.data.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compare(this.data[parent], this.data[i]) <= 0) break;
      [this.data[parent], this.data[i]] = [this.data[i], this.data[parent]];
      i = parent;
    }
  }

  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) {
      this.data[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        let best = i;
        if (l < this.data.length && this.compare(this.data[l], this.data[best]) < 0) best = l;
        if (r < this.data.length && this.compare(this.data[r], this.data[best]) < 0) best = r;
        if (best === i) break;
        [this.data[best], this.data[i]] = [this.data[i], this.data[best]];
        i = best;
      }
    }
    return top;
  }
}

/**
 * Count with a Map, then keep the k best in a min-heap whose root is the *worst* kept
 * entry. Documented tie-break: worse = lower count, or on a count tie the larger value;
 * the output is frequency-descending with value-ascending tie-breaks. O(n + d log k).
 */
function topKFrequent(nums, k) {
  if (k <= 0) return [];
  const freq = new Map();
  for (const x of nums) freq.set(x, (freq.get(x) || 0) + 1);
  const heap = new MinHeap((a, b) => (a.count !== b.count ? a.count - b.count : b.value - a.value));
  for (const [value, count] of freq) {
    const entry = { count, value };
    if (heap.size < k) {
      heap.push(entry);
    } else if (isBetter(entry, heap.peek())) {
      heap.pop();
      heap.push(entry);
    }
  }
  return heap.data
    .slice()
    .sort((a, b) => (a.count !== b.count ? b.count - a.count : a.value - b.value))
    .map((e) => e.value);
}

function isBetter(candidate, worst) {
  if (candidate.count !== worst.count) return candidate.count > worst.count;
  return candidate.value < worst.value;
}

module.exports = { MinHeap, topKFrequent };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(topKFrequent([1, 1, 1, 2, 2, 3], 2), [1, 2]);
  assert.deepStrictEqual(topKFrequent([1, 1, 1, 2, 2, 3], 3), [1, 2, 3]);
  assert.deepStrictEqual(topKFrequent([1, 1, 1, 2, 2, 3], 6), [1, 2, 3]);   // k >= distinct
  assert.deepStrictEqual(topKFrequent([1, 1, 1, 2, 2, 3], 99), [1, 2, 3]);
  assert.deepStrictEqual(topKFrequent([1, 1, 1, 2, 2, 3], 0), []);
  assert.deepStrictEqual(topKFrequent([1, 1, 1, 2, 2, 3], 1), [1]);

  assert.deepStrictEqual(topKFrequent([4, 4, 5, 5, 6], 2), [4, 5]);         // count tie
  assert.deepStrictEqual(topKFrequent([3, 1, 2], 2), [1, 2]);               // all count 1
  assert.deepStrictEqual(topKFrequent([2, 2, 3, 3, 1], 1), [2]);            // eviction tie
  assert.deepStrictEqual(topKFrequent([-1, -1, 2, 2], 2), [-1, 2]);
  assert.deepStrictEqual(topKFrequent([5], 1), [5]);
  assert.deepStrictEqual(topKFrequent([], 3), []);
  assert.deepStrictEqual(topKFrequent([7, 7, 7], 3), [7]);
  console.log("PASS");
}
