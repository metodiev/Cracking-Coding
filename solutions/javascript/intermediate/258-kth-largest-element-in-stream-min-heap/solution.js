/** Task 258 - Kth largest element in stream (min-heap). (Tier: Intermediate, CTCI Ch 10) */
"use strict";

/** Min-heap of numbers ("worst" = smallest at the root). */
class MinHeap {
  constructor() {
    this.data = [];
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
      if (this.data[parent] <= this.data[i]) break;
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
        if (l < this.data.length && this.data[l] < this.data[best]) best = l;
        if (r < this.data.length && this.data[r] < this.data[best]) best = r;
        if (best === i) break;
        [this.data[best], this.data[i]] = [this.data[i], this.data[best]];
        i = best;
      }
    }
    return top;
  }
}

/**
 * Keeps just the k largest values seen, so heap.peek() is the k-th largest. add() returns
 * null until at least k values have been seen (documented behaviour when k is larger than
 * the initial count). Each add is O(log k).
 */
class KthLargest {
  constructor(k, nums) {
    if (k < 1) throw new Error("k must be >= 1");
    this.k = k;
    this.heap = new MinHeap();
    for (const v of nums) this.heap.push(v);
    while (this.heap.size > k) this.heap.pop();
  }

  add(val) {
    this.heap.push(val);
    if (this.heap.size > this.k) this.heap.pop();
    return this.heap.size < this.k ? null : this.heap.peek();
  }
}

module.exports = { MinHeap, KthLargest };

if (require.main === module) {
  const assert = require("node:assert");

  const kl = new KthLargest(3, [4, 5, 8, 2]);
  assert.strictEqual(kl.add(3), 4);
  assert.strictEqual(kl.add(5), 5);
  assert.strictEqual(kl.add(10), 5);
  assert.strictEqual(kl.add(9), 8);
  assert.strictEqual(kl.add(4), 8);

  const grow = new KthLargest(5, [3, 1]);                 // k larger than the initial count
  assert.strictEqual(grow.add(2), null);
  assert.strictEqual(grow.add(7), null);
  assert.strictEqual(grow.add(9), 1);                     // {1,2,3,7,9}: 5th largest is 1
  assert.strictEqual(grow.add(4), 2);                     // {2,3,4,7,9}: 2

  const top = new KthLargest(1, [4, 5, 8, 2]);            // k = 1: running maximum
  assert.strictEqual(top.add(3), 8);
  assert.strictEqual(top.add(10), 10);
  assert.strictEqual(top.add(4), 10);

  const dupes = new KthLargest(2, [5, 5, 5]);
  assert.strictEqual(dupes.add(5), 5);
  assert.strictEqual(dupes.add(4), 5);
  assert.strictEqual(dupes.add(6), 5);
  assert.strictEqual(dupes.add(7), 6);

  const empty = new KthLargest(2, []);
  assert.strictEqual(empty.add(5), null);
  assert.strictEqual(empty.add(3), 3);
  assert.strictEqual(empty.add(10), 5);
  assert.strictEqual(empty.add(8), 8);

  const start = new KthLargest(2, [1, 2, 3]);             // initial pile trimmed to top k
  assert.strictEqual(start.add(0), 2);
  console.log("PASS");
}
