/** Task 067 - Implement max-heap push/pop. (Tier: Beginner, CTCI Ch 4) */
"use strict";

class MaxHeap {
  constructor() {
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

  peek() {
    if (this.data.length === 0) throw new Error("heap empty");
    return this.data[0];
  }

  siftUp(i) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.data[parent] >= this.data[i]) break;
      [this.data[i], this.data[parent]] = [this.data[parent], this.data[i]];
      i = parent;
    }
  }

  siftDown(i) {
    const n = this.data.length;
    while (true) {
      let largest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.data[l] > this.data[largest]) largest = l;
      if (r < n && this.data[r] > this.data[largest]) largest = r;
      if (largest === i) break;
      [this.data[i], this.data[largest]] = [this.data[largest], this.data[i]];
      i = largest;
    }
  }

  get size() { return this.data.length; }
}

module.exports = { MaxHeap };

if (require.main === module) {
  const assert = require("node:assert");
  const h = new MaxHeap();
  for (const v of [5, 3, 8, 1, 9, 2]) h.push(v);
  assert.strictEqual(h.peek(), 9);
  const drained = [];
  for (let i = 0; i < 6; i++) drained.push(h.pop());
  assert.deepStrictEqual(drained, [9, 8, 5, 3, 2, 1]);
  console.log("PASS");
}
