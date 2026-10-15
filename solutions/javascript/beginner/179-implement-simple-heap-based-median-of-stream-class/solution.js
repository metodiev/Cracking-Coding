/** Task 179 - Implement simple heap-based median-of-stream class. (Tier: Beginner, CTCI Ch 4) */
"use strict";

class IntHeap {
  constructor(maxFirst) {
    this.data = [];
    this.maxFirst = maxFirst;
  }

  better(a, b) { return this.maxFirst ? a > b : a < b; }

  push(value) {
    this.data.push(value);
    let i = this.data.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!this.better(this.data[i], this.data[parent])) break;
      [this.data[i], this.data[parent]] = [this.data[parent], this.data[i]];
      i = parent;
    }
  }

  pop() {
    if (this.data.length === 0) throw new Error("heap empty");
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length) {
      this.data[0] = last;
      let i = 0;
      while (true) {
        let best = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < this.data.length && this.better(this.data[l], this.data[best])) best = l;
        if (r < this.data.length && this.better(this.data[r], this.data[best])) best = r;
        if (best === i) break;
        [this.data[i], this.data[best]] = [this.data[best], this.data[i]];
        i = best;
      }
    }
    return top;
  }

  peek() {
    if (this.data.length === 0) throw new Error("heap empty");
    return this.data[0];
  }

  get size() { return this.data.length; }
}

class MedianOfStream {
  constructor() {
    this.low = new IntHeap(true);    // max-heap
    this.high = new IntHeap(false);  // min-heap
    this.count = 0;
  }

  add(value) {
    this.count++;
    if (this.low.size === 0 || value <= this.low.peek()) this.low.push(value);
    else this.high.push(value);
    if (this.low.size > this.high.size + 1) this.high.push(this.low.pop());
    else if (this.high.size > this.low.size) this.low.push(this.high.pop());
  }

  median() {
    if (this.count === 0) throw new Error("no values yet");
    if (this.low.size === this.high.size) return (this.low.peek() + this.high.peek()) / 2;
    return this.low.peek();
  }
}

module.exports = { IntHeap, MedianOfStream };

if (require.main === module) {
  const assert = require("node:assert");
  const m = new MedianOfStream();
  const values = [1, 2, 3, 4, 5];
  const expected = [1.0, 1.5, 2.0, 2.5, 3.0];
  values.forEach((v, i) => {
    m.add(v);
    assert.strictEqual(m.median(), expected[i]);
  });

  const m2 = new MedianOfStream();
  m2.add(10);
  assert.strictEqual(m2.median(), 10);
  m2.add(-5);
  assert.strictEqual(m2.median(), 2.5);
  m2.add(-5);
  assert.strictEqual(m2.median(), -5);
  console.log("PASS");
}
