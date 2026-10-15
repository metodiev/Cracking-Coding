/** Task 164 - Compute running median using two heaps. (Tier: Beginner, CTCI Ch 4) */
"use strict";

class Heap {
  constructor(compare) {
    this.data = [];
    this.compare = compare;
  }

  push(value) {
    this.data.push(value);
    let i = this.data.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.compare(this.data[p], this.data[i]) <= 0) break;
      [this.data[p], this.data[i]] = [this.data[i], this.data[p]];
      i = p;
    }
  }

  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length) {
      this.data[0] = last;
      let i = 0;
      while (true) {
        let s = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < this.data.length && this.compare(this.data[l], this.data[s]) < 0) s = l;
        if (r < this.data.length && this.compare(this.data[r], this.data[s]) < 0) s = r;
        if (s === i) break;
        [this.data[i], this.data[s]] = [this.data[s], this.data[i]];
        i = s;
      }
    }
    return top;
  }

  peek() { return this.data[0]; }
  get size() { return this.data.length; }
  get empty() { return this.data.length === 0; }
}

class RunningMedian {
  constructor() {
    this.low = new Heap((a, b) => b - a);   // max-heap
    this.high = new Heap((a, b) => a - b);  // min-heap
  }

  add(value) {
    if (this.low.empty || value <= this.low.peek()) this.low.push(value);
    else this.high.push(value);
    if (this.low.size > this.high.size + 1) this.high.push(this.low.pop());
    else if (this.high.size > this.low.size) this.low.push(this.high.pop());
  }

  median() {
    if (this.low.empty) throw new Error("no values yet");
    if (this.low.size === this.high.size) return (this.low.peek() + this.high.peek()) / 2;
    return this.low.peek();
  }
}

function runningMedians(stream) {
  const rm = new RunningMedian();
  return stream.map((x) => {
    rm.add(x);
    return rm.median();
  });
}

module.exports = { RunningMedian, runningMedians };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(runningMedians([1, 2, 3]), [1.0, 1.5, 2.0]);
  assert.deepStrictEqual(runningMedians([5, 15, 1, 3]), [5.0, 10.0, 5.0, 4.0]);
  assert.deepStrictEqual(runningMedians([-1, -2, -3]), [-1.0, -1.5, -2.0]);
  assert.deepStrictEqual(runningMedians([2, 2, 2]), [2.0, 2.0, 2.0]);
  console.log("PASS");
}
