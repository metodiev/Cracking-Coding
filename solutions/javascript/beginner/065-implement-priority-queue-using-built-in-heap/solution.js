/** Task 065 - Implement priority queue using built-in heap. (Tier: Beginner, CTCI Ch 3)
 *
 * JavaScript has no built-in binary heap, so here is the minimal array-backed one
 * (the standard "what would PriorityQueue be" answer in interviews).
 */
"use strict";

class PriorityQueue {
  constructor(compare = (a, b) => a - b) {
    this.heap = [];
    this.compare = compare;
    this.counter = 0;           // stable ordering for equal priorities
  }

  push(item, priority) {
    this.heap.push({ priority, counter: this.counter++, item });
    this.siftUp(this.heap.length - 1);
  }

  pop() {
    if (this.heap.length === 0) throw new Error("queue empty");
    const top = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.siftDown(0);
    }
    return top.item;
  }

  peek() {
    if (this.heap.length === 0) throw new Error("queue empty");
    return this.heap[0].item;
  }

  less(a, b) {
    const cmp = this.compare(a.priority, b.priority);
    return cmp !== 0 ? cmp < 0 : a.counter < b.counter;
  }

  siftUp(i) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!this.less(this.heap[i], this.heap[parent])) break;
      [this.heap[i], this.heap[parent]] = [this.heap[parent], this.heap[i]];
      i = parent;
    }
  }

  siftDown(i) {
    const n = this.heap.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.less(this.heap[l], this.heap[smallest])) smallest = l;
      if (r < n && this.less(this.heap[r], this.heap[smallest])) smallest = r;
      if (smallest === i) break;
      [this.heap[i], this.heap[smallest]] = [this.heap[smallest], this.heap[i]];
      i = smallest;
    }
  }

  get size() { return this.heap.length; }
}

module.exports = { PriorityQueue };

if (require.main === module) {
  const assert = require("node:assert");
  const pq = new PriorityQueue();
  pq.push("low", 5);
  pq.push("high", 1);
  pq.push("mid", 3);
  assert.strictEqual(pq.peek(), "high");
  assert.deepStrictEqual([pq.pop(), pq.pop(), pq.pop()], ["high", "mid", "low"]);

  const maxPq = new PriorityQueue((a, b) => b - a);   // max-priority
  maxPq.push("a", 2);
  maxPq.push("b", 9);
  assert.strictEqual(maxPq.pop(), "b");
  console.log("PASS");
}
