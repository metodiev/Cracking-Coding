/** Task 173 - Implement queue that supports getMin in O(1). (Tier: Beginner, CTCI Ch 3) */
"use strict";

class MinStackInner {
  constructor() {
    this.data = [];
    this.mins = [];
  }

  push(value) {
    this.data.push(value);
    this.mins.push(this.mins.length === 0 ? value : Math.min(value, this.mins[this.mins.length - 1]));
  }

  pop() {
    this.mins.pop();
    return this.data.pop();
  }

  peek() { return this.data[this.data.length - 1]; }
  min() { return this.mins[this.mins.length - 1]; }
  get size() { return this.data.length; }
}

class MinQueue {
  constructor() {
    this.inbox = new MinStackInner();
    this.outbox = new MinStackInner();
  }

  enqueue(value) { this.inbox.push(value); }

  dequeue() {
    if (this.isEmpty) throw new Error("queue empty");
    if (this.outbox.size === 0) {
      while (this.inbox.size > 0) this.outbox.push(this.inbox.pop());
    }
    return this.outbox.pop();
  }

  min() {
    if (this.isEmpty) throw new Error("queue empty");
    if (this.inbox.size === 0) return this.outbox.min();
    if (this.outbox.size === 0) return this.inbox.min();
    return Math.min(this.inbox.min(), this.outbox.min());
  }

  get isEmpty() { return this.inbox.size === 0 && this.outbox.size === 0; }
}

module.exports = { MinQueue };

if (require.main === module) {
  const assert = require("node:assert");
  const q = new MinQueue();
  for (const v of [3, 1, 4, 1, 5]) q.enqueue(v);
  assert.strictEqual(q.min(), 1);
  assert.strictEqual(q.dequeue(), 3);
  assert.strictEqual(q.min(), 1);
  assert.strictEqual(q.dequeue(), 1);
  assert.strictEqual(q.min(), 1);
  assert.strictEqual(q.dequeue(), 4);
  assert.strictEqual(q.min(), 1);
  assert.strictEqual(q.dequeue(), 1);
  assert.strictEqual(q.min(), 5);
  q.enqueue(0);
  assert.strictEqual(q.min(), 0);
  assert.strictEqual(q.dequeue(), 5);
  assert.strictEqual(q.dequeue(), 0);
  assert.strictEqual(q.isEmpty, true);
  console.log("PASS");
}
