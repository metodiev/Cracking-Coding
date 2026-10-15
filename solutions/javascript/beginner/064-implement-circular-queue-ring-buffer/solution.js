/** Task 064 - Implement circular queue (ring buffer). (Tier: Beginner, CTCI Ch 3) */
"use strict";

class CircularQueue {
  constructor(capacity) {
    if (capacity <= 0) throw new Error("capacity must be positive");
    this.data = new Array(capacity).fill(null);
    this.head = 0;
    this.count = 0;
  }

  enqueue(value) {
    if (this.isFull()) throw new Error("queue full");
    this.data[(this.head + this.count) % this.data.length] = value;
    this.count++;
  }

  dequeue() {
    if (this.isEmpty()) throw new Error("queue empty");
    const value = this.data[this.head];
    this.data[this.head] = null;
    this.head = (this.head + 1) % this.data.length;
    this.count--;
    return value;
  }

  front() {
    if (this.isEmpty()) throw new Error("queue empty");
    return this.data[this.head];
  }

  rear() {
    if (this.isEmpty()) throw new Error("queue empty");
    return this.data[(this.head + this.count - 1) % this.data.length];
  }

  isEmpty() { return this.count === 0; }
  isFull() { return this.count === this.data.length; }
  size() { return this.count; }
}

module.exports = { CircularQueue };

if (require.main === module) {
  const assert = require("node:assert");
  const q = new CircularQueue(3);
  assert.strictEqual(q.isEmpty(), true);
  q.enqueue(1); q.enqueue(2); q.enqueue(3);
  assert.strictEqual(q.isFull(), true);
  assert.strictEqual(q.front(), 1);
  assert.strictEqual(q.rear(), 3);
  assert.strictEqual(q.dequeue(), 1);
  q.enqueue(4);
  assert.strictEqual(q.front(), 2);
  assert.strictEqual(q.rear(), 4);
  assert.deepStrictEqual([q.dequeue(), q.dequeue(), q.dequeue()], [2, 3, 4]);
  assert.strictEqual(q.isEmpty(), true);
  console.log("PASS");
}
