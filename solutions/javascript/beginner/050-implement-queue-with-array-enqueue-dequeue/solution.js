/** Task 050 - Implement queue with array (enqueue/dequeue). (Tier: Beginner, CTCI Ch 3) */
class QueueArray {
  constructor(capacity) {
    this.data = new Array(capacity);
    this.head = 0;
    this.count = 0;
  }

  enqueue(value) {
    if (this.count === this.data.length) throw new Error("queue full");
    this.data[(this.head + this.count) % this.data.length] = value;
    this.count++;
  }

  dequeue() {
    if (this.isEmpty()) throw new Error("queue empty");
    const value = this.data[this.head];
    this.head = (this.head + 1) % this.data.length;
    this.count--;
    return value;
  }

  front() {
    if (this.isEmpty()) throw new Error("queue empty");
    return this.data[this.head];
  }

  isEmpty() { return this.count === 0; }
  size() { return this.count; }
}

module.exports = { QueueArray };

if (require.main === module) {
  const assert = require("node:assert");
  const q = new QueueArray(3);
  q.enqueue(1); q.enqueue(2); q.enqueue(3);
  assert.strictEqual(q.front(), 1);
  assert.strictEqual(q.dequeue(), 1);
  q.enqueue(4);
  assert.deepStrictEqual([q.dequeue(), q.dequeue(), q.dequeue()], [2, 3, 4]);
  assert.strictEqual(q.isEmpty(), true);
  console.log("PASS");
}
