/**
 * Task 195 - Implement basic producer-consumer using threads (mutex/condition). (Tier: Beginner, CTCI Ch 15)
 *
 * JavaScript in this repo runs single-threaded: there are no shared-memory locks/condition
 * variables here. This is an equivalent async simulation - BoundedBuffer keeps waiters as
 * promise resolvers (standing in for condition.wait/signal): a producer suspends while the
 * buffer is full and a consumer suspends while it is empty. The event loop serializes the
 * take/put operations, so the single queued stream can be asserted exactly.
 */
"use strict";

const ITEMS = 1000;
const CAPACITY = 7;

class BoundedBuffer {
  constructor(capacity) {
    if (capacity <= 0) throw new Error("capacity must be positive");
    this.capacity = capacity;
    this.items = [];
    this.waiters = [];
    this.closed = false;
    this.maxSeen = 0;
  }

  async put(item) {
    while (!this.closed && this.items.length >= this.capacity) await this._wait();
    if (this.closed) return;
    this.items.push(item);
    this.maxSeen = Math.max(this.maxSeen, this.items.length);
    this._notifyAll();
  }

  async take() {
    while (!this.closed && this.items.length === 0) await this._wait();
    if (this.items.length === 0) return null;
    const item = this.items.shift();
    this._notifyAll();
    return item;
  }

  close() {
    this.closed = true;
    this._notifyAll();
  }

  _wait() {
    return new Promise((resolve) => this.waiters.push(resolve));
  }

  _notifyAll() {
    const pending = this.waiters;
    this.waiters = [];
    for (const resolve of pending) resolve();
  }
}

async function main() {
  const assert = require("node:assert");
  const buffer = new BoundedBuffer(CAPACITY);
  const consumed = [];

  const producerA = (async () => {
    for (let i = 0; i < ITEMS / 2; i++) await buffer.put(i);
  })();
  const producerB = (async () => {
    for (let i = ITEMS / 2; i < ITEMS; i++) await buffer.put(i);
  })();
  const consumerA = (async () => {
    while (true) {
      const item = await buffer.take();
      if (item === null) break;
      consumed.push(item);
    }
  })();
  const consumerB = (async () => {
    while (true) {
      const item = await buffer.take();
      if (item === null) break;
      consumed.push(item);
    }
  })();

  await Promise.all([producerA, producerB]);
  buffer.close();
  await Promise.all([consumerA, consumerB]);

  const expected = Array.from({ length: ITEMS }, (_, i) => i);
  assert.strictEqual(consumed.length, ITEMS);                    // no loss/duplication
  assert.deepStrictEqual([...consumed].sort((a, b) => a - b), expected);
  // The event loop interleaves the two producers, but each one's own items keep their order.
  const half = ITEMS / 2;
  assert.deepStrictEqual(consumed.filter((v) => v < half), Array.from({ length: half }, (_, i) => i));
  assert.deepStrictEqual(consumed.filter((v) => v >= half), Array.from({ length: half }, (_, i) => i + half));
  assert.ok(buffer.maxSeen <= CAPACITY, "capacity never exceeded");
  assert.ok(buffer.maxSeen >= 1, "buffer was actually used");
}

module.exports = { BoundedBuffer };

if (require.main === module) {
  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
