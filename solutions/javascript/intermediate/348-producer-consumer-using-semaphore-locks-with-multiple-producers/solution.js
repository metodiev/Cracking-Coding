/**
 * Task 348 - Multi-producer/multi-consumer with counting semaphores. (Tier: Intermediate, CTCI Ch 15)
 *
 * Node runs a single-threaded event loop, so this is the documented async simulation (the task 195
 * pattern): a counting-semaphore class keeps suspended acquirers as promise resolvers, `empty`
 * counts free slots and `full` counts queued items.  Buffer mutations happen synchronously between
 * awaits, so put/take are atomic - the event loop serializes them.  Two producers fill disjoint
 * halves of 0..999; after they finish, the main task pushes one -1 sentinel per consumer, so every
 * task terminates deterministically with no timing guesses.
 */
"use strict";

class Semaphore {
  constructor(permits) {
    this.permits = permits;
    this.waiters = [];
  }

  async acquire() {
    while (this.permits === 0) {
      await new Promise((resolve) => this.waiters.push(resolve));
    }
    this.permits--;
  }

  release() {
    this.permits++;
    const waiters = this.waiters;
    this.waiters = [];
    for (const resolve of waiters) resolve();
  }
}

class BoundedBuffer {
  constructor(capacity) {
    this.capacity = capacity;
    this.empty = new Semaphore(capacity);
    this.full = new Semaphore(0);
    this.items = [];
    this.consumed = [];                  // global consumption order
    this.maxSeen = 0;
  }

  async put(item) {
    await this.empty.acquire();
    this.items.push(item);               // atomic: no await between acquire and push
    this.maxSeen = Math.max(this.maxSeen, this.items.length);
    this.full.release();
  }

  async take(sentinel) {
    await this.full.acquire();
    const item = this.items.shift();     // atomic: no await between acquire and shift
    this.empty.release();
    if (item === sentinel) return null;
    this.consumed.push(item);
    return item;
  }
}

module.exports = { Semaphore, BoundedBuffer };

if (require.main === module) {
  const main = async () => {
    const assert = require("node:assert");

    const ITEMS = 1000;
    const SENTINEL = -1;
    const buffer = new BoundedBuffer(7);

    const produce = async (first, last) => {
      for (let item = first; item < last; item++) await buffer.put(item);
    };
    const consume = async () => {
      for (;;) {
        const item = await buffer.take(SENTINEL);
        if (item === null) break;
      }
    };

    const producers = [produce(0, ITEMS / 2), produce(ITEMS / 2, ITEMS)];
    const consumers = [consume(), consume()];

    await Promise.all(producers);
    await buffer.put(SENTINEL);          // one sentinel per consumer, after all real items
    await buffer.put(SENTINEL);
    await Promise.all(consumers);

    const ordered = buffer.consumed;
    assert.strictEqual(ordered.length, ITEMS, "no item lost or duplicated");
    const sorted = [...ordered].sort((a, b) => a - b);
    assert.deepStrictEqual(sorted, Array.from({ length: ITEMS }, (_, i) => i), "every item consumed exactly once");
    assert.ok(buffer.maxSeen <= 7, "capacity never exceeded");
    assert.ok(buffer.maxSeen >= 1, "buffer was actually used");
    assert.deepStrictEqual(ordered.filter((v) => v < ITEMS / 2), Array.from({ length: ITEMS / 2 }, (_, i) => i),
      "producer A order preserved");
    assert.deepStrictEqual(ordered.filter((v) => v >= ITEMS / 2), Array.from({ length: ITEMS / 2 }, (_, i) => i + ITEMS / 2),
      "producer B order preserved");
  };

  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
