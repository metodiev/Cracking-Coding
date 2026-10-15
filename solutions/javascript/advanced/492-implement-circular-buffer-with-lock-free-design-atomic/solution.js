/**
 * Task 492 - Implement circular buffer with lock-free design (atomic). (Tier: Advanced, CTCI Ch 15)
 *
 * Bounded MPMC ring buffer in the Vyukov style: capacity slots each carry a sequence number
 * (slot i starts at i), head and tail are plain counters advanced only by compare-and-swap, and
 * every operation retries when its CAS loses the race. A slot is publishable when seq === pos
 * and consumable when seq === pos + 1; the claimer overwrites the slot and then stores
 * seq = pos + 1 (enqueue) or seq = pos + capacity (dequeue), which re-opens the slot one lap
 * later. One extra occupancy check (tail - head >= capacity => full) is required because for
 * capacity 1 the final lap's free and published sequence values coincide. Node runs one thread
 * per process, so - exactly as in task 354 - concurrency is the documented simulation: the CAS
 * is a plain compare + store inside a holder object that cannot be preempted by the event loop
 * (the same "one indivisible instruction" abstraction the threaded languages get from
 * hardware), producers and consumers are virtual coroutines that yield between operations, and
 * the retry path is exercised deterministically by a hook that claims the position just before
 * the outer CAS, forcing exactly one failed CAS and one retry lap. The tests assert:
 * single-threaded wraparound/full/empty semantics, exact global FIFO for one producer / one
 * consumer, and exactly-once delivery plus per-producer FIFO for many producers / consumers -
 * with the claimed dequeue positions acting as the linearization witnesses.
 */
"use strict";

const FULL = false;
const EMPTY = null;

class LockFreeCircularBuffer {
  constructor(capacity, casHook = null) {
    if (capacity <= 0) throw new Error("capacity must be positive");
    this.capacity = capacity;
    this.slots = new Array(capacity).fill(null);
    this.seq = [];
    for (let i = 0; i < capacity; i++) this.seq.push(i);   // slot i is free at position i
    this.head = 0;                                         // position of the next dequeue
    this.tail = 0;                                         // position of the next enqueue
    this.casAttempts = 0;
    this.casFailures = 0;
    this.enqueueRetries = 0;
    this.dequeueRetries = 0;
    this.casHook = casHook;                                // fires once, just before a CAS
  }

  cas(holder, name, expected, newValue) {
    this.casAttempts += 1;
    if (holder[name] === expected) {
      holder[name] = newValue;
      return true;
    }
    this.casFailures += 1;
    return false;
  }

  runHook(pos) {
    const hook = this.casHook;
    this.casHook = null;
    if (hook !== null) hook(this, pos);
  }

  /** Returns true when value was enqueued, false when the queue is full. */
  tryEnqueue(value) {
    for (;;) {
      const pos = this.tail;
      if (pos - this.head >= this.capacity) {
        return FULL;                                     // occupancy guard (needed for cap 1)
      }
      const slotIndex = pos % this.capacity;
      const slotSeq = this.seq[slotIndex];
      if (slotSeq === pos) {
        this.runHook(pos);
        if (this.cas(this, "tail", pos, pos + 1)) {
          this.slots[slotIndex] = value;
          this.seq[slotIndex] = pos + 1;                 // publish (release store)
          return true;
        }
        // CAS lost: another producer claimed pos; reread the tail and retry.
      } else if (slotSeq < pos) {
        return FULL;                                     // the previous lap has not caught up
      }
      this.enqueueRetries += 1;
    }
  }

  /** Returns the oldest value, or EMPTY (null) when the queue is empty. onClaim, when given,
   * receives the claimed position right after the head CAS as a linearization witness. */
  tryDequeue(onClaim = null) {
    for (;;) {
      const pos = this.head;
      const slotIndex = pos % this.capacity;
      const slotSeq = this.seq[slotIndex];
      if (slotSeq === pos + 1) {
        if (this.cas(this, "head", pos, pos + 1)) {
          if (onClaim !== null) onClaim(pos);
          const value = this.slots[slotIndex];
          this.seq[slotIndex] = pos + this.capacity;     // reopen one lap later
          return value;
        }
        // CAS lost: another consumer took pos; reread the head and retry.
      } else if (slotSeq < pos + 1) {
        return EMPTY;
      }
      this.dequeueRetries += 1;
    }
  }
}

async function main() {
  const assert = require("node:assert");

  // Single-threaded semantics: wraparound, full, empty, capacity 1.
  const buf = new LockFreeCircularBuffer(4);
  for (const value of [1, 2, 3, 4]) assert.strictEqual(buf.tryEnqueue(value), true);
  assert.strictEqual(buf.tryEnqueue(5), FULL);
  assert.deepStrictEqual([buf.tryDequeue(), buf.tryDequeue(), buf.tryDequeue(), buf.tryDequeue()],
    [1, 2, 3, 4]);
  assert.strictEqual(buf.tryDequeue(), EMPTY);
  assert.strictEqual(buf.tryEnqueue(5), true);
  assert.strictEqual(buf.tryEnqueue(6), true);
  assert.strictEqual(buf.tryDequeue(), 5);
  assert.strictEqual(buf.tryDequeue(), 6);               // positions wrapped twice
  assert.strictEqual(buf.tryDequeue(), EMPTY);

  const single = new LockFreeCircularBuffer(1);
  assert.strictEqual(single.tryEnqueue("a"), true);
  assert.strictEqual(single.tryEnqueue("b"), FULL);
  assert.strictEqual(single.tryDequeue(), "a");
  assert.strictEqual(single.tryEnqueue("b"), true);
  assert.strictEqual(single.tryDequeue(), "b");

  // Deterministic retry-path demonstration: the hook, running just before the outer CAS, lets
  // a "concurrent" producer claim the same position; the outer CAS must fail exactly once,
  // reread the tail and succeed on the next lap, with the hook's item first in FIFO order.
  const forced = new LockFreeCircularBuffer(8, (buffer, pos) => {
    assert.strictEqual(buffer.tryEnqueue("stealer"), true);
    assert.strictEqual(buffer.head, 0);
    assert.strictEqual(buffer.tail, pos + 1);
  });
  assert.strictEqual(forced.tryEnqueue("mine"), true);
  assert.strictEqual(forced.casFailures, 1, "the stale CAS must fail exactly once");
  assert.strictEqual(forced.enqueueRetries, 1, "exactly one retry lap");
  assert.strictEqual(forced.tryDequeue(), "stealer");
  assert.strictEqual(forced.tryDequeue(), "mine");
  assert.strictEqual(forced.tryDequeue(), EMPTY);

  // SPSC virtual concurrency: bounded capacity forces the producer to yield on FULL and the
  // consumer to yield on EMPTY; the dequeued stream must still be exactly 0, 1, ..., total-1.
  const total = 5000;
  const spsc = new LockFreeCircularBuffer(8);
  const received = [];
  const producer = (async () => {
    for (let value = 0; value < total; value++) {
      while (spsc.tryEnqueue(value) === FULL) await Promise.resolve();
      if (value % 64 === 63) await Promise.resolve();     // interleave with the consumer
    }
  })();
  const consumer = (async () => {
    while (received.length < total) {
      const value = spsc.tryDequeue();
      if (value !== EMPTY) {
        received.push(value);
      } else {
        await Promise.resolve();
      }
    }
  })();
  await Promise.all([producer, consumer]);
  assert.deepStrictEqual(received, Array.from({ length: total }, (_, i) => i), "SPSC FIFO");

  // MPMC accounting: 4 producers x 150 tagged items, 3 consumers, capacity 16. Consumers
  // record the position claimed by the head CAS, so the merged record must reproduce the
  // dequeue order: positions 0..total-1 exactly once, and each producer's own items in the
  // order that producer enqueued them.
  const producers = 4;
  const perProducer = 150;
  const consumers = 3;
  const mpmc = new LockFreeCircularBuffer(16);
  const records = [[], [], []];
  let tally = 0;

  const mpWorkers = [];
  for (let p = 0; p < producers; p++) {
    mpWorkers.push((async () => {
      for (let i = 0; i < perProducer; i++) {
        while (mpmc.tryEnqueue([p, i]) === FULL) await Promise.resolve();
        if (i % 32 === 31) await Promise.resolve();
      }
    })());
  }
  const mcWorkers = [];
  for (let index = 0; index < consumers; index++) {
    mcWorkers.push((async () => {
      const local = records[index];
      while (tally < producers * perProducer) {
        let claimed = -1;
        const value = mpmc.tryDequeue((pos) => { claimed = pos; });
        if (value === EMPTY) {
          await Promise.resolve();
          continue;
        }
        local.push([claimed, value]);
        tally += 1;
      }
    })());
  }
  await Promise.all(mpWorkers.concat(mcWorkers));

  assert.strictEqual(tally, producers * perProducer, "exactly-once delivery");
  const merged = records.flat().sort((a, b) => a[0] - b[0]);
  assert.deepStrictEqual(merged.map((record) => record[0]),
    Array.from({ length: producers * perProducer }, (_, i) => i),
    "claimed positions must be a permutation of 0..total-1");
  const order = new Map();
  for (let p = 0; p < producers; p++) order.set(p, []);
  for (const [, [p, i]] of merged) order.get(p).push(i);
  for (let p = 0; p < producers; p++) {
    assert.deepStrictEqual(order.get(p), Array.from({ length: perProducer }, (_, i) => i),
      "per-producer FIFO violated");
  }
  assert.ok(mpmc.casAttempts >= producers * perProducer, "every operation used the CAS path");
}

module.exports = { FULL, EMPTY, LockFreeCircularBuffer };

if (require.main === module) {
  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
