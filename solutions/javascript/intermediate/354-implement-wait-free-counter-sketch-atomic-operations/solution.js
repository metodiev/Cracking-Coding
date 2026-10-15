/**
 * Task 354 - Implement wait-free counter sketch (atomic operations). (Tier: Intermediate, CTCI Ch 15)
 *
 * JavaScript runs single-threaded, so this is the documented task-195 simulation: the CAS
 * counter keeps its value inside a holder object, and compareAndSwap is a plain compare +
 * store that is trivially atomic with respect to the event loop - the same "one indivisible
 * instruction" abstraction the threaded languages get from hardware CAS. THREADS virtual
 * coroutines each run INCREMENTS increments, and the total must be exactly
 * THREADS*INCREMENTS. The retry counter is asserted >= 0 and is documented to be 0 here
 * (no preemption inside the synchronous CAS loop); the deterministic afterRead hook forces
 * one stolen value so the retry path is exercised and checked to return exactly 1. Note:
 * a CAS retry loop is lock-free, not wait-free - fetch-and-add is the wait-free primitive.
 */
"use strict";

const THREADS = 8;
const INCREMENTS = 20000;

class CasCounter {
  constructor() {
    this.state = { value: 0 };
    this.casAttempts = 0;
    this.casFailures = 0;
  }

  get() {
    return this.state.value;
  }

  compareAndSwap(expected, newValue) {
    this.casAttempts += 1;
    if (this.state.value === expected) {
      this.state.value = newValue;
      return true;
    }
    this.casFailures += 1;
    return false;
  }

  increment(afterRead = null) {
    let retries = 0;
    let hook = afterRead;
    for (;;) {
      const current = this.state.value;
      if (hook) {
        const once = hook;
        hook = null;
        once(current);
      }
      if (this.compareAndSwap(current, current + 1)) return retries;
      retries += 1;
    }
  }
}

async function main() {
  const assert = require("node:assert");

  // Single-threaded baseline: no contention, no retries.
  const counter = new CasCounter();
  for (let i = 0; i < 100; i++) {
    assert.strictEqual(counter.increment(), 0, "uncontended CAS never retries");
  }
  assert.strictEqual(counter.get(), 100, "baseline total");
  assert.strictEqual(counter.casAttempts, 100, "baseline attempts");
  assert.strictEqual(counter.casFailures, 0, "baseline failures");

  // Deterministic retry-path demonstration: a concurrent writer bumps the value after our
  // read, so the first CAS fails (retries == 1) and the loop rereads and succeeds.
  const forced = new CasCounter();
  const stolenRetries = forced.increment(
    (current) => forced.compareAndSwap(current, current + 1));
  assert.strictEqual(stolenRetries, 1, "stale read forces exactly one retry");
  assert.strictEqual(forced.get(), 2, "both the stealing write and our retry are counted");

  // Simulated concurrency: 8 virtual coroutines run INCREMENTS increments each, yielding
  // between batches so the event loop interleaves them; no increment may be lost.
  const shared = new CasCounter();
  const retriesPerThread = new Array(THREADS).fill(0);
  const workers = [];
  for (let threadIndex = 0; threadIndex < THREADS; threadIndex++) {
    workers.push((async () => {
      let retries = 0;
      for (let i = 0; i < INCREMENTS; i++) {
        retries += shared.increment();
        if (i % 64 === 63) await Promise.resolve();       // interleave with other workers
      }
      retriesPerThread[threadIndex] = retries;
    })());
  }
  await Promise.all(workers);

  assert.strictEqual(shared.get(), THREADS * INCREMENTS, "no increment lost");
  let totalRetries = 0;
  for (const retries of retriesPerThread) {
    assert.ok(retries >= 0, "retry counters are non-negative");
    totalRetries += retries;
  }
  // Exact CAS accounting: every retry corresponds to exactly one failed CAS.
  assert.strictEqual(totalRetries, shared.casFailures, "retries equal recorded CAS failures");
  assert.strictEqual(shared.casAttempts, THREADS * INCREMENTS + shared.casFailures,
    "attempts == increments + failures");
}

module.exports = { CasCounter };

if (require.main === module) {
  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
