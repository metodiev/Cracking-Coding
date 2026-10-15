/**
 * Task 359 - Readers-writers fairness implementation and tests. (Tier: Intermediate, CTCI Ch 15)
 *
 * JavaScript runs single-threaded, so this is the documented task-195 async simulation:
 * waiters are promise resolvers standing in for a condition variable. The fairness policy
 * is WRITER PREFERENCE - a reader may enter only when no writer is active and no writer is
 * waiting (waitingWriters > 0), so a stream of arriving readers can never starve a writer;
 * a writer enters when no writer is active and all active readers have left. The self-test
 * repeats the scripted ordering scenario (reader A holds, writer W registers, reader B
 * arrives; W must enter before B) and runs a workload of 4 readers x 50 and 3 writers x 50
 * virtual coroutines, asserting that writer entries == writer requests and that every
 * entry/exit kept mutual exclusion.
 */
"use strict";

const READERS = 4;
const WRITERS = 3;
const ITERATIONS = 50;

class AsyncReaderWriter {
  constructor() {
    this.activeReaders = 0;
    this.activeWriter = false;
    this.waitingWriters = 0;
    this.readerEntries = 0;
    this.writerEntries = 0;
    this.maxActiveReaders = 0;
    this.violations = 0;
    this.waiters = [];
  }

  async acquireRead() {
    while (this.activeWriter || this.waitingWriters > 0) await this._wait();
    if (this.activeWriter) this.violations += 1;      // unreachable: writer would overlap
    this.activeReaders += 1;
    this.maxActiveReaders = Math.max(this.maxActiveReaders, this.activeReaders);
    this.readerEntries += 1;
  }

  releaseRead() {
    if (this.activeReaders <= 0 || this.activeWriter) this.violations += 1;
    this.activeReaders -= 1;
    if (this.activeReaders === 0) this._notifyAll();  // last reader lets writers proceed
  }

  async acquireWrite(beforeWait = null) {
    this.waitingWriters += 1;
    if (beforeWait) beforeWait();                     // hook runs with the intent registered
    while (this.activeWriter || this.activeReaders > 0) await this._wait();
    if (this.activeWriter || this.activeReaders > 0) this.violations += 1;
    this.waitingWriters -= 1;
    this.activeWriter = true;
    this.writerEntries += 1;
  }

  releaseWrite() {
    if (!this.activeWriter || this.activeReaders > 0) this.violations += 1;
    this.activeWriter = false;
    this._notifyAll();
  }

  get isClean() {
    return this.activeReaders === 0 && !this.activeWriter;
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

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

async function main() {
  const assert = require("node:assert");

  // Part 1: scripted writer-preference ordering test.
  const rw = new AsyncReaderWriter();
  const log = [];
  const aAcquired = deferred();
  const releaseA = deferred();
  const wRegistered = deferred();
  const wEntered = deferred();
  const releaseW = deferred();
  const bEntered = deferred();
  let bHasEntered = false;

  const readerA = (async () => {
    await rw.acquireRead();
    log.push("readA");
    aAcquired.resolve();
    await releaseA.promise;
    rw.releaseRead();
  })();
  const writerW = (async () => {
    await rw.acquireWrite(() => wRegistered.resolve());
    log.push("writeW");
    wEntered.resolve();
    await releaseW.promise;
    rw.releaseWrite();
  })();
  const readerB = (async () => {
    await rw.acquireRead();
    log.push("readB");
    bHasEntered = true;
    bEntered.resolve();
    rw.releaseRead();
  })();

  await aAcquired.promise;
  await wRegistered.promise;
  releaseA.resolve();
  await wEntered.promise;
  assert.deepStrictEqual(log, ["readA", "writeW"], "writer takes precedence over the new reader");
  assert.strictEqual(bHasEntered, false, "new reader waits while a writer is pending");
  releaseW.resolve();
  await bEntered.promise;
  await Promise.all([readerA, writerW, readerB]);

  assert.deepStrictEqual(log, ["readA", "writeW", "readB"], "writer-preference order");
  assert.strictEqual(rw.writerEntries, 1, "one writer entry");
  assert.strictEqual(rw.readerEntries, 2, "two reader entries");
  assert.strictEqual(rw.violations, 0, "scripted scenario kept mutual exclusion");

  // Part 2 + 3: simulated workload with the fairness accounting.
  const workload = new AsyncReaderWriter();
  let value = 0;                                     // guarded by the write lock
  const observations = [];
  const writerCompletions = new Array(WRITERS).fill(0);
  const readerCompletions = new Array(READERS).fill(0);
  const workers = [];

  for (let readerId = 0; readerId < READERS; readerId++) {
    workers.push((async () => {
      for (let iteration = 0; iteration < ITERATIONS; iteration++) {
        await workload.acquireRead();
        observations.push(value);
        workload.releaseRead();
        readerCompletions[readerId] += 1;
        if (iteration % 8 === 7) await Promise.resolve();   // interleave with the others
      }
    })());
  }
  for (let writerId = 0; writerId < WRITERS; writerId++) {
    workers.push((async () => {
      for (let iteration = 0; iteration < ITERATIONS; iteration++) {
        await workload.acquireWrite();
        value += 1;
        workload.releaseWrite();
        writerCompletions[writerId] += 1;
        if (iteration % 8 === 7) await Promise.resolve();
      }
    })());
  }
  await Promise.all(workers);

  assert.strictEqual(writerCompletions.reduce((a, b) => a + b, 0), WRITERS * ITERATIONS,
    "every writer request was served");
  assert.strictEqual(workload.writerEntries, WRITERS * ITERATIONS,
    "writer entries == writer requests (no starvation)");
  assert.strictEqual(workload.writerEntries, 150, "hand-checked writer entry total");
  assert.strictEqual(workload.readerEntries, READERS * ITERATIONS, "reader entries == reader requests");
  assert.strictEqual(workload.readerEntries, 200, "hand-checked reader entry total");
  assert.strictEqual(value, 150, "all writer increments applied exactly once");
  assert.strictEqual(workload.violations, 0, "mutual exclusion held throughout the workload");
  assert.strictEqual(observations.length, 200, "every read produced an observation");
  assert.ok(observations.every((seen) => seen >= 0 && seen <= 150),
    "readers only saw committed values");
  assert.ok(workload.maxActiveReaders >= 1, "readers actually ran");
  assert.strictEqual(workload.isClean, true, "lock left clean");
}

module.exports = { AsyncReaderWriter };

if (require.main === module) {
  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
