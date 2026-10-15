/**
 * Task 347 - Read-write lock with writer preference (no writer starvation). (Tier: Intermediate, CTCI Ch 15)
 *
 * Policy: once a writer waits, newly arriving readers block; waiting writers are served when the
 * current readers/writer drain, so readers cannot starve writers.
 *
 * Node runs a single-threaded event loop - there are no shared-memory locks here.  This is the
 * canonical async simulation used in this repo (see task 195): blocked callers are stored as
 * promise resolvers standing in for condition variables, and grants happen synchronously inside
 * acquire/release, so the scripted interleaving below is exact - no assertion depends on timing.
 */
"use strict";

class ReadWriteLock {
  constructor() {
    this.readers = 0;
    this.writer = false;
    this.writersWaiting = 0;
    this.waiters = [];
    this.grantLog = [];
    this.violations = 0;
    this.maxReaders = 0;
  }

  acquireRead(tag = "") {
    if (!this.writer && this.writersWaiting === 0) {
      this.grant("r", tag);
      return Promise.resolve();
    }
    return new Promise((resolve) => this.waiters.push({ kind: "r", tag, resolve }));
  }

  acquireWrite(tag = "") {
    this.writersWaiting++;
    if (!this.writer && this.readers === 0) {
      this.writersWaiting--;
      this.grant("w", tag);
      return Promise.resolve();
    }
    return new Promise((resolve) => this.waiters.push({ kind: "w", tag, resolve }));
  }

  releaseRead() {
    if (this.readers === 0) {
      this.violations++;                 // release without a matching acquire
      return;
    }
    this.readers--;
    if (this.readers === 0) this.wakeAll();
  }

  releaseWrite() {
    if (!this.writer) {
      this.violations++;
      return;
    }
    this.writer = false;
    this.wakeAll();
  }

  wakeAll() {
    const pending = this.waiters;
    this.waiters = [];
    for (const waiter of pending) {
      if (!this.tryGrant(waiter)) this.waiters.push(waiter);
    }
  }

  tryGrant(waiter) {
    if (waiter.kind === "w") {
      if (this.writer || this.readers > 0) return false;
      this.writersWaiting--;
      this.grant("w", waiter.tag);
      waiter.resolve();
      return true;
    }
    if (this.writer || this.writersWaiting > 0) return false;
    this.grant("r", waiter.tag);
    waiter.resolve();
    return true;
  }

  grant(kind, tag) {
    if (kind === "w") {
      if (this.writer || this.readers > 0) this.violations++;   // must never happen
      this.writer = true;
    } else {
      if (this.writer) this.violations++;                        // must never happen
      this.readers++;
      this.maxReaders = Math.max(this.maxReaders, this.readers);
    }
    this.grantLog.push(`${kind === "w" ? "W" : "R"}:${tag}`);
  }
}

module.exports = { ReadWriteLock };

if (require.main === module) {
  const main = async () => {
    const assert = require("node:assert");
    const flush = async () => {                  // drain the microtask queue deterministically
      for (let i = 0; i < 8; i++) await Promise.resolve();
    };

    const rw = new ReadWriteLock();
    await rw.acquireRead("R1");
    await rw.acquireRead("R2");
    assert.strictEqual(rw.readers, 2);
    assert.deepStrictEqual(rw.grantLog, ["R:R1", "R:R2"], "two readers share the lock");

    const w1 = rw.acquireWrite("W1");            // blocked: readers hold the lock
    const r3 = rw.acquireRead("R3");             // blocked: writer preference
    await flush();
    assert.strictEqual(rw.writersWaiting, 1);
    assert.strictEqual(rw.writer, false);
    assert.strictEqual(rw.readers, 2);
    assert.deepStrictEqual(rw.grantLog, ["R:R1", "R:R2"], "W1 and R3 are still waiting");

    rw.releaseRead();                            // R1
    await flush();
    assert.deepStrictEqual(rw.grantLog, ["R:R1", "R:R2"], "one reader left, nothing granted yet");
    assert.strictEqual(rw.writersWaiting, 1);
    rw.releaseRead();                            // R2 -> the waiting writer drains first
    await flush();
    assert.deepStrictEqual(rw.grantLog, ["R:R1", "R:R2", "W:W1"], "W1 inherits the lock");
    assert.strictEqual(rw.writer, true);
    assert.strictEqual(rw.readers, 0);
    assert.ok(!rw.grantLog.includes("R:R3"), "R3 blocked while the writer holds");
    await w1;

    rw.releaseWrite();
    await flush();
    assert.deepStrictEqual(rw.grantLog, ["R:R1", "R:R2", "W:W1", "R:R3"], "R3 finally granted");
    assert.strictEqual(rw.readers, 1);
    await r3;
    rw.releaseRead();
    await flush();

    // phase 2: a writer with no readers acquires immediately; a reader behind it waits
    await rw.acquireWrite("W2");
    const r4 = rw.acquireRead("R4");
    await flush();
    assert.strictEqual(rw.writer, true);
    assert.strictEqual(rw.readers, 0);
    assert.ok(!rw.grantLog.includes("R:R4"), "R4 blocked behind the active writer");
    rw.releaseWrite();
    await flush();
    assert.deepStrictEqual(rw.grantLog, ["R:R1", "R:R2", "W:W1", "R:R3", "W:W2", "R:R4"], "final grant order");
    assert.strictEqual(rw.readers, 1);
    await r4;
    rw.releaseRead();

    assert.strictEqual(rw.violations, 0, "readers and writers never overlapped");
    assert.strictEqual(rw.maxReaders, 2, "two readers shared the lock at peak");
    assert.strictEqual(rw.readers, 0);
    assert.strictEqual(rw.writer, false);
    assert.strictEqual(rw.writersWaiting, 0);
  };

  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
