/** Task 615 - Implement transactional memory simulator basics. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * A single-threaded, deterministic software transactional memory: deferred update with commit
 * time validation (the "TL2" idea). reading a location logs the value together with the memory
 * version stamp that was current when it was read; writes only land in the transaction's write
 * set; commit re-checks every logged stamp against the live memory and then installs the whole
 * write set under a new global version, so a commit is atomic and an abort leaves no trace.
 * A conflicting transaction therefore aborts and has to be replayed - which is exactly the
 * behaviour the simulator demonstrates.
 *
 * How interleaving is modelled: there are no threads. A *script* is a flat list of steps
 * {txn, op, ...} that the scheduler executes in order; a "commit" step that validates fails
 * re-queues that transaction's steps at the tail of the schedule with a fresh transaction
 * object, which is what a real runtime does when it retries an aborted transaction. Two
 * transactions with the same address in their read/write sets therefore conflict whenever the
 * script interleaves them in the wrong order, and the simulator detects, retries and reports it.
 *
 * What is simplified: no real concurrency (so no CAS, no contention manager, no invalidation
 * broadcasts, no nested transactions or durability), and addresses are the simulator's whole
 * memory. Verified: sequential scripts produce exactly the sequential result; write-write and
 * read-write conflicts abort and retry until they commit; a transaction's writes stay invisible
 * until it commits; an aborted transaction leaves values, stamps and commit counts untouched;
 * read-only transactions commit without bumping the version and still validate their reads;
 * livelock is bounded by the attempt limit instead of hanging; and repeated runs of one script
 * produce identical logs.
 */
"use strict";

class TxnMemory {
  constructor(size) {
    if (!Number.isInteger(size) || size <= 0) throw new RangeError("memory size must be positive");
    this.values = new Array(size).fill(0);
    this.stamps = new Array(size).fill(0);
    this.version = 0;
    this.commits = 0;
    this.aborts = 0;
    this.installedWrites = 0;
  }

  get size() { return this.values.length; }

  #check(addr) {
    if (!Number.isInteger(addr) || addr < 0 || addr >= this.values.length) {
      throw new RangeError(`address ${addr} out of range`);
    }
  }

  begin() { return { reads: new Map(), writes: new Map(), finished: false, aborted: false }; }

  /** Reads see the transaction's own writes first, then its read set, then the live memory. */
  read(txn, addr) {
    this.#check(addr);
    this.#checkTxn(txn);
    if (txn.writes.has(addr)) return txn.writes.get(addr);
    const logged = txn.reads.get(addr);
    if (logged !== undefined) return logged.value;
    const entry = { value: this.values[addr], stamp: this.stamps[addr] };
    txn.reads.set(addr, entry);
    return entry.value;
  }

  write(txn, addr, value) {
    this.#check(addr);
    this.#checkTxn(txn);
    if (!Number.isInteger(value)) throw new TypeError("values must be integers");
    txn.writes.set(addr, value);
  }

  #checkTxn(txn) {
    if (typeof txn !== "object" || txn === null || !(txn.reads instanceof Map) || !(txn.writes instanceof Map)) {
      throw new TypeError("not a transaction");
    }
    if (txn.finished) throw new Error("transaction is already finished");
  }

  /** Validate then install; returns "committed" (stamps bumped) or "aborted" (nothing changed). */
  commit(txn) {
    this.#checkTxn(txn);
    for (const [addr, entry] of txn.reads) {
      if (this.stamps[addr] !== entry.stamp) {
        txn.finished = true;
        txn.aborted = true;
        this.aborts++;
        return "aborted";
      }
    }
    txn.finished = true;
    if (txn.writes.size === 0) {
      this.commits++;
      return "committed";                       // read-only: no version bump, nothing installed
    }
    this.version++;
    for (const [addr, value] of txn.writes) {
      this.values[addr] = value;
      this.stamps[addr] = this.version;
      this.installedWrites++;
    }
    this.commits++;
    return "committed";
  }

  snapshot() {
    return { values: [...this.values], stamps: [...this.stamps], version: this.version };
  }
}

/** Runs an interleaved script, replaying a transaction from scratch whenever its commit aborts. */
function runScript(memory, script, options = {}) {
  const maxAttempts = options.maxAttempts ?? 4;
  const opsByTxn = new Map();
  for (const step of script) {
    if (!opsByTxn.has(step.txn)) opsByTxn.set(step.txn, []);
    opsByTxn.get(step.txn).push(step);
  }
  const queue = script.map((step) => ({ step, attempt: 1 }));
  const live = new Map();
  const log = [];
  const stats = { committed: 0, aborted: 0, retries: 0, abandoned: 0, steps: 0 };
  let guard = 0;
  while (queue.length > 0) {
    if (guard++ > 500) throw new Error("scheduler did not converge");
    const { step, attempt } = queue.shift();
    let txn = live.get(step.txn);
    if (txn === undefined) {
      txn = memory.begin();
      live.set(step.txn, txn);
      log.push(`${step.txn}#${attempt}:begin`);
    }
    stats.steps++;
    if (step.op === "read") {
      log.push(`${step.txn}#${attempt}:read(${step.addr})=${memory.read(txn, step.addr)}`);
    } else if (step.op === "write") {
      memory.write(txn, step.addr, step.value);
      log.push(`${step.txn}#${attempt}:write(${step.addr},${step.value})`);
    } else if (step.op === "commit") {
      const outcome = memory.commit(txn);
      log.push(`${step.txn}#${attempt}:${outcome}`);
      live.delete(step.txn);
      if (outcome === "committed") {
        stats.committed++;
      } else {
        stats.aborted++;
        if (attempt >= maxAttempts) {
          stats.abandoned++;
          log.push(`${step.txn}:abandoned`);
        } else {
          stats.retries++;
          for (const op of opsByTxn.get(step.txn)) queue.push({ step: op, attempt: attempt + 1 });
        }
      }
    } else {
      throw new Error(`unknown op ${step.op}`);
    }
  }
  return { log, stats };
}

module.exports = { TxnMemory, runScript };

if (require.main === module) {
  const assert = require("node:assert");

  // --- sequential script: the result must equal the sequential application ---
  const sequential = new TxnMemory(4);
  const sequentialRun = runScript(sequential, [
    { txn: "a", op: "write", addr: 0, value: 10 },
    { txn: "a", op: "commit" },
    { txn: "b", op: "read", addr: 0 },
    { txn: "b", op: "write", addr: 1, value: 20 },
    { txn: "b", op: "commit" },
  ]);
  assert.strictEqual(sequentialRun.stats.committed, 2);
  assert.strictEqual(sequentialRun.stats.aborted, 0);
  assert.deepStrictEqual(sequential.values, [10, 20, 0, 0]);
  assert.ok(sequential.version >= 1);

  // --- write-write conflict: the second commit must abort and be replayed ---
  const ww = new TxnMemory(2);
  const wwRun = runScript(ww, [
    { txn: "t1", op: "read", addr: 0 },
    { txn: "t2", op: "read", addr: 0 },
    { txn: "t1", op: "write", addr: 0, value: 1 },
    { txn: "t1", op: "commit" },
    { txn: "t2", op: "write", addr: 0, value: 2 },
    { txn: "t2", op: "commit" },
  ]);
  assert.strictEqual(wwRun.stats.committed, 2, "the retry must succeed");
  assert.strictEqual(wwRun.stats.aborted, 1, "the interleaving must produce one conflict");
  assert.strictEqual(wwRun.stats.retries, 1);
  assert.ok(wwRun.log.includes("t2#1:aborted") && wwRun.log.includes("t2#2:committed"));
  assert.deepStrictEqual(ww.values, [2, 0], "the retry wrote last");

  // --- read-write conflict: t1's log entry is stale after t2 commits ---
  const rw = new TxnMemory(2);
  const rwRun = runScript(rw, [
    { txn: "t1", op: "read", addr: 0 },
    { txn: "t2", op: "write", addr: 0, value: 5 },
    { txn: "t2", op: "commit" },
    { txn: "t1", op: "write", addr: 1, value: 6 },
    { txn: "t1", op: "commit" },
  ]);
  assert.strictEqual(rwRun.stats.aborted, 1, "a write that invalidates a read must abort the reader");
  assert.ok(rwRun.log.includes("t1#1:aborted"));
  assert.strictEqual(rw.values[1], 6, "the retried transaction still writes its own address");
  assert.strictEqual(rw.version, 2, "one version per committed write transaction");

  // --- atomicity: values are invisible until commit, and aborts leave no trace ---
  const isolate = new TxnMemory(2);
  isolate.write(isolate.begin(), 0, 99);
  assert.strictEqual(isolate.values[0], 0, "the write set must not leak into memory");
  const probe = isolate.begin();
  assert.strictEqual(isolate.read(probe, 0), 0, "another transaction still sees the old value");
  const writer = isolate.begin();
  isolate.write(writer, 0, 7);
  assert.strictEqual(isolate.read(writer, 0), 7, "a transaction reads its own writes");
  assert.strictEqual(isolate.commit(writer), "committed");
  const afterCommit = isolate.snapshot();
  const doomed = isolate.begin();
  isolate.read(doomed, 0);
  const meddler = isolate.begin();
  isolate.write(meddler, 0, 55);
  assert.strictEqual(isolate.commit(meddler), "committed");
  const beforeAbort = isolate.snapshot();
  isolate.write(doomed, 0, 1234);
  assert.strictEqual(isolate.commit(doomed), "aborted", "the aborted attempt must not install");
  assert.deepStrictEqual(isolate.snapshot(), beforeAbort, "an abort changes nothing");
  assert.notDeepStrictEqual(isolate.snapshot(), afterCommit, "the conflicting commit did take effect");
  assert.strictEqual(isolate.read(isolate.begin(), 0), 55);

  // --- read-only transactions validate but never bump the version ---
  const reader = new TxnMemory(2);
  const first = reader.begin();
  reader.write(first, 0, 3);
  reader.commit(first);
  const versionAfterWrite = reader.version;
  const readOnly = reader.begin();
  reader.read(readOnly, 0);
  assert.strictEqual(reader.commit(readOnly), "committed");
  assert.strictEqual(reader.version, versionAfterWrite, "read-only commits do not bump the version");
  assert.strictEqual(reader.installedWrites, 1);
  const stale = reader.begin();
  reader.read(stale, 0);
  const other = reader.begin();
  reader.write(other, 0, 4);
  reader.commit(other);
  assert.strictEqual(reader.commit(stale), "aborted", "read-only transactions validate too");

  // --- livelock is bounded by the attempt limit ---
  const livelock = new TxnMemory(2);
  const livelockScript = [];
  for (let round = 0; round < 6; round++) {
    livelockScript.push({ txn: `p${round}`, op: "read", addr: 0 });
    livelockScript.push({ txn: `p${round}`, op: "write", addr: 0, value: round + 1 });
  }
  for (let round = 0; round < 6; round++) livelockScript.push({ txn: `p${round}`, op: "commit" });
  const livelockRun = runScript(livelock, livelockScript, { maxAttempts: 1 });
  assert.ok(livelockRun.stats.abandoned > 0, "a single attempt gives up on the conflicts");
  assert.ok(livelockRun.stats.committed >= 1);
  const patientRun = runScript(new TxnMemory(2), livelockScript, { maxAttempts: 8 });
  assert.strictEqual(patientRun.stats.abandoned, 0, "with enough attempts every transaction commits");

  // --- error cases and determinism ---
  const errors = new TxnMemory(2);
  const txn = errors.begin();
  assert.throws(() => errors.read(txn, 5), RangeError);
  assert.throws(() => errors.write(txn, -1, 0), RangeError);
  assert.throws(() => errors.write(txn, 0, 1.5), TypeError);
  assert.throws(() => errors.read({}, 0), TypeError);
  errors.commit(txn);
  assert.throws(() => errors.commit(txn), /already finished/);
  assert.throws(() => errors.read(txn, 0), /already finished/);
  assert.throws(() => new TxnMemory(0), RangeError);
  const replayA = runScript(new TxnMemory(2), livelockScript, { maxAttempts: 8 });
  const replayB = runScript(new TxnMemory(2), livelockScript, { maxAttempts: 8 });
  assert.deepStrictEqual(replayA.log, replayB.log, "the simulator is deterministic");

  console.log(`commits=${wwRun.stats.committed} aborts=${wwRun.stats.aborted} retries=${wwRun.stats.retries} steps=${wwRun.stats.steps}`);
  console.log("PASS");
}
