/** Task 620 - Implement lock elision simulation and contention measurement. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * Lock elision with a simulation of the contention that decides whether it pays. Two
 * implementations of the same two-word critical section (each operation increments `value` and
 * `applied`, which must stay in step):
 *
 *   CoarseCounter - three steps per operation: acquire, mutate, release. Every operation pays a
 *                   lock acquisition and a contended operation is counted as blocked.
 *   ElidedCounter - the operation speculates: it snapshots the version stamp and the two words,
 *                   and later tries to commit by validating the stamp (a single CAS on the
 *                   version, standing in for the hardware read-set validation of an elided
 *                   critical section). A failed validation means somebody else committed in
 *                   between, so the attempt aborts; after `maxRetries` failed attempts the
 *                   operation falls back to the real lock, which is exactly what a production
 *                   elision implementation does under persistent contention.
 *
 * No threads are used: a scheduler walks a list of step owners, so the same code runs the
 * uncontended schedule (each thread runs to completion) and the maximally interleaved schedule
 * (round-robin, so every thread speculates before anybody commits).
 *
 * What is simplified: one shared version stamp instead of per-word stamps, no real CAS or cache
 * line ownership, and a counter workload instead of a general critical section. Verified: both
 * implementations always end with exact, consistent state (no lost update, `value == applied`),
 * the elided version takes no lock at all when uncontended and strictly fewer acquisitions than
 * the coarse one, aborts grow with the number of threads on the contended schedule, the fallback
 * path keeps the result exact when retries are capped, bookkeeping identities hold
 * (elisions + fallback commits == completed, stamp == value), and identical schedules replay
 * identically.
 */
"use strict";

class CoarseCounter {
  constructor(threads, ops) {
    if (!Number.isInteger(threads) || threads < 1) throw new RangeError("threads must be positive");
    if (!Number.isInteger(ops) || ops < 0) throw new RangeError("ops must not be negative");
    this.threads = threads;
    this.ops = ops;
    this.value = 0;
    this.applied = 0;
    this.lock = false;
    this.owner = null;
    this.phase = new Array(threads).fill(0);
    this.done = new Array(threads).fill(0);
    this.acquisitions = 0;
    this.blocked = 0;
    this.completed = 0;
    this.lockCommits = 0;
  }

  get finished() { return this.completed === this.ops * this.threads; }

  step(thread) {
    if (this.done[thread] >= this.ops) return "finished";
    const phase = this.phase[thread];
    if (phase === 0) {
      if (this.lock) { this.blocked++; return "blocked"; }
      this.lock = true;
      this.owner = thread;
      this.acquisitions++;
      this.phase[thread] = 1;
      return "lock";
    }
    if (phase === 1) {
      this.value++;
      this.applied++;
      this.phase[thread] = 2;
      return "mutate";
    }
    this.lock = false;
    this.owner = null;
    this.phase[thread] = 0;
    this.done[thread]++;
    this.completed++;
    this.lockCommits++;
    return "unlock";
  }
}

class ElidedCounter {
  constructor(threads, ops, maxRetries = 3) {
    if (!Number.isInteger(threads) || threads < 1) throw new RangeError("threads must be positive");
    if (!Number.isInteger(ops) || ops < 0) throw new RangeError("ops must not be negative");
    if (!Number.isInteger(maxRetries) || maxRetries < 1) throw new RangeError("maxRetries must be positive");
    this.threads = threads;
    this.ops = ops;
    this.maxRetries = maxRetries;
    this.value = 0;
    this.applied = 0;
    this.stamp = 0;                            // version of both words, bumped by every commit
    this.lock = false;
    this.owner = null;
    this.phase = new Array(threads).fill(0);
    this.done = new Array(threads).fill(0);
    this.attempts = new Array(threads).fill(0);
    this.view = new Array(threads).fill(null);
    this.elisions = 0;
    this.lockCommits = 0;
    this.aborts = 0;
    this.fallbacks = 0;
    this.acquisitions = 0;
    this.blocked = 0;
    this.completed = 0;
  }

  get finished() { return this.completed === this.ops * this.threads; }

  get commitAttempts() { return this.elisions + this.lockCommits + this.aborts; }

  step(thread) {
    if (this.done[thread] >= this.ops) return "finished";
    const phase = this.phase[thread];
    if (phase === 0) {                                       // speculate: snapshot the read set
      this.view[thread] = { stamp: this.stamp, value: this.value, applied: this.applied };
      this.phase[thread] = 1;
      return "speculate";
    }
    if (phase === 1) {                                       // validate and commit without the lock
      if (this.view[thread].stamp === this.stamp) {
        this.value = this.view[thread].value + 1;
        this.applied = this.view[thread].applied + 1;
        this.stamp++;
        this.elisions++;
        this.phase[thread] = 3;
        return "commit";
      }
      this.aborts++;
      this.attempts[thread]++;
      if (this.attempts[thread] >= this.maxRetries) {
        this.fallbacks++;
        this.phase[thread] = 4;                              // give up and take the lock
        return "abort-to-fallback";
      }
      this.phase[thread] = 0;                                // retry the speculation
      return "abort";
    }
    if (phase === 3) {                                       // finish a successful elision
      this.phase[thread] = 0;
      this.attempts[thread] = 0;
      this.done[thread]++;
      this.completed++;
      return "done";
    }
    if (phase === 4) {
      if (this.lock) { this.blocked++; return "blocked-fallback"; }
      this.lock = true;
      this.owner = thread;
      this.acquisitions++;
      this.phase[thread] = 5;
      return "lock";
    }
    if (phase === 5) {
      this.value++;
      this.applied++;
      this.stamp++;
      this.phase[thread] = 6;
      return "mutate";
    }
    this.lock = false;
    this.owner = null;
    this.phase[thread] = 0;
    this.attempts[thread] = 0;
    this.done[thread]++;
    this.completed++;
    this.lockCommits++;
    return "unlock";
  }
}

/** Runs a schedule: an array of thread ids, or a function (sim, stepIndex) -> thread id. */
function drive(sim, schedule, limit = 50000) {
  const one = typeof schedule === "function" ? null : schedule;
  let steps = 0;
  while (!sim.finished) {
    if (steps > limit) throw new Error("schedule did not converge");
    const thread = one === null ? schedule(sim, steps) : one[steps % one.length];
    sim.step(thread);
    steps++;
  }
  return { steps, metrics: metricsOf(sim) };
}

const roundRobin = (threads) => Array.from({ length: threads }, (_, i) => i);
const runToCompletion = (sim) => {
  for (let thread = 0; thread < sim.threads; thread++) if (sim.done[thread] < sim.ops) return thread;
  return 0;
};

const metricsOf = (sim) => ({
  value: sim.value, applied: sim.applied, completed: sim.completed, stamp: sim.stamp ?? null,
  acquisitions: sim.acquisitions, blocked: sim.blocked, aborts: sim.aborts ?? 0, elisions: sim.elisions ?? 0,
  fallbacks: sim.fallbacks ?? 0, lockCommits: sim.lockCommits, locked: sim.lock,
});

/** Every run must end exact: no lost update, the two words in step, the lock released. */
function checkInvariants(sim, threads, ops, label) {
  const expected = threads * ops;
  if (sim.value !== expected || sim.applied !== expected) {
    throw new Error(`${label}: lost updates (value=${sim.value}, applied=${sim.applied}, expected ${expected})`);
  }
  if (sim.completed !== expected) throw new Error(`${label}: ${sim.completed} operations completed, expected ${expected}`);
  if (sim.lock) throw new Error(`${label}: the lock is still held at the end`);
  if (sim.stamp !== undefined && sim.stamp !== expected) throw new Error(`${label}: stamp ${sim.stamp} != ${expected}`);
  if (sim.elisions !== undefined && sim.elisions + sim.lockCommits !== expected) {
    throw new Error(`${label}: elisions (${sim.elisions}) + lock commits (${sim.lockCommits}) != ${expected}`);
  }
  if (sim.fallbacks !== undefined && (sim.aborts < sim.fallbacks || sim.lockCommits < sim.fallbacks)) {
    throw new Error(`${label}: inconsistent fallback bookkeeping`);
  }
  return true;
}

/** Aborts per operation: the contention the elided version actually saw. */
function contention(sim) {
  return sim.aborts / (sim.threads * sim.ops);
}

module.exports = { CoarseCounter, ElidedCounter, drive, checkInvariants, contention, metricsOf };

if (require.main === module) {
  const assert = require("node:assert");
  const table = [];

  // --- uncontended schedule: elision never touches the lock ---
  for (const threads of [1, 2, 4, 8]) {
    const ops = 4;
    const coarse = new CoarseCounter(threads, ops);
    drive(coarse, runToCompletion);
    checkInvariants(coarse, threads, ops, `coarse sequential ${threads}`);
    assert.strictEqual(coarse.acquisitions, threads * ops, "the coarse version locks once per operation");
    assert.strictEqual(coarse.blocked, 0, "running to completion never contends");

    const elided = new ElidedCounter(threads, ops);
    drive(elided, runToCompletion);
    checkInvariants(elided, threads, ops, `elided sequential ${threads}`);
    assert.strictEqual(elided.acquisitions, 0, "an uncontended elided operation must not lock");
    assert.strictEqual(elided.aborts, 0, "nothing can invalidate an unshared read set");
    assert.strictEqual(elided.elisions, threads * ops);
    assert.ok(elided.commitAttempts === threads * ops, "one commit attempt per operation");
  }

  // --- contended schedule: round-robin forces every thread to speculate before anybody commits ---
  for (const threads of [1, 2, 4, 8]) {
    const ops = 4;
    const elided = new ElidedCounter(threads, ops);
    drive(elided, roundRobin(threads));
    checkInvariants(elided, threads, ops, `elided contended ${threads}`);
    const coarse = new CoarseCounter(threads, ops);
    drive(coarse, roundRobin(threads));
    checkInvariants(coarse, threads, ops, `coarse contended ${threads}`);
    assert.ok(coarse.acquisitions >= threads * ops, "the coarse version always locks");
    if (threads > 1) {
      assert.ok(elided.aborts > 0, `contention at ${threads} threads must produce aborts`);
      assert.ok(coarse.blocked > 0, `contention at ${threads} threads must block the coarse lock`);
    }
    table.push({ threads, elidedAborts: elided.aborts, elidedLocks: elided.acquisitions, coarseLocks: coarse.acquisitions, coarseBlocked: coarse.blocked, contention: contention(elided).toFixed(2) });
  }
  const abortsOf = (entry) => entry.elidedAborts;
  const ordered = table.map(abortsOf);
  assert.strictEqual(ordered[0], 0, "a single thread never aborts");
  assert.ok(ordered[3] > ordered[1] && ordered[1] >= ordered[0], "aborts grow with the thread count");
  const contendedEight = table[table.length - 1];
  assert.ok(contendedEight.elidedLocks >= 0 && contendedEight.coarseLocks > contendedEight.elidedLocks,
    "even under contention the elided version locks less than the coarse one");

  // --- a cap on retries pushes operations onto the fallback path ---
  const capped = new ElidedCounter(8, 4, 1);
  drive(capped, roundRobin(8));
  checkInvariants(capped, 8, 4, "elided maxRetries=1");
  assert.ok(capped.fallbacks > 0, "with a single attempt some operations must fall back to the lock");
  assert.ok(capped.acquisitions > 0 && capped.acquisitions <= capped.fallbacks + 8, "the fallback takes the lock");
  assert.strictEqual(capped.value, 32, "the fallback keeps the result exact");
  const patient = new ElidedCounter(8, 4, 50);
  drive(patient, roundRobin(8));
  checkInvariants(patient, 8, 4, "elided maxRetries=50");
  assert.strictEqual(patient.fallbacks, 0, "a patient transaction never needs the lock");
  assert.ok(patient.aborts > 0, "and it keeps retrying while others commit");

  // --- degenerate cases, replay determinism and argument validation ---
  const zeroOps = new ElidedCounter(4, 0);
  drive(zeroOps, roundRobin(4));
  checkInvariants(zeroOps, 4, 0, "no operations");
  assert.strictEqual(zeroOps.aborts, 0);
  const single = new ElidedCounter(1, 1);
  drive(single, roundRobin(1));
  checkInvariants(single, 1, 1, "one operation");
  assert.strictEqual(single.value, 1);
  const replayA = drive(new ElidedCounter(4, 6), roundRobin(4));
  const replayB = drive(new ElidedCounter(4, 6), roundRobin(4));
  assert.deepStrictEqual(replayA.metrics, replayB.metrics, "identical schedules replay identically");
  assert.deepStrictEqual(replayA.metrics, drive(new ElidedCounter(4, 6), roundRobin(4)).metrics);
  assert.throws(() => new ElidedCounter(0, 1), RangeError);
  assert.throws(() => new ElidedCounter(2, -1), RangeError);
  assert.throws(() => new ElidedCounter(2, 1, 0), RangeError);
  assert.throws(() => new CoarseCounter(0, 1), RangeError);
  assert.throws(() => new CoarseCounter(1, -1), RangeError);

  for (const entry of table) {
    console.log(`threads=${entry.threads} elidedAborts=${entry.elidedAborts} elidedLocks=${entry.elidedLocks} ` +
      `coarseLocks=${entry.coarseLocks} coarseBlocked=${entry.coarseBlocked} contention/op=${entry.contention}`);
  }
  console.log("PASS");
}
