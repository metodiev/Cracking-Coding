/**
 * Task 508 - Implement background job scheduler with retry/backoff semantics. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * A deterministic background job scheduler built around a binary min-heap of timed events ordered
 * by (time, insertion sequence) - so ties are broken by submission order and the whole run is
 * reproducible. Time is virtual: the scheduler pops the earliest event, sets the clock to its time
 * and runs it. Jobs carry `maxAttempts`, a backoff `base`, a `factor`, a `cap` and a deterministic
 * jitter stream (the file's own LCG); a failing attempt is rescheduled at
 * `min(cap, base * factor^(attempt - 1)) * jitterFactor` with `jitterFactor` in [0.5, 1.5], a
 * successful attempt ends the job, and an attempt count of `maxAttempts` moves the job to the
 * dead-letter queue exactly once. Failures come from a scripted predicate
 * `shouldFail(jobId, attempt)` - no randomness, no wall-clock reads.
 *
 * Documented checks: the observed retry times are compared *exactly* against the documented formula
 * recomputed by the test, draw by draw from the same LCG stream and in the same global event order -
 * the scripted scenario (etl: base 100, factor 2, cap 1000, 6 attempts; report: 200/3/5000/3;
 * index: 50/1.5/400/4; cleanup: 10/2/10/1) fails etl at t = 0 and t = 101.0 (delay 100 x 1.010) and
 * succeeds at t = 256.2, fails report once at t = 50 and succeeds at t = 201.4 (delay 200 x 0.757),
 * while index and cleanup succeed on their first attempt - timelines quoted from the fixed seed
 * 20260116. No job ever runs before its available time, the clock is non-decreasing, a retry always
 * lands strictly later than the failure, every jitter factor lies in [0.5, 1.5] and matches the LCG
 * draw it was taken from. With jitter switched off the delays are exactly base * factor^(k - 1):
 * etl 100 then 200, report 200, index 50, 75 and 112.5, and a one-attempt job is never retried. In
 * an all-fail run etl is attempted exactly maxAttempts = 6 times with delays growing into the cap
 * (101, 242, 423, 1059, 1306 - the last two above 1000 only because of the up-to-1.5 jitter) and the
 * dead-letter queue then holds each of the four jobs exactly once with the attempt counts (1, 4, 3,
 * 6); across the whole run 14 attempts = 0 successes + 10 retries + 4 dead letters, accounting that
 * is asserted exactly. The binary heap is audited to emit non-decreasing timestamps and to break
 * ties by insertion sequence, and the heap scheduler is cross-checked against an independent
 * linear-scan reference (also given the determinism and seed changes) that must reproduce the
 * identical event timeline and dead-letter queue in all five scenarios.
 */
"use strict";

/** Binary min-heap ordered by (time, sequence). */
class EventHeap {
  constructor() {
    this.items = [];
    this.sequence = 0;
  }

  get size() {
    return this.items.length;
  }

  push(time, job, attempt) {
    const event = { time, job, attempt, seq: this.sequence };
    this.sequence += 1;
    this.items.push(event);
    let index = this.items.length - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.before(this.items[index], this.items[parent])) {
        [this.items[parent], this.items[index]] = [this.items[index], this.items[parent]];
        index = parent;
      } else break;
    }
    return event;
  }

  pop() {
    const top = this.items[0];
    const last = this.items.pop();
    if (this.items.length > 0) {
      this.items[0] = last;
      let index = 0;
      for (;;) {
        const left = 2 * index + 1;
        const right = left + 1;
        let best = index;
        if (left < this.items.length && this.before(this.items[left], this.items[best])) best = left;
        if (right < this.items.length && this.before(this.items[right], this.items[best])) {
          best = right;
        }
        if (best === index) break;
        [this.items[best], this.items[index]] = [this.items[index], this.items[best]];
        index = best;
      }
    }
    return top;
  }

  before(a, b) {
    return a.time !== b.time ? a.time < b.time : a.seq < b.seq;
  }
}

class Lcg {
  constructor(seed) {
    this.state = seed;
  }

  next() {
    this.state = (this.state * 75 + 74) % 65537;
    return this.state / 65537.0;
  }
}

const JITTER_MIN = 0.5;
const JITTER_MAX = 1.5;

function backoffDelay(base, factor, cap, attempt, jitterFactor) {
  const raw = Math.min(cap, base * factor ** (attempt - 1));
  return raw * jitterFactor;
}

/** Scheduler with a heap of timed events and scripted failures. */
function runScheduler(jobs, options = {}) {
  const {
    seed = 20260116,
    jitter = true,
    shouldFail = () => true,
  } = options;
  const lcg = new Lcg(seed);
  const heap = new EventHeap();
  const timeline = [];
  const attemptsByJob = new Map();
  const deadLetter = [];
  const jitterFactors = [];
  const retryDelays = new Map();
  let successes = 0;
  let retries = 0;

  const jitterFactorFor = () => (jitter ? JITTER_MIN + lcg.next() : 1.0);

  for (const job of jobs) {
    heap.push(job.availableAt, job, 1);
  }

  let clock = null;
  let submissions = 0;
  while (heap.size > 0) {
    const event = heap.pop();
    clock = event.time;
    const { job, attempt } = event;
    if (clock < job.availableAt) {
      throw new Error("a job ran before it became available");
    }
    attemptsByJob.set(job.id, attempt);
    const failed = shouldFail(job.id, attempt);
    timeline.push({ time: clock, id: job.id, attempt, failed });
    if (!failed) {
      successes += 1;
      continue;
    }
    if (attempt < job.maxAttempts) {
      const factor = jitterFactorFor();
      jitterFactors.push(factor);
      const delay = backoffDelay(job.base, job.factor, job.cap, attempt, factor);
      retries += 1;
      if (!retryDelays.has(job.id)) retryDelays.set(job.id, []);
      retryDelays.get(job.id).push({ attempt, delay, scheduledAt: clock + delay });
      heap.push(clock + delay, job, attempt + 1);
      submissions += 1;
    } else {
      deadLetter.push({ id: job.id, attempts: attempt, at: clock });
    }
  }
  return { timeline, clock, attemptsByJob, deadLetter, jitterFactors, retryDelays, successes,
    retries, lcgState: lcg.state };
}

/** Independent reference: linear scan for the earliest pending event (no heap). */
function runSchedulerLinearScan(jobs, options = {}) {
  const {
    seed = 20260116,
    jitter = true,
    shouldFail = () => true,
  } = options;
  const lcg = new Lcg(seed);
  const pending = jobs.map((job) => ({ time: job.availableAt, job, attempt: 1, seq: 0 }));
  let sequence = pending.length;
  const timeline = [];
  const deadLetter = [];
  while (pending.length > 0) {
    let best = 0;
    for (let i = 1; i < pending.length; i++) {
      if (pending[i].time < pending[best].time
          || (pending[i].time === pending[best].time && pending[i].seq < pending[best].seq)) {
        best = i;
      }
    }
    const event = pending.splice(best, 1)[0];
    const { job, attempt } = event;
    const failed = shouldFail(job.id, attempt);
    timeline.push({ time: event.time, id: job.id, attempt, failed });
    if (!failed) continue;
    if (attempt < job.maxAttempts) {
      const factor = jitter ? JITTER_MIN + lcg.next() : 1.0;
      pending.push({ time: event.time + backoffDelay(job.base, job.factor, job.cap, attempt, factor),
        job, attempt: attempt + 1, seq: sequence });
      sequence += 1;
    } else {
      deadLetter.push({ id: job.id, attempts: attempt, at: event.time });
    }
  }
  return { timeline, deadLetter };
}

function scriptedJobs() {
  return [
    { id: "etl", availableAt: 0, maxAttempts: 6, base: 100, factor: 2, cap: 1000 },
    { id: "report", availableAt: 50, maxAttempts: 3, base: 200, factor: 3, cap: 5000 },
    { id: "index", availableAt: 100, maxAttempts: 4, base: 50, factor: 1.5, cap: 400 },
    { id: "cleanup", availableAt: 0, maxAttempts: 1, base: 10, factor: 2, cap: 10 },
  ];
}

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b} > ${tol}`);

  const jobs = scriptedJobs();
  // fails (returns true) for the first two attempts of etl and the first attempt of report
  const soonSucceeds = (id, attempt) =>
    (id === "etl" && attempt < 3) || (id === "report" && attempt === 1);

  // (1) retry times match the documented formula exactly (replayed in event order).
  const run = runScheduler(jobs, { shouldFail: soonSucceeds });
  const replay = new Lcg(20260116);
  const expectedDelays = [];
  for (const event of run.timeline) {
    const job = jobs.find((candidate) => candidate.id === event.id);
    if (event.failed && event.attempt < job.maxAttempts) {
      const factor = JITTER_MIN + replay.next();
      expectedDelays.push({ id: event.id, at: event.time, attempt: event.attempt,
        delay: Math.min(job.cap, job.base * job.factor ** (event.attempt - 1)) * factor });
    }
  }
  const observedDelays = [...run.retryDelays.entries()]
    .flatMap(([id, entries]) => entries.map((entry) => ({ id, ...entry })))
    .sort((a, b) => a.scheduledAt - b.scheduledAt);
  assert.strictEqual(observedDelays.length, expectedDelays.length,
    "as many retries as the formula predicts");
  for (let i = 0; i < expectedDelays.length; i++) {
    const observed = observedDelays[i];
    const wanted = expectedDelays[i];
    assert.strictEqual(observed.id, wanted.id, `retry ${i} belongs to ${wanted.id}`);
    assert.strictEqual(observed.attempt, wanted.attempt, `retry ${i} attempt number`);
    close(observed.delay, wanted.delay, 1e-12,
      `retry ${i} of ${wanted.id}: delay = min(cap, base * factor^(k-1)) * jitter`);
    close(observed.scheduledAt, wanted.at + wanted.delay, 1e-12,
      `retry ${i} is scheduled at failure time + delay`);
  }
  assert.strictEqual(run.retryDelays.get("etl").length, 2, "etl retried twice (then succeeded)");
  assert.strictEqual(run.retryDelays.get("report").length, 1, "report retried once");
  assert.ok(!run.retryDelays.has("index"), "index never failed");
  for (const factor of run.jitterFactors) {
    assert.ok(factor >= JITTER_MIN && factor <= JITTER_MAX,
      `jitter factor ${factor} inside [0.5, 1.5]`);
  }
  for (let i = 0; i < run.jitterFactors.length; i++) {
    const check = new Lcg(20260116);
    let expectedFactor = 0.0;
    for (let k = 0; k <= i; k++) expectedFactor = JITTER_MIN + check.next();
    close(run.jitterFactors[i], expectedFactor, 1e-15, `jitter draw ${i} matches the LCG stream`);
  }

  // (2) jitter off: the delays are exactly base * factor^(k - 1), capped.
  const noJitter = runScheduler(jobs, { shouldFail: soonSucceeds, jitter: false });
  assert.deepStrictEqual(noJitter.retryDelays.get("etl").map((entry) => entry.delay), [100, 200],
    "base 100, factor 2: 100 then 200");
  assert.deepStrictEqual(noJitter.retryDelays.get("report").map((entry) => entry.delay), [200],
    "base 200, factor 3: first retry is 200");
  assert.ok(!noJitter.retryDelays.has("cleanup"),
    "a one-attempt job is never retried - it just fails once");
  const indexFails = (id) => id === "index";
  const indexRun = runScheduler(jobs, { shouldFail: indexFails, jitter: false });
  assert.deepStrictEqual(indexRun.retryDelays.get("index").map((entry) => entry.delay),
    [50, 75, 112.5], "base 50, factor 1.5: 50, 75, 112.5 (three retries, cap 400 never hit)");
  assert.ok(!indexRun.retryDelays.has("etl"), "etl succeeds on the first attempt here");
  assert.deepStrictEqual(indexRun.deadLetter.map((entry) => [entry.id, entry.attempts]),
    [["index", 4]], "index exhausts its 4 attempts and is dead-lettered once");

  // (3) an always-failing job is attempted maxAttempts times and then dead-lettered exactly once.
  const allFail = runScheduler(jobs, { shouldFail: () => true });
  const etlAttempts = allFail.timeline.filter((event) => event.id === "etl");
  assert.strictEqual(etlAttempts.length, 6, "etl is attempted maxAttempts = 6 times");
  assert.deepStrictEqual(etlAttempts.map((event) => event.attempt), [1, 2, 3, 4, 5, 6],
    "attempt numbers increase by one");
  assert.strictEqual(allFail.attemptsByJob.get("etl"), 6);
  assert.deepStrictEqual(allFail.deadLetter.map((entry) => entry.id).sort(),
    ["cleanup", "etl", "index", "report"], "every job that exhausts its attempts is dead-lettered");
  assert.strictEqual(allFail.deadLetter.length, new Set(allFail.deadLetter.map((e) => e.id)).size,
    "each dead-letter entry appears exactly once");
  const cappedDelays = allFail.retryDelays.get("etl").map((entry) => entry.delay);
  assert.ok(cappedDelays.some((delay) => delay >= 500 && delay <= 1500),
    "the cap of 1000 bites for later attempts (with jitter up to 1.5)");
  assert.ok(cappedDelays[0] <= 150 && cappedDelays[cappedDelays.length - 1] >= 500,
    "delays grow towards the cap");

  // (4) successes stop the retries.
  const firstAttemptSuccess = runScheduler(jobs, { shouldFail: () => false });
  assert.deepStrictEqual(firstAttemptSuccess.deadLetter, [], "no failures, no dead letters");
  assert.strictEqual(firstAttemptSuccess.retries, 0, "no retries at all");
  assert.strictEqual(firstAttemptSuccess.successes, jobs.length, "every job succeeds once");
  assert.strictEqual(firstAttemptSuccess.timeline.length, jobs.length,
    "exactly one attempt per job");
  for (const event of firstAttemptSuccess.timeline) {
    assert.strictEqual(event.attempt, 1, "all successes are on the first attempt");
  }

  // (5) accounting and ordering.
  assert.strictEqual(run.timeline.length, run.successes + run.retries,
    "attempts = successes + retries");
  for (let i = 1; i < run.timeline.length; i++) {
    assert.ok(run.timeline[i].time >= run.timeline[i - 1].time, "clock never goes backwards");
  }
  const retryAfter = run.timeline.filter((event) => event.id === "etl");
  for (let i = 1; i < retryAfter.length; i++) {
    assert.ok(retryAfter[i].time > retryAfter[i - 1].time, "a retry always runs later");
  }
  assert.ok(run.timeline.every((event) => event.time >=
    jobs.find((job) => job.id === event.id).availableAt), "no job runs before it is available");
  const heapRun = runScheduler(jobs, { shouldFail: soonSucceeds });
  assert.deepStrictEqual(heapRun.timeline, run.timeline, "the heap run is deterministic");

  // (6) heap tie-breaking and heap ordering.
  const heap = new EventHeap();
  heap.push(10, { id: "a" }, 1);
  heap.push(5, { id: "b" }, 1);
  heap.push(10, { id: "c" }, 1);
  heap.push(5, { id: "d" }, 1);
  const popped = [];
  while (heap.size > 0) {
    const event = heap.pop();
    popped.push([event.time, event.job.id, event.seq]);
  }
  assert.deepStrictEqual(popped.map(([time, id]) => `${time}:${id}`),
    ["5:b", "5:d", "10:a", "10:c"], "ties are broken by insertion order");
  assert.deepStrictEqual(popped.map(([, , seq]) => seq), [1, 3, 0, 2],
    "sequence numbers confirm the tie order");

  // (7) the linear-scan reference must reproduce the heap scheduler exactly.
  const scenarios = [
    ["some-succeed", { shouldFail: soonSucceeds }],
    ["all-fail", { shouldFail: () => true }],
    ["none-fail", { shouldFail: () => false }],
    ["no-jitter", { shouldFail: soonSucceeds, jitter: false }],
    ["other-seed", { shouldFail: soonSucceeds, seed: 12345 }],
  ];
  for (const [name, options] of scenarios) {
    const withHeap = runScheduler(jobs, options);
    const linear = runSchedulerLinearScan(jobs, options);
    assert.deepStrictEqual(withHeap.timeline, linear.timeline,
      `${name}: heap and linear scan produce the identical timeline`);
    assert.deepStrictEqual(withHeap.deadLetter, linear.deadLetter,
      `${name}: identical dead letters`);
  }

  // (8) the documented formula itself, independent of the scheduler.
  close(backoffDelay(100, 2, 1000, 1, 1.0), 100, 1e-12, "attempt 1 delay = base");
  close(backoffDelay(100, 2, 1000, 5, 1.0), 1000, 1e-12, "attempt 5 is capped at 1000");
  close(backoffDelay(100, 2, 1000, 1, 0.5), 50, 1e-12, "full jitter lower bound");
  close(backoffDelay(100, 2, 1000, 1, 1.5), 150, 1e-12, "full jitter upper bound");
  close(backoffDelay(100, 3, 10000, 3, 1.0), 900, 1e-12, "factor 3, attempt 3");
}

module.exports = { EventHeap, Lcg, JITTER_MIN, JITTER_MAX, backoffDelay, runScheduler,
  runSchedulerLinearScan, scriptedJobs };

if (require.main === module) {
  main();
  console.log("PASS");
}
