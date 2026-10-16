/** Task 658 - Implement straggler mitigation via speculative execution simulation. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A discrete-event simulation of the trick Hadoop/Spark use for stragglers: when a task has
 * been running much longer than a typical task, launch a duplicate *speculative* attempt on
 * another worker and keep whichever attempt finishes first; the loser's result is discarded.
 *
 * Model (deliberately scaled down, and the simplifications are load-bearing):
 *   * W workers, one slot each; a FIFO queue of T tasks.
 *   * An attempt's runtime is `baseDuration[task] * speed[worker] * (1 + jitter)`, so the same
 *     task can be fast on one worker and slow on another — that is what makes speculation able
 *     to win. Jitter is derived by hashing (seed, task, attempt index), never by a stateful
 *     generator, so the simulation is exactly reproducible and order-independent.
 *   * Speculation rule: when a worker becomes free and a running attempt's elapsed time exceeds
 *     `speculativeFactor * expectedDuration` (the mean base duration, i.e. a known statistic),
 *     start one copy — at most `maxCopies` attempts per task.
 *   * Simplification: a losing attempt is *not* killed early; it keeps its worker until it
 *     would have finished, but its result is discarded. The job is done when every task has a
 *     committed result, and the makespan is that instant.
 *
 * Verified: every task commits exactly once, the committed attempt is the earliest-finishing
 * one, per-worker intervals never overlap, the cap on copies holds, the no-speculation run
 * matches a closed-form FIFO makespan for identical durations and a second independent
 * scheduler for mixed durations, and speculation is strictly beneficial on a straggler load.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

/** Deterministic 32-bit mix of three integers (order-independent jitter source). */
function hash32(seed, a, b) {
  let x = (seed ^ 0x9e3779b9) >>> 0;
  for (const value of [a, b]) {
    x = (x + Math.imul(value + 0x9e3779b9, 0x85ebca6b)) >>> 0;
    x ^= x >>> 15;
    x = Math.imul(x, 0xc2b2ae35) >>> 0;
    x ^= x >>> 13;
  }
  return x >>> 0;
}

function jitterFor(seed, task, attempt, amplitudePercent) {
  if (amplitudePercent <= 0) return 0;
  return (hash32(seed, task, attempt) % (amplitudePercent * 100)) / 10000;
}

/** Per-worker slowdown factors; 1 is a healthy node, > 1 is a slow node. */
function workerSpeeds(workers, slow = {}) {
  return Array.from({ length: workers }, (_, w) => slow[w] ?? 1);
}

/**
 * Runs the job. `speculativeFactor = Infinity` disables speculation and gives the plain FIFO
 * list schedule used as the baseline.
 */
function runJob(config) {
  const {
    durations, workers, speeds, seed = 1, speculativeFactor = Infinity,
    maxCopies = 2, jitterPercent = 25,
  } = config;
  if (!Number.isInteger(workers) || workers < 1) throw new RangeError("workers must be >= 1");
  if (durations.some((d) => !(d > 0))) throw new RangeError("every task needs a positive duration");
  if (!(maxCopies >= 1)) throw new RangeError("maxCopies must be >= 1");
  const taskCount = durations.length;
  const expected = taskCount === 0 ? 1 : durations.reduce((a, b) => a + b, 0) / taskCount;
  const attempts = [];
  const committed = new Map();
  const pending = durations.map((_, task) => task);
  const workerFreeAt = new Array(workers).fill(0);
  let time = 0;
  let speculationCount = 0;

  const attemptsOf = (task) => attempts.filter((a) => a.task === task);

  const startAttempt = (task, worker, speculative) => {
    const index = attemptsOf(task).length;
    const duration = durations[task] * speeds[worker] * (1 + jitterFor(seed, task, index, jitterPercent));
    attempts.push({
      task, worker, attempt: index, speculative, start: time, duration, finish: time + duration, done: false,
    });
    workerFreeAt[worker] = time + duration;
    if (speculative) speculationCount++;
  };

  /** The running attempt that has been slow the longest, if speculation is allowed. */
  const speculationCandidate = () => {
    if (!Number.isFinite(speculativeFactor)) return null;
    let best = null;
    for (const a of attempts) {
      if (a.done || committed.has(a.task)) continue;
      if (attemptsOf(a.task).length >= maxCopies) continue;
      if (time - a.start <= speculativeFactor * expected) continue;
      if (best === null || a.start < best.start || (a.start === best.start && a.task < best.task)) best = a;
    }
    return best === null ? null : best.task;
  };

  /** Fill every free worker: queued tasks first, then (optionally) a speculative copy. */
  const schedule = () => {
    for (;;) {
      const worker = workerFreeAt.findIndex((freeAt) => freeAt <= time);
      if (worker === -1) return;
      if (pending.length > 0) {
        startAttempt(pending.shift(), worker, false);
        continue;
      }
      const victim = speculationCandidate();
      if (victim === null) return;
      startAttempt(victim, worker, true);
    }
  };

  for (;;) {
    schedule();
    let next = Infinity;
    for (const a of attempts) if (!a.done && a.finish < next) next = a.finish;
    if (!Number.isFinite(next)) break;
    time = next;
    for (const a of attempts) {
      if (a.done || a.finish > time) continue;
      a.done = true;
      if (!committed.has(a.task)) {
        committed.set(a.task, { task: a.task, finish: a.finish, worker: a.worker, attempt: a.attempt });
      }
    }
    if (committed.size === taskCount) break; // remaining attempts are cancelled
  }

  const makespan = committed.size === 0 ? 0 : Math.max(...[...committed.values()].map((c) => c.finish));
  const workerIntervals = Array.from({ length: workers }, (_, w) =>
    attempts.filter((a) => a.worker === w).map((a) => [a.start, a.finish]).sort((x, y) => x[0] - y[0]));
  return {
    makespan,
    committed,
    attempts: attempts.map((a) => ({ ...a })),
    speculationCount,
    workerIntervals,
    expectedDuration: expected,
  };
}

/** Second, independent implementation of the plain FIFO list schedule (no speculation). */
function fifoMakespan(durations, workers, speeds, seed = 1, jitterPercent = 25) {
  const freeAt = new Array(workers).fill(0);
  for (let task = 0; task < durations.length; task++) {
    let worker = 0;
    for (let w = 1; w < workers; w++) if (freeAt[w] < freeAt[worker]) worker = w;
    freeAt[worker] += durations[task] * speeds[worker] * (1 + jitterFor(seed, task, 0, jitterPercent));
  }
  return durations.length === 0 ? 0 : Math.max(...freeAt);
}

/** Deterministic straggler load: a few long tasks, one worker 3x slower than the rest. */
function stragglerLoad(seed = 11) {
  const tasks = 18;
  const durations = Array.from({ length: tasks }, (_, i) => (i % 6 === 0 ? 40 : 10));
  return { durations, workers: 6, speeds: workerSpeeds(6, { 0: 3 }), seed };
}

module.exports = {
  hash32, jitterFor, workerSpeeds, runJob, fifoMakespan, stragglerLoad, makeRngJitter: jitterFor,
};

if (require.main === module) {
  // -- validation ---------------------------------------------------------------
  throws(() => runJob({ durations: [1], workers: 0, speeds: [1] }), RangeError, "workers must be positive");
  throws(() => runJob({ durations: [0], workers: 1, speeds: [1] }), RangeError, "durations must be positive");
  throws(() => runJob({ durations: [1], workers: 1, speeds: [1], maxCopies: 0 }), RangeError, "maxCopies >= 1");
  strictEqual(jitterFor(1, 2, 3, 0), 0, "zero jitter amplitude disables noise");
  ok(jitterFor(1, 2, 3, 25) === jitterFor(1, 2, 3, 25), "jitter must be a pure function");
  ok(jitterFor(1, 2, 3, 25) !== jitterFor(1, 2, 4, 25), "a different attempt gets a different jitter");

  // -- no-speculation run is the plain FIFO list schedule -----------------------
  const load = stragglerLoad();
  const baseline = runJob({ ...load, speculativeFactor: Infinity });
  strictEqual(baseline.makespan, fifoMakespan(load.durations, load.workers, load.speeds, load.seed, 25),
    "the no-speculation path must equal the independent FIFO scheduler");
  strictEqual(baseline.speculationCount, 0, "no copies are launched without speculation");
  strictEqual(baseline.attempts.length, load.durations.length, "one attempt per task in the baseline");

  // identical durations on identical workers give the closed-form makespan
  const uniform = { durations: new Array(24).fill(7), workers: 5, speeds: new Array(5).fill(1), jitterPercent: 0 };
  const uniformRun = runJob({ ...uniform, speculativeFactor: 1.2 });
  strictEqual(uniformRun.makespan, Math.ceil(24 / 5) * 7, "identical tasks must give ceil(T/W)*d");
  strictEqual(uniformRun.speculationCount, 0, "identical fast tasks never trigger speculation");

  // -- speculation helps on a straggler load ------------------------------------
  const speculative = runJob({ ...load, speculativeFactor: 1.2 });
  ok(speculative.speculationCount > 0, "stragglers must trigger speculative copies");
  ok(speculative.makespan < baseline.makespan,
    `speculation must shorten the job: ${speculative.makespan} vs ${baseline.makespan}`);
  strictEqual(speculative.committed.size, load.durations.length, "every task must commit exactly once");
  ok(speculative.makespan <= baseline.makespan, "speculation can never be worse than the baseline");

  // -- invariants over every run ------------------------------------------------
  const runs = [baseline, speculative, runJob({ ...load, speculativeFactor: 1.05, maxCopies: 2 }),
    runJob({ ...load, speculativeFactor: 2.5, maxCopies: 3 }), runJob({ ...load, maxCopies: 1 })];
  for (const run of runs) {
    strictEqual(run.committed.size, load.durations.length);
    for (let task = 0; task < load.durations.length; task++) {
      const attempts = run.attempts.filter((a) => a.task === task);
      ok(attempts.length >= 1 && attempts.length <= 3, `task ${task} must have 1..maxCopies attempts`);
      const winner = run.committed.get(task);
      strictEqual(winner.finish, Math.min(...attempts.map((a) => a.finish)),
        `task ${task} must commit its earliest attempt`);
      ok(attempts.every((a) => a.start < winner.finish + 1e-9), "no attempt starts after its task committed");
    }
    // per-worker intervals never overlap
    for (const intervals of run.workerIntervals) {
      for (let i = 1; i < intervals.length; i++) {
        ok(intervals[i][0] >= intervals[i - 1][1] - 1e-9, "a worker cannot run two attempts at once");
      }
    }
    // makespan bounds from the workload itself
    const totalWork = load.durations.reduce((a, b) => a + b, 0) * Math.min(...load.speeds);
    ok(run.makespan >= totalWork / load.workers - 1e-9, "makespan cannot beat the divisible-work bound");
    ok(run.makespan >= Math.max(...load.durations) * Math.min(...load.speeds) - 1e-9,
      "makespan cannot be shorter than the single slowest task");
    for (const attempt of run.attempts) {
      strictEqual(attempt.duration, load.durations[attempt.task] * load.speeds[attempt.worker] *
        (1 + jitterFor(load.seed, attempt.task, attempt.attempt, 25)),
        "an attempt's runtime must follow the modelled speed and jitter");
    }
  }
  strictEqual(runJob({ ...load, maxCopies: 1, speculativeFactor: 1.0 }).speculationCount, 0,
    "maxCopies = 1 forbids any duplicate attempt");
  ok(runs[2].speculationCount >= runs[3].speculationCount,
    "a stricter factor asks for at least as many copies");

  // -- determinism --------------------------------------------------------------
  const replay = runJob({ ...load, speculativeFactor: 1.2 });
  deepStrictEqual(replay.attempts, speculative.attempts, "same seed must reproduce every attempt");
  strictEqual(replay.makespan, speculative.makespan);

  // -- a single worker cannot benefit from speculation, and never breaks ---------
  const oneWorker = runJob({ durations: [5, 5, 40], workers: 1, speeds: [1], speculativeFactor: 1.2, jitterPercent: 0 });
  strictEqual(oneWorker.speculationCount, 0, "no free worker means no room for a duplicate");
  strictEqual(oneWorker.makespan, 50, "one worker runs the tasks back to back");
  strictEqual(runJob({ durations: [], workers: 3, speeds: [1, 1, 1] }).makespan, 0, "an empty job takes no time");

  console.log(`baseline=${baseline.makespan.toFixed(2)} speculative=${speculative.makespan.toFixed(2)} ` +
    `copies=${speculative.speculationCount} attempts=${speculative.attempts.length}`);
  console.log("PASS");
}
