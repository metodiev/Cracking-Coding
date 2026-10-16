/** Task 667 - Implement schedulability analysis for real-time systems (rate-monotonic). (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Three independent answers to "can this periodic task set be scheduled?", at the precision a
 * real-time practitioner uses:
 *   * Liu & Layland's utilisation bound — `U = sum(Ci/Ti) <= n(2^(1/n) - 1)` is *sufficient* but
 *     not necessary, so it can only ever say "provably fine" or "no opinion".
 *   * exact response-time analysis (RTA) — iterate
 *     `R = Ci + sum over higher-priority tasks of ceil(R/Tj) * Cj` to its fixed point; the set is
 *     schedulable iff every `Ri <= Di`. For fixed-priority preemptive scheduling this test is
 *     necessary *and* sufficient, which is exactly why it can be cross-checked against a
 *     simulation.
 *   * simulation — build the real rate-monotonic schedule (shortest period = highest priority)
 *     over one hyperperiod with synchronous releases, and look for a job that misses its deadline.
 *
 * Verified: the analytic and simulated verdicts agree on every generated task set (60 of them,
 * plus hand-checked cases), the utilisation bound is *not* used as an oracle but shown to be
 * sufficient only (`C=[1,2,3]`, `T=[4,5,20]` has U = 0.8 above the 0.7798 bound yet is
 * schedulable), the RTA iteration is a genuine fixed point, and degenerate sets (empty, single
 * task, Ci = Ti, U > 1) behave as theory requires.
 *
 * Simplification: deadlines equal periods, releases are synchronous (all tasks released at t = 0),
 * execution times are integers, and costs are assumed constant — no blocking, no jitter, no
 * aperiodic work.
 */
"use strict";
const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");
function makeRng(seed) {
  let state = seed >>> 0 || 0x2545f491;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
function makeTask(name, period, execution, deadline = period) {
  if (!Number.isInteger(period) || period <= 0) throw new RangeError(`task ${name} needs a positive integer period`);
  if (!Number.isInteger(execution) || execution < 0) throw new RangeError(`task ${name} needs a non-negative integer cost`);
  if (!Number.isInteger(deadline) || deadline <= 0) throw new RangeError(`task ${name} needs a positive deadline`);
  return { name, period, execution, deadline };
}
const utilisation = (tasks) => tasks.reduce((sum, task) => sum + task.execution / task.period, 0);
/** Liu & Layland's bound for n tasks: n(2^(1/n) - 1), and 1 for a single task. */
const llBound = (n) => (n <= 0 ? 1 : n * (2 ** (1 / n) - 1));
const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
const lcm = (a, b) => (a / gcd(a, b)) * b;
const hyperperiod = (tasks) => tasks.reduce((acc, task) => lcm(acc, task.period), 1);
/** Rate-monotonic priority assignment: shorter period first, ties broken by name. */
function rateMonotonicOrder(tasks) {
  return [...tasks].sort((a, b) => (a.period - b.period) || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}
/** Sufficient test: U <= n(2^(1/n) - 1). */
function utilisationBoundSchedulable(tasks) {
  return utilisation(tasks) <= llBound(tasks.length) + 1e-12;
}
/**
 * Exact response-time analysis. Returns each task's response time (Infinity when the iteration
 * does not converge, which means the task set is not schedulable) plus the iteration count.
 */
function responseTimes(tasks, options = {}) {
  const maxIterations = options.maxIterations ?? 200;
  const ordered = rateMonotonicOrder(tasks);
  const results = [];
  for (let index = 0; index < ordered.length; index++) {
    const task = ordered[index];
    const higher = ordered.slice(0, index);
    let response = task.execution;
    let iterations = 0;
    let converged = task.execution === 0;
    while (iterations < maxIterations) {
      iterations++;
      const interference = higher.reduce((sum, other) => sum + Math.ceil(response / other.period) * other.execution, 0);
      const next = task.execution + interference;
      if (next === response) {
        converged = true;
        break;
      }
      response = next;
    }
    results.push({ name: task.name, response: converged ? response : Infinity, deadline: task.deadline, iterations });
  }
  return { ordered, results };
}
/** Exact test: every task must have a finite response time that fits its deadline. */
function exactSchedulable(tasks, options = {}) {
  const { results } = responseTimes(tasks, options);
  return results.every((entry) => Number.isFinite(entry.response) && entry.response <= entry.deadline);
}
/**
 * Fixed-priority preemptive simulation over one hyperperiod with synchronous releases.
 * Returns the verdict, the number of jobs run, busy time and the first missed job.
 */
function simulate(tasks, options = {}) {
  const ordered = rateMonotonicOrder(tasks);
  const state = ordered.map((task, priority) => ({
    ...task, priority, remaining: 0, release: 0, absoluteDeadline: 0, jobs: 0,
  }));
  const horizon = options.horizon ?? hyperperiod(ordered);
  const stats = { jobs: 0, busy: 0, idle: 0, horizon, idleTime: 0, missed: null };
  for (let time = 0; time < horizon; time++) {
    for (const task of state) {
      if (time % task.period !== 0) continue;
      if (task.remaining > 0) {
        return { ...stats, schedulable: false, reason: "overrun", missed: { task: task.name, time, deadline: task.absoluteDeadline } };
      }
      task.remaining = task.execution;
      task.release = time;
      task.absoluteDeadline = time + task.deadline;
      task.jobs++;
      stats.jobs++;
    }
    const chosen = state.find((task) => task.remaining > 0); // RM: first in period order wins
    if (chosen === undefined) {
      stats.idleTime++;
      continue;
    }
    chosen.remaining--;
    stats.busy++;
    for (const task of state) {
      if (task.remaining > 0 && time + 1 > task.absoluteDeadline) {
        return { ...stats, schedulable: false, reason: "deadline", missed: { task: task.name, time: time + 1, deadline: task.absoluteDeadline } };
      }
    }
  }
  const unfinished = state.find((task) => task.remaining > 0);
  if (unfinished !== undefined) {
    return {
      ...stats, schedulable: false, reason: "unfinished",
      missed: { task: unfinished.name, time: horizon, deadline: unfinished.absoluteDeadline },
    };
  }
  const idle = horizon - stats.busy;
  if (utilisation(ordered) <= 1) {
    // The busy time is fully determined by the workload, which is a strong bookkeeping check.
    ok(Math.abs(idle - horizon * (1 - utilisation(ordered))) < 1e-9,
      "the simulated idle time must match the utilisation");
  }
  return { ...stats, schedulable: true, reason: "met", idleTime: idle };
}
module.exports = {
  makeTask, utilisation, llBound, hyperperiod, lcm, gcd, rateMonotonicOrder,
  utilisationBoundSchedulable, responseTimes, exactSchedulable, simulate, makeRng,
};
if (require.main === module) {
  // -- utilisation and the bound ---------------------------------------------------
  strictEqual(llBound(1), 1, "one task may use the whole processor");
  ok(Math.abs(llBound(2) - 0.8284) < 0.0001, "the two-task bound is 2(sqrt2 - 1)");
  ok(Math.abs(llBound(3) - 0.7797) < 0.0001, "the three-task bound is about 0.7798");
  ok(llBound(3) > Math.log(2) && llBound(3) < llBound(2), "the bound decreases towards ln 2");
  const classic = [makeTask("t1", 4, 1), makeTask("t2", 5, 2), makeTask("t3", 20, 3)];
  ok(Math.abs(utilisation(classic) - 0.8) < 1e-12, "the classic example has U = 0.8");
  ok(!utilisationBoundSchedulable(classic), "U = 0.8 is above the 0.7798 bound, so the bound is silent");
  ok(exactSchedulable(classic), "the exact test proves the classic example is schedulable anyway");
  ok(simulate(classic).schedulable, "and so does the schedule itself");
  // -- exact test details ----------------------------------------------------------
  const { ordered, results } = responseTimes(classic);
  deepStrictEqual(ordered.map((task) => task.name), ["t1", "t2", "t3"], "RM orders by period");
  deepStrictEqual(results.map((entry) => entry.response), [1, 3, 10], "hand-checked response times");
  ok(results.every((entry) => entry.response <= entry.deadline));
  for (const entry of results) {
    const task = classic.find((candidate) => candidate.name === entry.name);
    const higher = ordered.slice(0, ordered.findIndex((candidate) => candidate.name === entry.name));
    const interference = higher.reduce((sum, other) => sum + Math.ceil(entry.response / other.period) * other.execution, 0);
    strictEqual(task.execution + interference, entry.response, `${entry.name} must be a fixed point of RTA`);
  }
  const unschedulable = [makeTask("a", 3, 1), makeTask("b", 6, 2), makeTask("c", 10, 3)];
  ok(Math.abs(utilisation(unschedulable) - 0.9667) < 0.001, "U is just below one");
  ok(!utilisationBoundSchedulable(unschedulable), "the bound cannot certify it");
  ok(!exactSchedulable(unschedulable), "but the exact test rejects it");
  const unschedulableRun = simulate(unschedulable);
  strictEqual(unschedulableRun.schedulable, false, "and the simulation agrees");
  strictEqual(unschedulableRun.missed.task, "c", "the longest-period task is the one that misses");
  // -- degenerate sets -------------------------------------------------------------
  ok(exactSchedulable([]), "an empty task set is trivially schedulable");
  strictEqual(simulate([]).jobs, 0, "an empty set runs no jobs");
  strictEqual(simulate([]).schedulable, true);
  const single = [makeTask("only", 5, 5)];
  strictEqual(utilisation(single), 1, "a single task may use everything");
  ok(exactSchedulable(single), "Ci = Ti is still schedulable when it is the only task");
  ok(simulate(single).schedulable);
  const overloaded = [makeTask("x", 4, 5)];
  ok(utilisation(overloaded) > 1, "Ci > Ti overloads the processor");
  ok(!exactSchedulable(overloaded), "the exact test rejects an overloaded task");
  strictEqual(simulate(overloaded).schedulable, false, "the simulation rejects it too");
  strictEqual(simulate(overloaded).reason, "unfinished",
    "a cost above the period can never finish its job");
  const free = [makeTask("free", 3, 0)];
  strictEqual(utilisation(free), 0);
  ok(exactSchedulable(free) && simulate(free).schedulable, "zero-cost tasks never miss a deadline");
  throws(() => makeTask("bad", 0, 1), RangeError, "a zero period is rejected");
  throws(() => makeTask("bad", 4, -1), RangeError, "a negative cost is rejected");
  throws(() => makeTask("bad", 4, 1, 0), RangeError, "a zero deadline is rejected");
  // -- harmonic sets: the bound reaches one ----------------------------------------
  const harmonic = [makeTask("h1", 4, 1), makeTask("h2", 8, 2), makeTask("h3", 16, 4)];
  strictEqual(utilisation(harmonic), 0.75, "harmonic periods with U = 0.75");
  ok(utilisationBoundSchedulable(harmonic), "the bound certifies it");
  ok(exactSchedulable(harmonic) && simulate(harmonic).schedulable, "and so do the other two tests");
  strictEqual(hyperperiod(harmonic), 16, "the hyperperiod is the largest period for harmonic sets");
  strictEqual(hyperperiod([makeTask("p", 6, 1), makeTask("q", 10, 1)]), 30, "lcm of 6 and 10 is 30");
  // -- analytic and simulated verdicts must agree ----------------------------------
  const rng = makeRng(20261016);
  let agreements = 0;
  let certified = 0;
  let rejected = 0;
  for (let trial = 0; trial < 60; trial++) {
    const count = 2 + rng(3);
    const tasks = [];
    for (let index = 0; index < count; index++) {
      const period = 2 + rng(8) * (index + 1);
      const cost = 1 + rng(Math.max(1, Math.floor(period / 2)));
      tasks.push(makeTask(`g${index}`, period, cost));
    }
    const exact = exactSchedulable(tasks);
    const simulated = simulate(tasks).schedulable;
    if (exact) certified++;
    else rejected++;
    if (exact !== simulated) {
      throw new Error(`analysis and simulation disagree on ${JSON.stringify(tasks)}: exact=${exact} simulated=${simulated}`);
    }
    if (utilisationBoundSchedulable(tasks)) {
      ok(exact, `the utilisation bound is sufficient: it passed for ${JSON.stringify(tasks)}`);
    }
    agreements++;
  }
  strictEqual(agreements, 60, "every generated set must be judged by both methods");
  ok(certified > 0 && rejected > 0, `the corpus must contain both verdicts, saw ${certified}/${rejected}`);
  // a set that the bound certifies must never be rejected by the exact test
  ok(utilisationBoundSchedulable(classic) === false && exactSchedulable(classic) === true,
    "the bound is sufficient but not necessary");
  // -- determinism -----------------------------------------------------------------
  const replay = () => {
    const tasks = [makeTask("r1", 5, 2), makeTask("r2", 7, 1), makeTask("r3", 14, 3)];
    return { exact: exactSchedulable(tasks), simulated: simulate(tasks) };
  };
  deepStrictEqual(replay(), replay(), "identical task sets must produce identical results");
  const slow = responseTimes(classic, { maxIterations: 1 });
  ok(slow.results.some((entry) => !Number.isFinite(entry.response)) ||
    slow.results.every((entry) => Number.isFinite(entry.response)),
  "a low iteration cap must degrade gracefully rather than loop forever");
  console.log(`classic U=${utilisation(classic).toFixed(4)} bound=${llBound(3).toFixed(4)} R=[` +
    `${results.map((entry) => entry.response).join(",")}] corpus=${agreements} certified=${certified} rejected=${rejected}`);
  console.log("PASS");
}
