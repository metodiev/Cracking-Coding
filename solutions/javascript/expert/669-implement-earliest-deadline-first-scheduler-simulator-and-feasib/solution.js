/**
 * Task 669 - Implement earliest deadline first scheduler simulator and feasibility check.
 * (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Preemptive uniprocessor simulator for synchronous periodic tasks (C, T, D) plus the three classic
 * feasibility tests, each implemented independently so they can be cross-checked against each other:
 *   - `simulate(tasks, horizon, policy)` releases one job per period, then on every tick runs the
 *     runnable job with the earliest absolute deadline (EDF) or the job of the task with the shortest
 *     period (rate monotonic, the classic fixed-priority scheme). A job is declared missed as soon as
 *     its absolute deadline passes with work remaining, or when it completes after its deadline.
 *   - `dbfTest` is the processor demand criterion: with dbf(L) = sum_i max(0, floor((L-D_i)/T_i)+1)*C_i
 *     EDF is feasible iff dbf(L) <= L at every candidate L = D_i + k*T_i.
 *   - `intervalDemandTest` is the brute-force form of the same criterion: it enumerates *all* integer
 *     intervals [a, b] inside the horizon and adds up every job whose release lies in the interval and
 *     whose deadline also lies in it, flagging the first overload demand > b - a. Being strictly
 *     stronger than dbf (it also moves the left endpoint), it is a genuine independent check.
 *   - `utilizationFeasible` compares sum C_i/T_i <= 1 exactly in integers (over the hyperperiod
 *     common denominator), so no floating point rounding can flip a borderline case.
 *
 * Documented results: EDF and the dbf test always agree with the simulation; sum C_i/T_i <= 1 is
 * equivalent for implicit deadlines (D = T) but only necessary when D < T, shown by the classic
 * counter-example (C=3,T=5,D=3) with (C=2,T=6,D=2); EDF dominates rate monotonic - on
 * (C=2,T=5,D=5) with (C=4,T=7,D=7) the utilization test and EDF succeed while RM misses a deadline;
 * and whenever EDF misses a deadline the interval demand test reports an overload, i.e. the miss is
 * forced and not a scheduling artefact.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed669n;

/** Deterministic 64-bit splitmix64, used to generate the random task sets. */
class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// --------------------------------------------------------------------------------- task algebra
const periodic = (id, C, T, D = T) => ({ id, C, T, D });

function gcd(a, b) {
  while (b) {
    const r = a % b;
    a = b;
    b = r;
  }
  return a;
}

const lcm = (a, b) => (a / gcd(a, b)) * b;
const hyperperiod = (tasks) => tasks.reduce((acc, t) => lcm(acc, t.T), 1);
const utilization = (tasks) => tasks.reduce((acc, t) => acc + t.C / t.T, 0);

/** Exact test sum C_i/T_i <= 1, evaluated in integers over the hyperperiod denominator. */
function utilizationFeasible(tasks) {
  const L = hyperperiod(tasks);
  const demand = tasks.reduce((acc, t) => acc + t.C * (L / t.T), 0);
  return { feasible: demand <= L, demand, period: L };
}

/** One job per period inside [0, horizon); `deadline` is absolute (release + D). */
function jobList(tasks, horizon) {
  const jobs = [];
  for (const t of tasks) {
    for (let r = 0; r < horizon; r += t.T) {
      jobs.push({ id: `${t.id}@${r}`, task: t, release: r, deadline: r + t.D, remaining: t.C,
        missed: false, finish: null });
    }
  }
  return jobs;
}

function edfOrder(a, b) {
  return (a.deadline - b.deadline) || (a.release - b.release) || (a.task.id < b.task.id ? -1 : 1);
}

function rmOrder(a, b) {
  return (a.task.T - b.task.T) || (a.release - b.release) || (a.task.id < b.task.id ? -1 : 1);
}

/**
 * Tick-by-tick simulation. `policy` is "edf" (earliest absolute deadline first) or "rm" (shortest
 * period first). Jobs keep running after their deadline so the miss list stays meaningful, but each
 * miss is recorded only once.
 */
function simulate(tasks, horizon, policy = "edf") {
  const jobs = jobList(tasks, horizon);
  const misses = [];
  const order = policy === "rm" ? rmOrder : edfOrder;
  let busy = 0;
  let idle = 0;
  const record = (job) => {
    job.missed = true;
    misses.push({ id: job.id, release: job.release, deadline: job.deadline, finish: job.finish });
  };
  for (let t = 0; t < horizon; t++) {
    for (const job of jobs) {
      if (!job.missed && job.remaining > 0 && job.deadline <= t && job.deadline <= horizon) record(job);
    }
    const ready = jobs.filter((j) => j.release <= t && j.remaining > 0);
    if (ready.length === 0) {
      idle += 1;
      continue;
    }
    ready.sort(order);
    const chosen = ready[0];
    chosen.remaining -= 1;
    busy += 1;
    if (chosen.remaining === 0) {
      chosen.finish = t + 1;
      if (chosen.finish > chosen.deadline && !chosen.missed) record(chosen);
    }
  }
  const judged = jobs.filter((j) => j.deadline <= horizon);
  return { misses, missed: misses.length > 0, busy, idle, judged: judged.length,
    completedJobs: judged.filter((j) => j.remaining === 0).length };
}

/** Processor demand: work of the jobs of `tasks` whose deadline falls in (0, L]. */
function dbf(tasks, L) {
  return tasks.reduce((acc, t) =>
    (L >= t.D ? acc + (Math.floor((L - t.D) / t.T) + 1) * t.C : acc), 0);
}

/** Demand bound criterion; dbf only changes value at L = D_i + k*T_i, so those L are sufficient. */
function dbfTest(tasks, limit) {
  let tightest = { L: 0, demand: 0, slack: Infinity };
  for (const t of tasks) {
    for (let L = t.D; L <= limit; L += t.T) {
      const demand = dbf(tasks, L);
      if (demand > L) return { feasible: false, witness: { L, demand, slack: L - demand } };
      if (L - demand < tightest.slack) tightest = { L, demand, slack: L - demand };
    }
  }
  return { feasible: true, tightest };
}

/** Brute-force demand criterion over every integer interval inside the horizon. */
function intervalDemandTest(tasks, horizon) {
  const jobs = [];
  for (const t of tasks) {
    for (let r = 0; r < horizon; r += t.T) jobs.push({ release: r, deadline: r + t.D, C: t.C });
  }
  let intervals = 0;
  for (let a = 0; a <= horizon; a++) {
    for (let b = a + 1; b <= horizon; b++) {
      intervals += 1;
      const demand = jobs.reduce((acc, j) =>
        (j.release >= a && j.deadline <= b ? acc + j.C : acc), 0);
      if (demand > b - a) return { feasible: false, witness: { a, b, demand, length: b - a }, intervals };
    }
  }
  return { feasible: true, intervals };
}

// -------------------------------------------------------------------------------------- cases
/** Classic constrained-deadline case: U = 0.933 < 1 yet infeasible because D < T. */
const constrainedInfeasible = () => [periodic("t1", 3, 5, 3), periodic("t2", 2, 6, 2)];

/** Rate monotonic misses, EDF succeeds, on the same implicit-deadline set (U = 0.971). */
const rmBeatsEdf = () => [periodic("t1", 2, 5, 5), periodic("t2", 4, 7, 7)];

/** Deterministic random implicit-deadline sets (small periods, so hyperperiods stay tiny). */
function randomSets(seed, count) {
  const rng = new SplitMix64(seed);
  const sets = [];
  for (let s = 0; s < count; s++) {
    const n = 2 + rng.nextInt(2);
    const tasks = [];
    for (let k = 0; k < n; k++) {
      const T = 3 + rng.nextInt(7);
      const C = 1 + rng.nextInt(Math.min(T, 4));
      tasks.push(periodic(`t${k}`, C, T));
    }
    sets.push(tasks);
  }
  return sets;
}

function main() {
  // (1) Task algebra and the exact utilization test.
  assert.strictEqual(hyperperiod([periodic("a", 1, 4), periodic("b", 1, 6)]), 12);
  assert.strictEqual(hyperperiod(rmBeatsEdf()), 35);
  assert.ok(Math.abs(utilization(rmBeatsEdf()) - 34 / 35) < 1e-12, "U = 2/5 + 4/7 = 34/35");
  assert.strictEqual(utilizationFeasible(rmBeatsEdf()).feasible, true);
  assert.strictEqual(utilizationFeasible([periodic("a", 2, 4), periodic("b", 3, 4)]).feasible, false,
    "U = 1.25 > 1 is infeasible exactly");
  assert.strictEqual(utilizationFeasible([periodic("a", 1, 3), periodic("b", 2, 6)]).feasible, true,
    "U = 2/3 exactly on the boundary");

  // (2) EDF on an implicit-deadline set: no misses, all judged jobs done, CPU idle when slack exists.
  const rmSet = rmBeatsEdf();
  const horizon = 2 * hyperperiod(rmSet);
  const edf = simulate(rmSet, horizon, "edf");
  const rm = simulate(rmSet, horizon, "rm");
  assert.strictEqual(edf.missed, false, "EDF meets every deadline on the 34/35 utilization set");
  assert.strictEqual(edf.completedJobs, edf.judged, "every judged job completed");
  assert.strictEqual(edf.busy + edf.idle, horizon, "every tick is either busy or idle");
  assert.ok(edf.idle > 0, `U < 1 leaves ${edf.idle} idle ticks in the horizon`);
  assert.strictEqual(rm.missed, true, "documented: rate monotonic misses a deadline on the same set");
  assert.ok(rm.misses.some((m) => m.id.startsWith("t2")),
    `the miss is t2's long job: ${JSON.stringify(rm.misses[0])}`);
  assert.strictEqual(dbfTest(rmSet, horizon).feasible, true, "the demand bound test agrees with EDF");

  // (3) The constrained-deadline counter-example: U < 1 but infeasible.
  const bad = constrainedInfeasible();
  const badHorizon = 2 * hyperperiod(bad);
  const badEdf = simulate(bad, badHorizon, "edf");
  assert.strictEqual(utilizationFeasible(bad).feasible, true,
    "documented: U = 3/5 + 2/6 = 0.933 <= 1, so the utilization test alone would accept this set");
  assert.strictEqual(badEdf.missed, true, "yet EDF misses a deadline, so U <= 1 is not sufficient");
  assert.strictEqual(dbfTest(bad, badHorizon).feasible, false, "the demand bound test rejects it");
  const badInterval = intervalDemandTest(bad, badHorizon);
  assert.strictEqual(badInterval.feasible, false,
    `documented: the brute-force interval test finds the overload ${JSON.stringify(badInterval.witness)}`);
  assert.ok(badInterval.witness.demand > badInterval.witness.length,
    "the witness really is an overloaded interval");

  // (4) Random sets: simulation, dbf, interval demand and the utilization test must agree.
  const sets = randomSets(DEFAULT_SEED, 10);
  let feasibleCount = 0;
  let intervalChecks = 0;
  for (const tasks of sets) {
    const h = 2 * hyperperiod(tasks);
    const run = simulate(tasks, h, "edf");
    const demand = dbfTest(tasks, h);
    const interval = intervalDemandTest(tasks, h);
    const util = utilizationFeasible(tasks);
    intervalChecks += interval.intervals;
    if (!run.missed) feasibleCount += 1;
    assert.strictEqual(demand.feasible, !run.missed,
      `dbf and simulation disagree on ${JSON.stringify(tasks)}`);
    assert.strictEqual(interval.feasible, !run.missed,
      `brute-force intervals and simulation disagree on ${JSON.stringify(tasks)}`);
    assert.strictEqual(util.feasible, !run.missed,
      `implicit deadlines: U <= 1 must decide ${JSON.stringify(tasks)}`);
    if (!run.missed) assert.strictEqual(run.completedJobs, run.judged, "feasible runs finish all work");
    if (run.missed) {
      // EDF is optimal, so a miss must be forced: the interval criterion has to fail as well.
      assert.ok(!demand.feasible && !interval.feasible, "a miss implies an unavoidable overload");
    }
  }
  assert.ok(feasibleCount > 0 && feasibleCount < sets.length,
    `${feasibleCount} of ${sets.length} random sets are feasible, so both verdicts are exercised`);
  assert.ok(intervalChecks > 20000, `the brute-force test examined ${intervalChecks} intervals`);

  // (5) EDF dominates rate monotonic: whenever RM is feasible, EDF is too.
  let rmFeasibleCount = 0;
  for (const tasks of sets) {
    const h = 2 * hyperperiod(tasks);
    const rmRun = simulate(tasks, h, "rm");
    const edfRun = simulate(tasks, h, "edf");
    if (!rmRun.missed) {
      rmFeasibleCount += 1;
      assert.strictEqual(edfRun.missed, false, "RM feasible implies EDF feasible (EDF optimality)");
    }
  }
  assert.ok(rmFeasibleCount > 0,
    `the dominance check was exercised: ${rmFeasibleCount} of ${sets.length} sets are RM feasible`);

  // (6) Determinism and a boundary case: a single task at exactly 100% utilization never idles.
  assert.deepStrictEqual(simulate(rmSet, horizon, "edf"), simulate(rmSet, horizon, "edf"));
  assert.deepStrictEqual(simulate(bad, badHorizon, "rm"), simulate(bad, badHorizon, "rm"));
  const saturated = [periodic("solo", 3, 3)];
  const sat = simulate(saturated, 12, "edf");
  assert.strictEqual(sat.missed, false, "C = T = 3 is feasible");
  assert.strictEqual(sat.idle, 0, "at 100% utilization EDF never idles");
  assert.strictEqual(sat.busy, 12, "and it executes exactly the released work");
  const overloaded = simulate([periodic("solo", 4, 3)], 12, "edf");
  assert.strictEqual(overloaded.missed, true, "C > T is always infeasible");
  assert.strictEqual(overloaded.idle, 0, "an overloaded task never lets the CPU idle");
}

module.exports = { SplitMix64, DEFAULT_SEED, periodic, gcd, lcm, hyperperiod, utilization,
  utilizationFeasible, jobList, edfOrder, rmOrder, simulate, dbf, dbfTest, intervalDemandTest,
  constrainedInfeasible, rmBeatsEdf, randomSets };

if (require.main === module) {
  main();
  console.log("PASS");
}
