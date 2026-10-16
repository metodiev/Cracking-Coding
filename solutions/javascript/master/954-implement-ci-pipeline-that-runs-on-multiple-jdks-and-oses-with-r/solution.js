/** Task 954 - Implement CI pipeline that runs on multiple JDKs and OSes with resource limits.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * An in-memory CI: the build matrix is expanded to legs (OS x JDK x language, with exclude/include
 * rules), each leg gets the five pipeline stages with a `needs` DAG, and a tick-driven scheduler packs
 * the jobs onto a named runner pool. Runners carry the resource limits -- class, memory budget, wall
 * limit and speed -- so a job only starts when a runner can host it (`mem <= memLimit` and its scaled
 * duration fits the wall limit); a stage whose footprint no runner can host is `unschedulable` and its
 * dependants are `skipped`, never silently green. A stage that hangs is killed at the runner's wall
 * limit and retried up to its `retries` budget. Everything runs on explicit logical ticks: no threads,
 * no sleeps, no wall clock, so a scenario replays byte-identically.
 *
 * Simplified: stages are abstract cost units instead of real toolchain invocations, and a hang is a
 * declared property of a job rather than a real timeout signal. Verified: matrix expansion with
 * excludes/includes, dependency ordering, concurrency never exceeding the pool, per-jober resource
 * limits re-derived from the schedule, unschedulable rejection with skipped dependants, timeout plus
 * retry, makespan equal to the independently computed critical path on a lightly loaded scenario, the
 * empty-matrix and single-job boundaries, and deterministic replay.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const CI_MATRIX = { os: ["ubuntu-latest", "windows-latest", "macos-latest"], jdk: [17, 21], language: ["python", "javascript"] };
const MATRIX_RULES = {
  exclude: [{ os: "macos-latest", jdk: 17 }],
  include: [{ os: "centos-latest", jdk: 17, language: "python" }], // an extra leg no dimension produced
};
/** Expand the matrix to legs; an exclude matches on the dimensions it names, include force-adds a leg. */
function expandMatrix(dimensions, { exclude = [], include = [] } = {}) {
  const keys = Object.keys(dimensions);
  const legs = [];
  const walk = (index, partial) => {
    if (index === keys.length) {
      const skip = exclude.some((rule) => Object.entries(rule).every(([k, v]) => partial[k] === v));
      if (!skip) legs.push({ id: legId(partial), ...partial });
      return;
    }
    const key = keys[index];
    for (const value of dimensions[key]) walk(index + 1, { ...partial, [key]: value });
  };
  walk(0, {});
  for (const extra of include) {
    const id = legId(extra);
    if (!legs.some((leg) => leg.id === id)) legs.push({ id, ...extra, included: true });
  }
  return legs;
}
const legId = (leg) => `ci-${leg.os}-jdk${leg.jdk}-${leg.language}`;
const STAGES = [
  { id: "checkout", needs: [], cost: 1, mem: 256, runnerClass: "standard" },
  { id: "build", needs: ["checkout"], cost: 4, mem: 512, runnerClass: "standard" },
  { id: "unit", needs: ["build"], cost: 3, mem: 512, runnerClass: "standard" },
  { id: "integration", needs: ["build"], cost: 6, mem: 2048, runnerClass: "large", retries: 1 },
  { id: "package", needs: ["unit", "integration"], cost: 2, mem: 512, runnerClass: "standard" },
];
/** Turn legs into jobs: one job per (leg, stage) with a leg-qualified dependency list. */
function buildJobs(legs, stages = STAGES) {
  const jobs = [];
  for (const leg of legs) {
    for (const stage of stages) {
      jobs.push({ id: `${leg.id}:${stage.id}`, leg: leg.id, stage: stage.id, os: leg.os, jdk: leg.jdk,
        cost: stage.cost, mem: stage.mem, runnerClass: stage.runnerClass, retries: stage.retries || 0,
        needs: stage.needs.map((need) => `${leg.id}:${need}`) });
    }
  }
  return jobs;
}
const POOL = [
  { id: "sm-1", supports: ["standard"], memLimit: 1024, timeLimit: 8, speed: 1 },
  { id: "sm-2", supports: ["standard"], memLimit: 1024, timeLimit: 8, speed: 1 },
  { id: "lg-1", supports: ["standard", "large"], memLimit: 4096, timeLimit: 8, speed: 2 },
];
const NO_LARGE = POOL.filter((runner) => !runner.supports.includes("large"));
const FAST_POOL = Array.from({ length: 6 }, (_, i) => ({ id: `fast-${i}`, supports: ["standard", "large"],
  memLimit: 8192, timeLimit: 100, speed: 1 }));
const durationOn = (job, runner) => Math.ceil(job.cost / runner.speed);
/** Runners that could ever host this job: class, memory budget and wall limit must all be satisfied. */
const runnersFor = (job, pool) => pool.filter((runner) => runner.supports.includes(job.runnerClass)
  && runner.memLimit >= job.mem && durationOn(job, runner) <= runner.timeLimit);

/**
 * Tick-driven scheduler. A pending job starts when every dependency is "ok" and a free feasible runner
 * exists; a hung attempt is killed at the runner's wall limit and retried until `retries` is exhausted.
 */
function schedule(jobs, pool, { flaky = {} } = {}) {
  const infeasible = new Set(jobs.filter((job) => runnersFor(job, pool).length === 0).map((job) => job.id));
  const state = new Map(jobs.map((job) => [job.id, { id: job.id, stage: job.stage, status: "pending",
    attempts: 0, start: null, end: null, runner: null }]));
  const blocked = (id) => ["failed", "timeout", "skipped", "unschedulable"].includes(state.get(id).status);
  const free = new Set(pool.map((runner) => runner.id));
  const running = [];
  const placements = [];
  let tick = 0;
  let maxConcurrency = 0;
  let progress = true;
  while (progress) {
    progress = false;
    for (const attempt of running.filter((r) => r.endTick === tick)) {
      const entry = state.get(attempt.job.id);
      entry.end = tick;
      entry.runner = attempt.runner.id;
      if (attempt.killed && entry.attempts <= attempt.job.retries) {
        entry.status = "pending"; // wall-limit kill with retries left: requeue for another attempt
      } else {
        entry.status = attempt.killed ? "timeout" : "ok";
      }
      running.splice(running.indexOf(attempt), 1);
      free.add(attempt.runner.id);
      progress = true;
    }
    for (const runner of pool) {
      if (!free.has(runner.id)) continue;
      const job = jobs.find((j) => state.get(j.id).status === "pending" && !running.some((r) => r.job.id === j.id)
        && !infeasible.has(j.id) && j.needs.every((need) => state.get(need).status === "ok")
        && runnersFor(j, pool).includes(runner));
      if (!job) continue;
      const entry = state.get(job.id);
      entry.attempts += 1;
      entry.start = entry.start === null ? tick : entry.start;
      const killed = hangCount(job, flaky) >= entry.attempts;
      const endTick = tick + (killed ? runner.timeLimit : durationOn(job, runner));
      running.push({ job, runner, endTick, killed });
      placements.push({ id: job.id, runner: runner.id, start: tick, end: endTick, killed, attempt: entry.attempts });
      free.delete(runner.id);
      progress = true;
    }
    maxConcurrency = Math.max(maxConcurrency, running.length);
    if (running.length) { tick += 1; progress = true; }
    for (const job of jobs) {
      const entry = state.get(job.id);
      if (entry.status !== "pending" || running.some((r) => r.job.id === job.id)) continue;
      if (infeasible.has(job.id)) { entry.status = "unschedulable"; progress = true; }
      else if (job.needs.some(blocked)) { entry.status = "skipped"; progress = true; }
    }
  }
  const statuses = [...state.values()].sort((a, b) => (a.id < b.id ? -1 : 1));
  return { statuses, placements, ticks: tick, maxConcurrency,
    makespan: Math.max(0, ...statuses.filter((s) => s.end !== null).map((s) => s.end)), pool };
}
/** How many leading attempts of this job are declared to hang. */
const hangCount = (job, flaky) => flaky[job.id] || 0;
/** Longest weighted path through the dependency DAG: the makespan of an unconstrained pool. */
function criticalPath(jobs, weight = (job) => job.cost) {
  const byId = new Map(jobs.map((job) => [job.id, job]));
  const memo = new Map();
  const depth = (id) => {
    if (memo.has(id)) return memo.get(id);
    const job = byId.get(id);
    const value = weight(job) + Math.max(0, ...job.needs.filter((n) => byId.has(n)).map(depth));
    memo.set(id, value);
    return value;
  };
  return Math.max(0, ...jobs.map((job) => depth(job.id)));
}

function main() {
  const rng = makeRng(20240954);
  const legs = expandMatrix(CI_MATRIX, MATRIX_RULES);
  const jobs = buildJobs(legs);
  const flakyJob = `${legs[rng.int(legs.length)].id}:integration`; // seeded pick of the leg we sabotage
  const full = schedule(jobs, POOL, { flaky: { [flakyJob]: 1 } });
  const byId = new Map(full.statuses.map((s) => [s.id, s]));
  // 1. Matrix: 3x2x2 legs minus the excluded pair plus the included extra leg, all ids distinct.
  assert.strictEqual(legs.length, 11, `legs ${legs.length}`);
  assert(!legs.some((leg) => leg.os === "macos-latest" && leg.jdk === 17), "exclude rule honoured");
  assert(legs.some((leg) => leg.included && leg.os === "centos-latest"), "include rule adds a leg");
  assert.strictEqual(new Set(legs.map((l) => l.id)).size, 11, "leg ids are unique");
  assert.strictEqual(jobs.length, 11 * STAGES.length, `jobs ${jobs.length}`);
  // 2. Dependency ordering: no job starts before every dependency of it has finished.
  const ordered = jobs.every((job) => {
    const entry = byId.get(job.id);
    return entry.status !== "ok" || job.needs.every((need) => byId.get(need).end <= entry.start);
  });
  assert(ordered, "every started job waits for all of its dependencies");
  // 3. Concurrency and per-runner resource limits, re-derived from the placement log.
  assert(full.maxConcurrency <= POOL.length, `concurrency ${full.maxConcurrency} <= ${POOL.length} runners`);
  const limitsHeld = full.placements.every((p) => {
    const job = jobs.find((j) => j.id === p.id);
    const runner = POOL.find((r) => r.id === p.runner);
    return runner.memLimit >= job.mem && runner.supports.includes(job.runnerClass)
      && p.end - p.start <= runner.timeLimit && runner.id === p.runner;
  });
  assert(limitsHeld, "no placement exceeds its runner's class, memory or wall limit");
  // 4. The large-only integration stage is rejected on a pool without a large runner; dependants skip.
  const small = schedule(jobs, NO_LARGE);
  const smallById = new Map(small.statuses.map((s) => [s.id, s]));
  const integrations = small.statuses.filter((s) => s.stage === "integration");
  assert(integrations.every((s) => s.status === "unschedulable") && integrations.length === 11, "integrations rejected");
  assert(small.statuses.filter((s) => s.stage === "package").every((s) => s.status === "skipped"),
    "dependants of an unschedulable stage are skipped, never green");
  assert(small.statuses.filter((s) => ["checkout", "build", "unit"].includes(s.stage)).every((s) => s.status === "ok"),
    "the rest of the pipeline still completes");
  // 5. Timeout + retry: the first attempt of the sabotaged stage hangs to the wall limit, attempt 2 passes.
  const attemptsOf = full.placements.filter((p) => p.id === flakyJob);
  assert(attemptsOf.length === 2 && attemptsOf[0].killed && !attemptsOf[1].killed, "one killed attempt, one clean");
  assert(byId.get(flakyJob).status === "ok" && byId.get(flakyJob).attempts === 2, "retry recovers the job");
  assert(attemptsOf[1].start >= attemptsOf[0].end, "the retry restarts after the kill");
  assert(smallById.get(byId.get(flakyJob).id) !== undefined, "the sabotaged leg exists in the small pool too");
  // 6. Oracle: on a lightly loaded pool the makespan equals the independently computed critical path.
  const oneLeg = legs.slice(0, 1);
  const single = schedule(buildJobs(oneLeg), FAST_POOL);
  const path = criticalPath(buildJobs(oneLeg));
  assert.deepStrictEqual(single.statuses.map((s) => s.status), Array(STAGES.length).fill("ok"), "single leg completes");
  assert(single.makespan === path && path === 13, `makespan ${single.makespan} == critical path ${path}`);
  assert(single.maxConcurrency === 2, "unit and integration overlap after build");
  // 7. Boundaries: an empty matrix schedules nothing; a lone job's makespan is its own cost.
  const empty = schedule([], POOL);
  assert(empty.statuses.length === 0 && empty.makespan === 0 && empty.placements.length === 0, "empty matrix");
  const lone = schedule([{ id: "lone", leg: "x", stage: "solo", cost: 5, mem: 128, runnerClass: "standard", needs: [], retries: 0 }], POOL);
  assert(lone.statuses[0].status === "ok" && lone.makespan === 5, `lone job makespan ${lone.makespan}`);
  // 8. Determinism: a second replay of the same scenario is byte-identical.
  const replay = schedule(jobs, POOL, { flaky: { [flakyJob]: 1 } });
  assert.deepStrictEqual(replay, full, "identical replay");
  console.log(`ci: legs=${legs.length} jobs=${jobs.length} ticks=${full.ticks} maxConcurrency=${full.maxConcurrency} `
    + `criticalPath=${path} rejected=${integrations.length}`);
  console.log("PASS");
}

module.exports = { makeRng, CI_MATRIX, MATRIX_RULES, expandMatrix, legId, STAGES, buildJobs, POOL, NO_LARGE,
  FAST_POOL, durationOn, runnersFor, schedule, criticalPath };
if (require.main === module) main();
