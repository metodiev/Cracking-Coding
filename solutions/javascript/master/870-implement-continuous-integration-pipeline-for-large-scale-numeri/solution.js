/** Task 870 - Implement continuous integration pipeline for large-scale numerical code with reproducibility. (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A deterministic, in-process CI engine for numerical code bases. The pipeline is a DAG of stages
 * (checkout -> compile -> unit tests -> numerical regression -> reproducibility gate -> benchmark ->
 * package) executed by a discrete-event simulator:
 *
 *   - time is a logical tick counter. A stage declared with `ticks: t` holds its worker for exactly t
 *     ticks, a retry costs an explicit `retryBackoff` in ticks and nothing ever sleeps or reads a clock;
 *   - the scheduler is work-conserving and deterministic: completion events are processed at the
 *     current tick, then the ready stage with the smallest topological index is dispatched to every
 *     free worker (ties by stage id), so the whole schedule is a pure function of the pipeline
 *     description and the resulting fingerprint is reproducible;
 *   - every stage input is content addressed (FNV-1a 64). A stage's cache key hashes its source
 *     revision, its configuration, the seed and the artifact hashes of its dependencies, so a warm
 *     store turns a run into zero executions and a single source change re-executes exactly the
 *     transitive closure of that stage, checked against a brute-force BFS oracle;
 *   - the reproducibility gate does real work: the same ill-conditioned reduction is evaluated with
 *     two different accumulation orders and must agree bitwise. Double accumulation always agrees; the
 *     float32-rounded variant drifts, so the gate reports the kernel as non-reproducible.
 *
 * Invariants verified: dependency ordering (no stage starts before every dependency finished), work
 * conservation (the makespan equals the critical path once workers >= DAG width and never beats it),
 * exact cache accounting, exact incremental-rebuild sets, identical schedules and artifact hashes
 * across repeated runs, failure propagation through the DAG without partial artifacts, retry budgets
 * honoured to the attempt, and rejection of cyclic, duplicate or malformed pipelines.
 *
 * What is simplified: there is no filesystem, no subprocess, no container and no wall-clock timer, so
 * "build"/"test" stages are pure deterministic functions of their inputs whose durations are declared
 * rather than measured; remote caching, artifact upload and historical flake statistics are out of
 * scope. Scale is expressed by stage count, worker count and the tick parameters.
 */
"use strict";

const { ok, throws } = require("node:assert");

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;

/** 64-bit FNV-1a over a string: the content-address function for artifacts and cache keys. */
function hashText(text) {
  let value = FNV_OFFSET;
  for (let i = 0; i < text.length; i++) {
    value ^= BigInt(text.charCodeAt(i));
    value = BigInt.asUintN(64, value * FNV_PRIME);
  }
  return value.toString(16).padStart(16, "0");
}

/** Deterministic 32-bit PRNG (mulberry32) for synthetic numerical payloads. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x2545f491;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Left-to-right accumulation, optionally rounded to float32 after every operation. */
function orderedSum(values, float32) {
  let total = 0;
  for (const value of values) total = float32 ? Math.fround(total + value) : total + value;
  return total;
}

/** Recursive pairwise (bisection) accumulation: the classic alternative reduction order. */
function pairwiseSum(values, float32) {
  const round = (v) => (float32 ? Math.fround(v) : v);
  const reduce = (lo, hi) => {
    if (hi - lo === 1) return values[lo];
    const mid = (lo + hi) >> 1;
    return round(reduce(lo, mid) + reduce(mid, hi));
  };
  return values.length === 0 ? 0 : reduce(0, values.length);
}

/** Kept for readability at the call sites: the second reduction order used by the CI gate. */
const blockedSum = pairwiseSum;

/**
 * Ill-conditioned payload for the reproducibility gate: one value at the float32 resolution threshold
 * (2^24), a deterministic shuffle of unit contributions whose exact sum is 1000, and the negated
 * threshold. In float64 every partial sum is an exact integer and both orders give 1000; with float32
 * rounding the left-to-right order silently rounds every unit contribution to zero while the pairwise
 * order keeps them, so the two orders disagree - exactly the failure the gate must catch.
 */
function makeNumericalPayload({ length = 4096, seed = 1234 } = {}) {
  if (!Number.isInteger(length) || length < 4) throw new RangeError("length must be an integer of at least 4");
  const rng = makeRng(seed);
  const middleCount = length - 2;
  const ones = (middleCount + 1000) / 2;
  if (!Number.isInteger(ones)) throw new RangeError("use an even middle length so the target sum is integral");
  const middle = [];
  for (let i = 0; i < middleCount; i++) middle.push(i < ones ? 1 : -1);
  for (let i = middleCount - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [middle[i], middle[j]] = [middle[j], middle[i]];
  }
  return [16777216, ...middle, -16777216];
}

/** "Can this reduction be reproduced bit-for-bit?" - the question the CI gate answers. */
function reductionIsReproducible(values, precision = "float64") {
  const float32 = precision === "float32";
  return Object.is(orderedSum(values, float32), blockedSum(values, float32));
}

const PRECISIONS = new Set(["float32", "float64"]);

/** Validates the pipeline and returns it in a deterministic topological order. */
function normalisePipeline(stages) {
  if (!Array.isArray(stages)) throw new TypeError("stages must be an array");
  const byId = new Map();
  for (const stage of stages) {
    if (!stage || typeof stage.id !== "string" || stage.id.length === 0) throw new TypeError("every stage needs a non-empty string id");
    if (byId.has(stage.id)) throw new Error(`duplicate stage id ${stage.id}`);
    if (!Number.isInteger(stage.ticks) || stage.ticks < 0) throw new RangeError(`stage ${stage.id}: ticks must be a non-negative integer`);
    if (stage.retries !== undefined && (!Number.isInteger(stage.retries) || stage.retries < 0)) throw new RangeError(`stage ${stage.id}: retries must be a non-negative integer`);
    const dependsOn = stage.dependsOn ?? [];
    if (!Array.isArray(dependsOn)) throw new TypeError(`stage ${stage.id}: dependsOn must be an array`);
    const precision = stage.precision ?? "float64";
    if (!PRECISIONS.has(precision)) throw new RangeError(`stage ${stage.id}: precision must be float32 or float64`);
    byId.set(stage.id, {
      id: stage.id,
      dependsOn: [...dependsOn],
      ticks: stage.ticks,
      retries: stage.retries ?? 0,
      kind: stage.kind ?? "compute",
      source: stage.source ?? stage.id,
      precision,
      failTimes: stage.failTimes ?? 0,
      failAlways: stage.failAlways === true,
    });
  }
  for (const stage of byId.values()) {
    for (const dependency of stage.dependsOn) {
      if (!byId.has(dependency)) throw new Error(`stage ${stage.id} depends on unknown stage ${dependency}`);
      if (dependency === stage.id) throw new Error(`stage ${stage.id} depends on itself`);
    }
  }
  // Kahn's algorithm; the produced order doubles as the deterministic dispatch priority.
  const indegree = new Map([...byId.keys()].map((id) => [id, 0]));
  for (const stage of byId.values()) for (const _ of stage.dependsOn) indegree.set(stage.id, indegree.get(stage.id) + 1);
  const queue = [...byId.keys()].filter((id) => indegree.get(id) === 0).sort();
  const order = [];
  while (queue.length > 0) {
    const id = queue.shift();
    order.push(id);
    const dependents = [...byId.values()].filter((s) => s.dependsOn.includes(id)).map((s) => s.id).sort();
    for (const dependent of dependents) {
      indegree.set(dependent, indegree.get(dependent) - 1);
      if (indegree.get(dependent) === 0) queue.push(dependent);
    }
  }
  if (order.length !== byId.size) throw new Error("the pipeline dependency graph contains a cycle");
  return { stages: order.map((id) => byId.get(id)), byId, index: new Map(order.map((id, i) => [id, i])), order };
}

/** Longest path through the DAG by declared ticks: the lower bound on the makespan. */
function criticalPath(stages) {
  const { stages: ordered } = normalisePipeline(stages);
  const longest = new Map();
  for (const stage of ordered) {
    const dependencyMax = stage.dependsOn.reduce((acc, id) => Math.max(acc, longest.get(id)), 0);
    longest.set(stage.id, dependencyMax + stage.ticks);
  }
  return [...longest.values()].reduce((acc, value) => Math.max(acc, value), 0);
}

/** Brute-force reachability: the set of stages that must rerun after a change to `ids`. */
function affectedStages(stages, ids) {
  const { stages: ordered } = normalisePipeline(stages);
  const affected = new Set(ids);
  let changed = true;
  while (changed) {
    changed = false;
    for (const stage of ordered) {
      if (!affected.has(stage.id) && stage.dependsOn.some((dependency) => affected.has(dependency))) {
        affected.add(stage.id);
        changed = true;
      }
    }
  }
  return affected;
}

/** Artifact payload of a stage; the reproducibility gate genuinely evaluates the reduction. */
function artifactContent(stage, cacheKey, payload) {
  if (stage.kind === "reproducibility-gate") {
    return reductionIsReproducible(payload, stage.precision)
      ? `reproducible:${stage.precision}:${cacheKey.slice(0, 8)}`
      : `nondeterministic:${stage.precision}:${cacheKey.slice(0, 8)}`;
  }
  return `${stage.kind}:${hashText(cacheKey)}`;
}

/**
 * Discrete-event scheduler with a content-addressed cache, retries and failure propagation. Returns
 * the schedule, artifacts, accounting counters and the pipeline verdict.
 */
function runPipeline(stages, {
  workers = 4, cache = new Map(), retries = 0, retryBackoff = 1, seed = 1234, payload = null,
} = {}) {
  if (!Number.isInteger(workers) || workers < 1) throw new RangeError("workers must be a positive integer");
  if (!Number.isInteger(retries) || retries < 0) throw new RangeError("retries must be a non-negative integer");
  if (!Number.isInteger(retryBackoff) || retryBackoff < 0) throw new RangeError("retryBackoff must be a non-negative integer");
  const { stages: ordered, byId } = normalisePipeline(stages);
  const data = payload ?? makeNumericalPayload({ seed });
  const status = new Map([...byId.keys()].map((id) => [id, "pending"]));
  const artifacts = new Map();
  const keys = new Map();
  const attempts = new Map([...byId.keys()].map((id) => [id, 0]));
  const readyAt = new Map([...byId.keys()].map((id) => [id, 0]));
  const startTick = new Map();
  const finishTick = new Map();
  const running = [];
  const schedule = [];
  let executedStages = 0;
  let cacheHits = 0;
  let tick = 0;
  let guard = 0;
  const cacheKeyOf = (stage) => hashText([
    stage.source, stage.kind, stage.precision, String(seed),
    ...stage.dependsOn.map((dependency) => artifacts.get(dependency).hash),
  ].join("|"));
  const dependenciesDone = (stage) => stage.dependsOn.every((dependency) => status.get(dependency) === "succeeded");
  const isReady = (stage) => status.get(stage.id) === "pending" && readyAt.get(stage.id) <= tick && dependenciesDone(stage);
  for (;;) {
    if (guard++ > 100000) throw new Error("scheduler did not converge");
    for (const task of [...running]) {
      if (task.dueTick > tick) continue;
      running.splice(running.indexOf(task), 1);
      const stage = byId.get(task.id);
      const attempt = attempts.get(stage.id);
      if (stage.failAlways || attempt <= stage.failTimes) {
        if (attempt <= stage.retries + retries) {
          status.set(stage.id, "pending");
          readyAt.set(stage.id, tick + retryBackoff);
        } else {
          status.set(stage.id, "failed");
        }
      } else {
        const content = artifactContent(stage, stage.cacheKey, data);
        status.set(stage.id, "succeeded");
        finishTick.set(stage.id, tick);
        artifacts.set(stage.id, { hash: hashText(`${stage.id}|${content}`), content });
        keys.set(stage.id, stage.cacheKey);
      }
    }
    for (const stage of ordered) {
      if (status.get(stage.id) === "pending" && stage.dependsOn.some((dependency) => ["failed", "skipped"].includes(status.get(dependency)))) {
        status.set(stage.id, "skipped");
      }
    }
    for (const stage of ordered) {
      if (running.length >= workers) break;
      if (!isReady(stage)) continue;
      stage.cacheKey = cacheKeyOf(stage);
      const cached = cache.get(stage.cacheKey);
      if (cached) {
        status.set(stage.id, "succeeded");
        startTick.set(stage.id, tick);
        finishTick.set(stage.id, tick);
        artifacts.set(stage.id, cached);
        keys.set(stage.id, stage.cacheKey);
        cacheHits += 1;
        schedule.push({ id: stage.id, start: tick, finish: tick, cached: true });
        continue;
      }
      attempts.set(stage.id, attempts.get(stage.id) + 1);
      startTick.set(stage.id, tick);
      status.set(stage.id, "running");
      running.push({ id: stage.id, dueTick: tick + stage.ticks });
      schedule.push({ id: stage.id, start: tick, finish: tick + stage.ticks, cached: false, attempt: attempts.get(stage.id) });
      executedStages += 1;
    }
    if (running.length === 0 && ![...status.values()].includes("pending")) break;
    const wakeups = running.map((task) => task.dueTick);
    for (const stage of ordered) {
      if (status.get(stage.id) === "pending" && readyAt.get(stage.id) > tick && dependenciesDone(stage)) wakeups.push(readyAt.get(stage.id));
    }
    if (wakeups.length === 0) break; // nothing can make progress: the rest of the DAG is blocked
    tick = Math.min(...wakeups);
  }
  for (const [id, artifact] of artifacts) cache.set(keys.get(id), artifact);
  const verdicts = ordered.filter((stage) => stage.kind === "reproducibility-gate").map((stage) => ({
    id: stage.id,
    reproducible: status.get(stage.id) === "succeeded" && artifacts.get(stage.id).content.startsWith("reproducible"),
    precision: stage.precision,
  }));
  return {
    status,
    artifacts,
    schedule,
    attempts,
    startTick,
    finishTick,
    makespan: [...finishTick.values()].reduce((acc, value) => Math.max(acc, value), 0),
    executedStages,
    cacheHits,
    ticks: tick,
    order: ordered.map((stage) => stage.id),
    verdicts,
    succeeded: [...status.values()].every((value) => value === "succeeded") && status.size > 0,
  };
}

const madeIdle = (run, id) => run.status.get(id);

const standardPipeline = (sourceRevision, precision = "float64") => [
  { id: "checkout", ticks: 1, kind: "checkout", source: sourceRevision },
  { id: "compile", ticks: 3, kind: "compile", dependsOn: ["checkout"] },
  { id: "unit-tests", ticks: 2, kind: "test", dependsOn: ["compile"] },
  { id: "numerics-regression", ticks: 4, kind: "numerical-regression", dependsOn: ["checkout"] },
  { id: "reproducibility-gate", ticks: 3, kind: "reproducibility-gate", dependsOn: ["numerics-regression"], precision },
  { id: "benchmark", ticks: 5, kind: "benchmark", dependsOn: ["compile"] },
  { id: "package", ticks: 2, kind: "package", dependsOn: ["unit-tests", "reproducibility-gate", "benchmark"] },
];

const withSource = (pipeline, id, source) => pipeline.map((stage) => (stage.id === id ? { ...stage, source } : stage));

const executedIds = (run) => new Set(run.schedule.filter((entry) => !entry.cached).map((entry) => entry.id));

const scheduleFingerprint = (run) => hashText(run.schedule.map((entry) => `${entry.id}@${entry.start}-${entry.finish}`).join(";"));

function main() {
  const diagnostics = [];
  const payload = makeNumericalPayload({ seed: 7 });

  // 1. Topology: the plan is a valid topological order and the DAG is validated up front.
  const pipeline = standardPipeline("rev-1");
  const plan = normalisePipeline(pipeline);
  const position = new Map(plan.order.map((id, i) => [id, i]));
  diagnostics.push(`pipeline: ${plan.stages.length} stages, order ${plan.order.join(" -> ")}`);
  ok(plan.stages.every((stage) => stage.dependsOn.every((dependency) => position.get(dependency) < position.get(stage.id))), "the plan must be a topological order");
  ok(plan.order[0] === "checkout" && plan.order[plan.order.length - 1] === "package", "the plan must start at checkout and end at package");
  ok(criticalPath(pipeline) === 11, `the critical path must be 11 ticks (got ${criticalPath(pipeline)})`);

  // 2. Cold run: everything executes once, dependencies are respected, the makespan dominates the path.
  const store = new Map();
  const cold = runPipeline(pipeline, { workers: 4, cache: store, payload });
  const dependencyOrdered = cold.schedule.every((entry) => plan.byId.get(entry.id).dependsOn
    .every((dependency) => cold.finishTick.get(dependency) <= entry.start));
  diagnostics.push(`cold run: makespan ${cold.makespan} ticks, ${cold.executedStages} executions, ${cold.cacheHits} cache hits`);
  ok(cold.succeeded, "the cold run must succeed");
  ok(cold.executedStages === pipeline.length, "a cold run must execute every stage exactly once");
  ok(cold.cacheHits === 0, "a cold store cannot produce hits");
  ok(dependencyOrdered, "no stage may start before all of its dependencies finished");
  ok(cold.makespan >= criticalPath(pipeline), "the makespan cannot beat the critical path");

  // 3. Worker scaling: monotone, effective, and exactly the critical path with a wide pool.
  const widths = [1, 2, 4, 8].map((workers) => {
    const run = runPipeline(pipeline, { workers, cache: new Map(), payload });
    return { workers, makespan: run.makespan, executed: run.executedStages };
  });
  diagnostics.push(`scaling: ${widths.map((w) => `${w.workers}w -> ${w.makespan}t`).join(", ")}`);
  for (let i = 1; i < widths.length; i++) {
    ok(widths[i].makespan <= widths[i - 1].makespan, `more workers must not increase the makespan (${widths[i - 1].workers} -> ${widths[i].workers})`);
  }
  ok(widths[0].makespan > widths[widths.length - 1].makespan, "parallelism must actually reduce the makespan");
  ok(widths[widths.length - 1].makespan === criticalPath(pipeline), "with a wide enough pool the makespan must equal the critical path");
  ok(widths[0].makespan === pipeline.reduce((acc, stage) => acc + stage.ticks, 0), "a single worker must take the sum of all stage durations");

  // 4. Warm cache: zero executions, bit-identical artifacts.
  const warm = runPipeline(pipeline, { workers: 4, cache: store, payload });
  const identicalArtifacts = pipeline.every((stage) => cold.artifacts.get(stage.id).hash === warm.artifacts.get(stage.id).hash);
  diagnostics.push(`warm run: ${warm.executedStages} executions, ${warm.cacheHits} hits, artifacts identical: ${identicalArtifacts}, makespan ${warm.makespan}`);
  ok(warm.executedStages === 0 && warm.cacheHits === pipeline.length, "a warm store must serve every stage from cache");
  ok(identicalArtifacts, "cached artifacts must be bit-identical to freshly built ones");
  ok(warm.succeeded && warm.makespan === 0, "a fully cached run must succeed instantly");

  // 5. Incremental rebuild after a source change: exactly the transitive closure of that stage.
  const revised = standardPipeline("rev-1").map((stage) => (stage.id === "numerics-regression" ? { ...stage, source: "rev-2" } : stage));
  const expected = affectedStages(revised, ["numerics-regression"]);
  const incremental = runPipeline(revised, { workers: 4, cache: new Map(store), payload });
  const executedNow = executedIds(incremental);
  diagnostics.push(`source change: executed ${[...executedNow].sort().join(",")} (expected ${[...expected].sort().join(",")})`);
  ok(executedNow.size === expected.size && [...expected].every((id) => executedNow.has(id)), "an incremental run must re-execute exactly the affected subgraph");
  ok(executedNow.has("package"), "downstream consumers must be rebuilt");

  // 6. A change in one branch leaves the independent branch cached.
  const local = withSource(standardPipeline("rev-1"), "unit-tests", "rev-unit");
  const localExpected = affectedStages(local, ["unit-tests"]);
  const localRun = runPipeline(local, { workers: 4, cache: new Map(store), payload });
  const localExecuted = executedIds(localRun);
  diagnostics.push(`local change: executed ${[...localExecuted].sort().join(",")} (expected ${[...localExpected].sort().join(",")})`);
  ok(localExecuted.size === localExpected.size && [...localExpected].every((id) => localExecuted.has(id)), "a local change must re-execute exactly its downstream cone");
  ok(!localExecuted.has("numerics-regression"), "an unrelated branch must stay cached");

  // 7. The reproducibility gate does real work: summation order matters in float32, not in float64.
  const doubleStable = Object.is(orderedSum(payload, false), blockedSum(payload, false));
  const float32Drift = Math.abs(orderedSum(payload, true) - blockedSum(payload, true));
  diagnostics.push(`reduction: double sum ${orderedSum(payload, false)}, orders agree=${doubleStable}, float32 drift ${float32Drift}`);
  ok(doubleStable, "double accumulation must be independent of the summation order");
  ok(orderedSum(payload, false) === 1000 && blockedSum(payload, false) === 1000, "the exact payload sum must be the planted 1000");
  ok(!Object.is(orderedSum(payload, true), blockedSum(payload, true)), "float32 accumulation must expose the order dependence");
  ok(Math.abs(float32Drift) >= 100, `the float32 drift must be visible (${float32Drift})`);
  const strictGate = runPipeline(standardPipeline("rev-1", "float64"), { workers: 4, cache: new Map(), payload });
  const looseGate = runPipeline(standardPipeline("rev-1", "float32"), { workers: 4, cache: new Map(), payload });
  ok(strictGate.verdicts.length === 1 && strictGate.verdicts[0].reproducible, "the deterministic kernel must pass the gate");
  ok(looseGate.verdicts.length === 1 && !looseGate.verdicts[0].reproducible, "the float32 kernel must be reported non-reproducible");
  ok(strictGate.status.get("reproducibility-gate") === "succeeded" && looseGate.status.get("reproducibility-gate") === "succeeded", "the gate stage itself must run to completion either way");

  // 8. Failure propagation: a permanently failing stage blocks its dependents and publishes nothing.
  const broken = standardPipeline("rev-1").map((stage) => (stage.id === "compile" ? { ...stage, failAlways: true } : stage));
  const brokenRun = runPipeline(broken, { workers: 4, cache: new Map(), payload });
  diagnostics.push(`failing compile: ${[...brokenRun.status].map(([id, value]) => `${id}=${value}`).join(", ")}`);
  ok(!brokenRun.succeeded, "a failing stage must fail the pipeline");
  ok(madeIdle(brokenRun, "compile") === "failed", "the failing stage must be reported as failed");
  ok(madeIdle(brokenRun, "package") === "skipped", "dependents of a failure must be skipped, not run");
  ok(!brokenRun.artifacts.has("package"), "no artifact may be published for a skipped stage");
  ok(brokenRun.artifacts.has("checkout"), "unaffected stages must still publish their artifacts");

  // 9. Retries: a flaky stage succeeds within its budget and exhausts it otherwise.
  const flaky = standardPipeline("rev-1").map((stage) => (stage.id === "benchmark" ? { ...stage, failTimes: 2, retries: 3 } : stage));
  const retried = runPipeline(flaky, { workers: 4, cache: new Map(), payload });
  diagnostics.push(`flaky benchmark: attempts=${retried.attempts.get("benchmark")}, makespan ${retried.makespan}, retry backoff respected`);
  ok(retried.succeeded, "a flaky stage inside its retry budget must eventually succeed");
  ok(retried.attempts.get("benchmark") === 3, `the flaky stage must take exactly three attempts (got ${retried.attempts.get("benchmark")})`);
  const noBudget = runPipeline(flaky.map((stage) => (stage.id === "benchmark" ? { ...stage, retries: 1 } : stage)), { workers: 4, cache: new Map(), payload });
  ok(!noBudget.succeeded && noBudget.attempts.get("benchmark") === 2, "an exhausted retry budget must fail after the last allowed attempt");
  const backoffSlow = runPipeline(flaky, { workers: 4, cache: new Map(), payload, retryBackoff: 5 });
  ok(backoffSlow.makespan === retried.makespan + 2 * (5 - 1), `backoff must add exactly its declared ticks (${backoffSlow.makespan} vs ${retried.makespan})`);

  // 10. Determinism: schedules and artifacts are pure functions of the description.
  const first = runPipeline(pipeline, { workers: 3, cache: new Map(), payload, seed: 99 });
  const second = runPipeline(pipeline, { workers: 3, cache: new Map(), payload, seed: 99 });
  diagnostics.push(`schedule fingerprint ${scheduleFingerprint(first)} (repeat ${scheduleFingerprint(second)})`);
  ok(scheduleFingerprint(first) === scheduleFingerprint(second), "the schedule must be reproducible");
  ok(pipeline.every((stage) => first.artifacts.get(stage.id).hash === second.artifacts.get(stage.id).hash), "artifact hashes must be reproducible");
  const differentSeed = runPipeline(pipeline, { workers: 3, cache: new Map(), payload, seed: 100 });
  ok(pipeline.every((stage) => first.artifacts.get(stage.id).hash !== differentSeed.artifacts.get(stage.id).hash), "a different seed must change the artifacts");
  const differentWorkers = runPipeline(pipeline, { workers: 2, cache: new Map(), payload, seed: 99 });
  ok(pipeline.every((stage) => first.artifacts.get(stage.id).hash === differentWorkers.artifacts.get(stage.id).hash), "worker count must not change the artifacts");

  // 11. Boundaries: empty pipeline, single stage, zero-tick stage.
  const empty = runPipeline([], { workers: 2 });
  ok(empty.makespan === 0 && empty.executedStages === 0 && !empty.succeeded, "an empty pipeline must be a no-op");
  const single = runPipeline([{ id: "solo", ticks: 7 }], { workers: 8 });
  ok(single.makespan === 7 && single.executedStages === 1, "a single stage must run for its declared duration");
  const instant = runPipeline([{ id: "instant", ticks: 0 }], { workers: 1 });
  ok(instant.makespan === 0 && instant.executedStages === 1, "a zero-tick stage must be scheduled exactly once and finish immediately");
  const zeroWorkers = runPipeline([{ id: "solo", ticks: 2 }], { workers: 1 });
  ok(zeroWorkers.makespan === 2, "a single worker must serialise the stages");

  // 12. Malformed pipelines are rejected instead of producing a made-up result.
  throws(() => normalisePipeline([{ id: "a", ticks: 0, dependsOn: ["b"] }]), /unknown stage/);
  throws(() => normalisePipeline([{ id: "a", ticks: 0, dependsOn: ["b"] }, { id: "b", ticks: 0, dependsOn: ["a"] }]), /cycle/);
  throws(() => normalisePipeline([{ id: "a", ticks: 0 }, { id: "a", ticks: 0 }]), /duplicate stage id/);
  throws(() => normalisePipeline([{ id: "a", ticks: -1 }]), RangeError);
  throws(() => normalisePipeline([{ id: "a", ticks: 0, retries: 0.5 }]), RangeError);
  throws(() => normalisePipeline([{ id: "a", ticks: 0, precision: "float16" }]), RangeError);
  throws(() => normalisePipeline([{ id: "", ticks: 0 }]), TypeError);
  throws(() => normalisePipeline("nope"), TypeError);
  throws(() => runPipeline([{ id: "a", ticks: 0 }], { workers: 0 }), RangeError);
  throws(() => runPipeline([{ id: "a", ticks: 0 }], { retries: -1 }), RangeError);
  throws(() => runPipeline([{ id: "a", ticks: 0 }], { retryBackoff: 1.5 }), RangeError);

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  hashText,
  makeRng,
  orderedSum,
  blockedSum,
  makeNumericalPayload,
  reductionIsReproducible,
  normalisePipeline,
  criticalPath,
  affectedStages,
  artifactContent,
  runPipeline,
  standardPipeline,
  withSource,
  executedIds,
  scheduleFingerprint,
};

if (require.main === module) main();
