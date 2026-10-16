/** Task 856 - Implement runtime adaptive operator re-optimization and replanning.
 *  (Tier: Master, CTCI Ch 14 — Databases)
 *
 * Model. A two-table equi-join (users JOIN orders ON users.id = orders.user_id) is planned and then
 * executed in-process. The optimizer picks one of four physical alternatives — hash join or index
 * nested loop, each with either input as the build/outer side — from a cost model driven by cardinality
 * estimates. The estimates can be stale: the statistics were collected when the users table had 10
 * rows, it now has 400. That is what makes an up-front plan go wrong at runtime:
 *
 *   * a hash join builds a hash table of `build` rows; beyond HASH_MEMORY rows it spills and every
 *     spilled row costs SPILL_PENALTY additional work;
 *   * an index nested loop pays PER_ROW_INDEX + log2(inner) for every outer row, so a large outer side
 *     is ruinous;
 *   * a nested loop also pays for every pair it probes.
 *
 * The adaptive executor re-plans at two points. (1) Before executing it samples the head of both inputs
 * and derives the real cardinality (cardinality feedback); if the sample contradicts the estimate by
 * at least `threshold`, the plan is recomputed. (2) At runtime a hash join observes the exact build
 * cardinality the moment its build phase finishes; if that contradicts the assumption and the planner
 * would now pick a different alternative, the executor discards the work done so far, pays
 * `reoptimizeCost` and restarts from the last pipeline boundary with the corrected plan. Restarts are
 * capped by `maxReplans`, so every query terminates.
 *
 * What is simplified: one process, no disk, one fixed join shape, a rule-of-thumb estimator instead of
 * histograms, head-of-input sampling instead of block sampling, restart-on-replan instead of stateful
 * operator migration, and work counted in abstract units rather than in time. One seeded mulberry32
 * stream builds the fixture — no timers, no threads, no wall clock, no randomness elsewhere.
 *
 * Invariants verified:
 *   safety      adaptive, stale-static and clairvoyant plans return exactly the brute-force result for
 *               every query and every statistics profile; a restart neither duplicates nor drops rows.
 *   liveness    re-plans never exceed maxReplans, attempts equal replans + 1, and every run terminates
 *               with a complete result.
 *   economy     a re-plan happens exactly when the observed cardinality deviates beyond `threshold`
 *               *and* the planner would choose a different alternative; inside the tolerance nothing
 *               is re-planned and the recorded work is exactly sampling + one plan execution.
 *   cost bound  with stale statistics the adaptive total work is far below the stale-static total,
 *               stays within a bounded factor of the clairvoyant (true-statistics) plan, and the
 *               reported work equals an independently computed cost.
 *   determinism identical inputs produce identical plans, decisions, work counts and rows.
 */
"use strict";

const assert = require("node:assert");

const ALGORITHMS = ["hash", "indexNestedLoop"];
const SIDES = ["users", "orders"];
const HASH_MEMORY = 450; // rows that fit in the build-side hash table before spilling
const SPILL_PENALTY = 4; // extra work units per spilled build row
const PER_ROW_INDEX = 12; // random-access penalty of one index probe

/** mulberry32 — the only source of randomness (fixture construction). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = (rand, arr) => arr[Math.floor(rand() * arr.length)];

function evalPredicate(pred, row) {
  if (pred === null) return true;
  if (pred.kind === "and") return pred.children.every((child) => evalPredicate(child, row));
  if (pred.kind === "or") return pred.children.some((child) => evalPredicate(child, row));
  const value = row[pred.column];
  if (pred.op === "==") return value === pred.value;
  if (pred.op === "<") return value < pred.value;
  if (pred.op === "<=") return value <= pred.value;
  if (pred.op === ">") return value > pred.value;
  if (pred.op === ">=") return value >= pred.value;
  throw new Error(`unknown operator ${pred.op}`);
}

/** Deterministic fixture: `users` and `orders`, shuffled so the head of each table is representative. */
function makeEnvironment({ users = 400, orders = 600, staleRows = 10, seed = 856 } = {}) {
  if (!Number.isInteger(users) || users < 1) throw new RangeError("users must be >= 1");
  if (!Number.isInteger(orders) || orders < 1) throw new RangeError("orders must be >= 1");
  if (!Number.isInteger(staleRows) || staleRows < 1) throw new RangeError("staleRows must be >= 1");
  const rand = mulberry32(seed);
  const userRows = [];
  for (let id = 0; id < users; id++) userRows.push({ id, dept: ["eng", "sales", "hr"][id % 3], age: 20 + ((id * 7) % 40) });
  const orderRows = [];
  for (let id = 0; id < orders; id++) orderRows.push({ id, user_id: (id * 11) % users, amount: 1 + ((id * 13) % 120) });
  const shuffle = (rows) => {
    for (let i = rows.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = rows[i];
      rows[i] = rows[j];
      rows[j] = tmp;
    }
    return rows;
  };
  shuffle(userRows);
  shuffle(orderRows);
  return { tables: { users: userRows, orders: orderRows }, stats: { users: staleRows, orders }, totals: { users, orders } };
}

/** Rule-of-thumb selectivity — what an optimizer without histograms would assume. */
function assumedSelectivity(pred) {
  if (pred === null) return 1;
  if (pred.kind === "and") return pred.children.reduce((acc, child) => acc * assumedSelectivity(child), 1);
  if (pred.kind === "or") {
    return pred.children.reduce((acc, child) => {
      const other = assumedSelectivity(child);
      return acc + other - acc * other;
    }, 0);
  }
  return pred.op === "==" ? 0.2 : 0.5;
}

/** What the optimizer believes each filtered input contains. */
function estimatedCardinalities(query, stats) {
  return {
    users: Math.max(1, Math.round(stats.users * assumedSelectivity(query.filterA))),
    orders: Math.max(1, Math.round(stats.orders * assumedSelectivity(query.filterB))),
  };
}

/** Exact work of a plan for true cardinalities — shared by the planner and the independent verifier. */
function planCost(plan, users, orders) {
  if (!ALGORITHMS.includes(plan.algorithm)) throw new Error(`unknown algorithm ${plan.algorithm}`);
  const out = Math.min(users, orders);
  if (plan.algorithm === "hash") {
    if (!SIDES.includes(plan.build)) throw new Error(`unknown build side ${plan.build}`);
    const build = plan.build === "users" ? users : orders;
    const probe = plan.build === "users" ? orders : users;
    return users + orders + build + probe + out + Math.max(0, build - HASH_MEMORY) * SPILL_PENALTY;
  }
  if (!SIDES.includes(plan.outer)) throw new Error(`unknown outer side ${plan.outer}`);
  const outer = plan.outer === "users" ? users : orders;
  const inner = plan.outer === "users" ? orders : users;
  return users + orders + outer * (PER_ROW_INDEX + Math.log2(inner + 1)) + out;
}

/** Enumerates all four physical alternatives and returns the cheapest one. */
function choosePlan(users, orders) {
  const candidates = [
    { algorithm: "hash", build: "users" },
    { algorithm: "hash", build: "orders" },
    { algorithm: "indexNestedLoop", outer: "users" },
    { algorithm: "indexNestedLoop", outer: "orders" },
  ];
  let best = null;
  for (const candidate of candidates) {
    const cost = planCost(candidate, users, orders);
    if (best === null || cost < best.cost) best = { ...candidate, cost };
  }
  return { ...best, users, orders };
}

/** Ratio by which two cardinalities differ, always >= 1. */
function deviation(expected, observed) {
  const a = Math.max(1, expected);
  const b = Math.max(1, observed);
  return a > b ? a / b : b / a;
}

const sideLabel = (plan) => `${plan.algorithm}/${plan.build || plan.outer}`;

/** Merged row: the order side first, then the user side, so `id` is always the user id. */
const merge = (orderRow, userRow) => ({ ...orderRow, ...userRow });

/** Real join execution for the chosen plan; the result never depends on the plan. */
function runPlan(plan, userRows, orderRows) {
  if (plan.algorithm === "hash") {
    const buildRows = plan.build === "users" ? userRows : orderRows;
    const probeRows = plan.build === "users" ? orderRows : userRows;
    const buildKey = plan.build === "users" ? "id" : "user_id";
    const probeKey = plan.build === "users" ? "user_id" : "id";
    const buckets = new Map();
    for (const row of buildRows) {
      const key = row[buildKey];
      const list = buckets.get(key);
      if (list === undefined) buckets.set(key, [row]);
      else list.push(row);
    }
    const rows = [];
    for (const row of probeRows) {
      for (const other of buckets.get(row[probeKey]) || []) {
        rows.push(plan.build === "users" ? merge(row, other) : merge(other, row));
      }
    }
    return rows;
  }
  const outerRows = plan.outer === "users" ? userRows : orderRows;
  const innerRows = plan.outer === "users" ? orderRows : userRows;
  const outerKey = plan.outer === "users" ? "id" : "user_id";
  const innerKey = plan.outer === "users" ? "user_id" : "id";
  const index = new Map();
  for (const row of innerRows) {
    const key = row[innerKey];
    const list = index.get(key);
    if (list === undefined) index.set(key, [row]);
    else list.push(row);
  }
  const rows = [];
  for (const row of outerRows) {
    for (const other of index.get(row[outerKey]) || []) {
      rows.push(plan.outer === "users" ? merge(other, row) : merge(row, other));
    }
  }
  return rows;
}

/** Brute-force oracle: plain nested loops over the two filtered inputs. */
function bruteForceJoin(userRows, orderRows) {
  const rows = [];
  for (const user of userRows) {
    for (const order of orderRows) {
      if (user.id === order.user_id) rows.push(merge(order, user));
    }
  }
  return rows;
}

function rowSignature(rows) {
  return rows
    .map((row) =>
      JSON.stringify(
        Object.keys(row)
          .sort()
          .map((key) => [key, row[key]]),
      ),
    )
    .sort()
    .join("|");
}

function filteredInputs(query, env) {
  return {
    userRows: env.tables.users.filter((row) => evalPredicate(query.filterA, row)),
    orderRows: env.tables.orders.filter((row) => evalPredicate(query.filterB, row)),
  };
}

/**
 * The adaptive executor. Phase 1 samples both inputs (cardinality feedback) and may re-plan before any
 * real work happens. Phase 2 executes the plan; a hash join checks the real build cardinality as its
 * build phase completes and may swap to a different alternative, in which case the work done so far is
 * lost and `reoptimizeCost` is charged for the planner call.
 */
class AdaptiveExecutor {
  constructor({ sampleSize = 40, threshold = 2, maxReplans = 2, reoptimizeCost = 50, adaptive = true } = {}) {
    if (!Number.isInteger(sampleSize) || sampleSize < 1) throw new RangeError("sampleSize must be >= 1");
    if (!(threshold > 1)) throw new RangeError("threshold must be > 1");
    if (!Number.isInteger(maxReplans) || maxReplans < 0) throw new RangeError("maxReplans must be >= 0");
    if (!(reoptimizeCost >= 0)) throw new RangeError("reoptimizeCost must be >= 0");
    this.sampleSize = sampleSize;
    this.threshold = threshold;
    this.maxReplans = maxReplans;
    this.reoptimizeCost = reoptimizeCost;
    this.adaptive = adaptive;
  }

  /** Head-of-input sampling: measured selectivity x true table size, the cardinality-feedback estimate. */
  sample(query, env) {
    const observed = {};
    let rowsRead = 0;
    const sides = [
      { name: "users", filter: query.filterA },
      { name: "orders", filter: query.filterB },
    ];
    for (const side of sides) {
      const rows = env.tables[side.name];
      const size = Math.min(this.sampleSize, rows.length);
      let matched = 0;
      for (let i = 0; i < size; i++) if (evalPredicate(side.filter, rows[i])) matched += 1;
      rowsRead += size;
      observed[side.name] = Math.max(1, Math.round((size === 0 ? 0 : matched / size) * env.totals[side.name]));
    }
    return { observed, work: rowsRead };
  }

  execute(query, env, { adaptive = this.adaptive } = {}) {
    if (query === null || typeof query !== "object") throw new TypeError("query required");
    const { userRows, orderRows } = filteredInputs(query, env);
    const truth = { users: userRows.length, orders: orderRows.length };
    const estimated = estimatedCardinalities(query, env.stats);
    let plan = choosePlan(estimated.users, estimated.orders);
    const initialPlan = sideLabel(plan);
    let work = 0;
    let replans = 0;
    let samplingWork = 0;
    let wastedWork = 0;
    const decisions = [];

    if (adaptive) {
      const sample = this.sample(query, env);
      samplingWork = sample.work;
      work += sample.work;
      const ratio = Math.max(deviation(estimated.users, sample.observed.users), deviation(estimated.orders, sample.observed.orders));
      const candidate = choosePlan(sample.observed.users, sample.observed.orders);
      if (ratio >= this.threshold && replans < this.maxReplans && sideLabel(candidate) !== sideLabel(plan)) {
        plan = candidate;
        replans += 1;
        work += this.reoptimizeCost;
        decisions.push({ when: "sampling", ratio, observed: sample.observed, replanned: true, plan: sideLabel(plan) });
      } else {
        decisions.push({ when: "sampling", ratio, observed: sample.observed, replanned: false, plan: sideLabel(plan) });
      }
    }

    let attempts = 0;
    for (;;) {
      attempts += 1;
      if (plan.algorithm === "hash") {
        const expectedBuild = plan.build === "users" ? plan.users : plan.orders;
        const realBuild = truth[plan.build];
        const ratio = deviation(expectedBuild, realBuild);
        if (ratio >= this.threshold && replans < this.maxReplans) {
          const corrected = {
            users: plan.build === "users" ? realBuild : plan.users,
            orders: plan.build === "orders" ? realBuild : plan.orders,
          };
          const candidate = choosePlan(corrected.users, corrected.orders);
          if (sideLabel(candidate) !== sideLabel(plan)) {
            // The build phase already read both inputs and filled the hash table.
            wastedWork += truth.users + truth.orders + realBuild;
            work += truth.users + truth.orders + realBuild;
            work += this.reoptimizeCost;
            replans += 1;
            plan = candidate;
            decisions.push({ when: "build-phase", ratio, observed: realBuild, replanned: true, plan: sideLabel(plan) });
            continue;
          }
          decisions.push({ when: "build-phase", ratio, observed: realBuild, replanned: false, plan: sideLabel(plan) });
        } else {
          decisions.push({ when: "build-phase", ratio, observed: realBuild, replanned: false, plan: sideLabel(plan) });
        }
      }
      work += planCost(plan, truth.users, truth.orders);
      break;
    }

    return {
      rows: runPlan(plan, userRows, orderRows),
      work,
      replans,
      attempts,
      samplingWork,
      wastedWork,
      plannerWork: replans * this.reoptimizeCost,
      decisions,
      plan,
      initialPlan,
      truth,
      estimated,
    };
  }
}

/** Static baseline: the plan chosen from the (possibly stale) statistics, executed exactly once. */
function staticExecution(query, env) {
  const { userRows, orderRows } = filteredInputs(query, env);
  const estimated = estimatedCardinalities(query, env.stats);
  const plan = choosePlan(estimated.users, estimated.orders);
  return {
    rows: runPlan(plan, userRows, orderRows),
    work: planCost(plan, userRows.length, orderRows.length),
    plan,
    replans: 0,
    truth: { users: userRows.length, orders: orderRows.length },
  };
}

/** Clairvoyant baseline: the plan chosen with the true cardinalities, executed exactly once. */
function oracleExecution(query, env) {
  const { userRows, orderRows } = filteredInputs(query, env);
  const plan = choosePlan(userRows.length, orderRows.length);
  return { rows: runPlan(plan, userRows, orderRows), work: planCost(plan, userRows.length, orderRows.length), plan };
}

function generateWorkload(rand, count) {
  const userFilters = [
    { kind: "cmp", column: "dept", op: "==", value: "eng" },
    { kind: "cmp", column: "dept", op: "==", value: "hr" },
    { kind: "cmp", column: "age", op: ">", value: 30 },
    { kind: "cmp", column: "age", op: "<=", value: 35 },
    { kind: "and", children: [{ kind: "cmp", column: "dept", op: "==", value: "eng" }, { kind: "cmp", column: "age", op: ">", value: 40 }] },
    { kind: "cmp", column: "id", op: "<", value: 12 },
  ];
  const orderFilters = [
    { kind: "cmp", column: "amount", op: ">", value: 20 },
    { kind: "cmp", column: "amount", op: ">", value: 90 },
    { kind: "cmp", column: "amount", op: "<=", value: 30 },
    { kind: "cmp", column: "amount", op: ">", value: 60 },
    null,
  ];
  const workload = [];
  for (let i = 0; i < count; i++) workload.push({ filterA: pick(rand, userFilters), filterB: pick(rand, orderFilters) });
  return workload;
}

function selfCheck() {
  // --- validation ------------------------------------------------------------------------------------
  assert.throws(() => new AdaptiveExecutor({ sampleSize: 0 }), /sampleSize/);
  assert.throws(() => new AdaptiveExecutor({ threshold: 1 }), /threshold/);
  assert.throws(() => new AdaptiveExecutor({ maxReplans: -1 }), /maxReplans/);
  assert.throws(() => new AdaptiveExecutor({ reoptimizeCost: -1 }), /reoptimizeCost/);
  assert.throws(() => planCost({ algorithm: "merge", build: "users" }, 1, 1), /unknown algorithm/);
  assert.throws(() => planCost({ algorithm: "hash", build: "sideways" }, 1, 1), /unknown build side/);
  assert.throws(() => planCost({ algorithm: "indexNestedLoop", outer: "sideways" }, 1, 1), /unknown outer side/);

  // --- cost model and plan choice ---------------------------------------------------------------------
  assert.strictEqual(planCost({ algorithm: "hash", build: "users" }, 300, 20), 300 + 20 + 300 + 20 + 20);
  assert.strictEqual(planCost({ algorithm: "hash", build: "users" }, 600, 20), 600 + 20 + 600 + 20 + 20 + 150 * SPILL_PENALTY);
  assert.strictEqual(planCost({ algorithm: "hash", build: "orders" }, 600, 20), 600 + 20 + 20 + 600 + 20);
  assert.strictEqual(planCost({ algorithm: "indexNestedLoop", outer: "orders" }, 2, 480), 482 + 480 * (PER_ROW_INDEX + Math.log2(3)) + 2);

  const bigUsers = choosePlan(133, 480);
  assert.strictEqual(bigUsers.algorithm, "hash");
  assert.strictEqual(bigUsers.build, "users", "the smaller input is built");
  const tinyUsers = choosePlan(2, 480);
  assert.strictEqual(tinyUsers.algorithm, "indexNestedLoop");
  assert.strictEqual(tinyUsers.outer, "users", "a tiny outer side favours an index nested loop");
  assert.strictEqual(choosePlan(600, 500).build, "orders", "the smaller side spills less, so it is built");
  assert.ok(planCost({ algorithm: "hash", build: "orders" }, 600, 500) < planCost({ algorithm: "hash", build: "users" }, 600, 500));
  assert.strictEqual(choosePlan(100, 20).build, "users", "equal costs are broken deterministically in favour of the first candidate");
  for (const [users, orders] of [[1, 1], [10, 400], [400, 10], [450, 451], [133, 480]]) {
    const chosen = choosePlan(users, orders);
    for (const candidate of [
      { algorithm: "hash", build: "users" },
      { algorithm: "hash", build: "orders" },
      { algorithm: "indexNestedLoop", outer: "users" },
      { algorithm: "indexNestedLoop", outer: "orders" },
    ]) {
      assert.ok(chosen.cost <= planCost(candidate, users, orders) + 1e-9, `plan choice must be minimal for ${users},${orders}`);
    }
    assert.ok(chosen.cost > 0);
  }
  assert.strictEqual(deviation(10, 40), 4);
  assert.strictEqual(deviation(40, 10), 4);
  assert.strictEqual(deviation(0, 0), 1);

  // --- fixtures ----------------------------------------------------------------------------------------
  const rand = mulberry32(856);
  const workload = generateWorkload(rand, 40);
  const staleEnv = makeEnvironment({ staleRows: 10, seed: 856 });
  const freshEnv = makeEnvironment({ staleRows: 400, seed: 856 });
  assert.strictEqual(staleEnv.totals.users, 400);
  assert.strictEqual(staleEnv.tables.users.length, 400);
  assert.strictEqual(staleEnv.tables.orders.length, 600);
  const executor = new AdaptiveExecutor({ sampleSize: 40, threshold: 2, maxReplans: 2, reoptimizeCost: 50 });

  // --- safety: every plan shape returns the brute-force result ------------------------------------------
  let verified = 0;
  const planShapes = new Set();
  for (const query of workload) {
    const { userRows, orderRows } = filteredInputs(query, staleEnv);
    const expected = rowSignature(bruteForceJoin(userRows, orderRows));
    const adaptive = executor.execute(query, staleEnv);
    const staticRun = staticExecution(query, staleEnv);
    const oracle = oracleExecution(query, staleEnv);
    assert.strictEqual(rowSignature(adaptive.rows), expected, `adaptive rows for ${JSON.stringify(query)}`);
    assert.strictEqual(rowSignature(staticRun.rows), expected);
    assert.strictEqual(rowSignature(oracle.rows), expected);
    assert.strictEqual(adaptive.rows.length, bruteForceJoin(userRows, orderRows).length, "a restart must not duplicate or drop rows");
    planShapes.add(sideLabel(adaptive.plan));
    planShapes.add(sideLabel(staticRun.plan));
    verified += 1;
  }
  assert.strictEqual(verified, workload.length);
  assert.ok(planShapes.size >= 2, `the workload must exercise several plan shapes: ${[...planShapes]}`);

  // --- economy: re-plan exactly when the observation breaks the tolerance (and the plan changes) ---------
  let insideTolerance = 0;
  let outsideTolerance = 0;
  let changedBySampling = 0;
  for (const query of workload) {
    const estimated = estimatedCardinalities(query, staleEnv.stats);
    const sample = executor.sample(query, staleEnv);
    const run = executor.execute(query, staleEnv);
    const sampledRatio = Math.max(deviation(estimated.users, sample.observed.users), deviation(estimated.orders, sample.observed.orders));
    const sampledPlan = sideLabel(choosePlan(sample.observed.users, sample.observed.orders));
    const estimatedPlan = sideLabel(choosePlan(estimated.users, estimated.orders));
    if (sampledRatio < executor.threshold || sampledPlan === estimatedPlan) {
      assert.strictEqual(run.decisions[0].replanned, false, `no sampling re-plan expected for ratio ${sampledRatio.toFixed(2)}`);
      insideTolerance += 1;
    } else {
      assert.strictEqual(run.decisions[0].replanned, true, `a sampling re-plan is expected for ratio ${sampledRatio.toFixed(2)}`);
      assert.strictEqual(run.initialPlan, estimatedPlan);
      changedBySampling += 1;
      outsideTolerance += 1;
    }
    assert.ok(run.decisions.every((decision) => typeof decision.plan === "string"));
    assert.strictEqual(run.decisions[run.decisions.length - 1].plan, sideLabel(run.plan), "the last decision describes the plan that ran");
    if (run.replans > 0) {
      assert.ok(run.decisions.some((decision) => decision.replanned), "at least one decision must be a re-plan");
      assert.notStrictEqual(run.initialPlan, sideLabel(run.plan));
    }
  }
  assert.ok(insideTolerance > 0 && outsideTolerance > 0, `both paths must be exercised (${insideTolerance}/${outsideTolerance})`);
  assert.ok(changedBySampling > 0, "sampling must repair some plans");

  // --- liveness: bounded re-plans, attempts = replans + 1 -------------------------------------------------
  let repaired = 0;
  for (const query of workload) {
    const run = executor.execute(query, staleEnv);
    assert.ok(run.replans <= executor.maxReplans, `replans ${run.replans} exceeded the budget`);
    const sampling = run.decisions.filter((decision) => decision.when === "sampling" && decision.replanned).length;
    const buildPhase = run.decisions.filter((decision) => decision.when === "build-phase" && decision.replanned).length;
    assert.strictEqual(run.replans, sampling + buildPhase, "every re-plan is recorded as a decision");
    assert.strictEqual(run.attempts, 1 + buildPhase, "each build-phase re-plan restarts exactly one attempt");
    assert.ok(run.decisions.length <= 2, "at most a sampling decision and a build-phase decision");
    if (run.replans > 0) repaired += 1;
  }
  assert.ok(repaired > 0, "stale statistics must force repairs");
  const capped = new AdaptiveExecutor({ maxReplans: 0 }).execute(workload[0], staleEnv);
  assert.strictEqual(capped.replans, 0, "maxReplans = 0 disables re-planning entirely");
  assert.strictEqual(capped.attempts, 1);
  assert.strictEqual(capped.initialPlan, sideLabel(capped.plan));
  assert.strictEqual(rowSignature(capped.rows), rowSignature(staticExecution(workload[0], staleEnv).rows));

  // --- exact accounting when nothing was re-planned --------------------------------------------------------
  const stableQuery = workload.find((query) => {
    const estimated = estimatedCardinalities(query, freshEnv.stats);
    const sample = executor.sample(query, freshEnv);
    const ratio = Math.max(deviation(estimated.users, sample.observed.users), deviation(estimated.orders, sample.observed.orders));
    return ratio < executor.threshold;
  });
  assert.ok(stableQuery !== undefined, "the fresh profile must contain a tolerant query");
  const stable = executor.execute(stableQuery, freshEnv);
  assert.strictEqual(stable.replans, 0);
  assert.strictEqual(stable.wastedWork, 0);
  assert.strictEqual(stable.work, stable.samplingWork + planCost(stable.plan, stable.truth.users, stable.truth.orders), "work = sampling + one plan execution");
  const nonAdaptive = new AdaptiveExecutor({ adaptive: false });
  assert.strictEqual(nonAdaptive.execute(stableQuery, freshEnv).work, planCost(stable.plan, stable.truth.users, stable.truth.orders), "a non-adaptive run pays no sampling cost");

  // --- cost bound: stale statistics are repaired, regret stays bounded --------------------------------------
  let adaptiveWork = 0;
  let staticWork = 0;
  let oracleWork = 0;
  for (const query of workload) {
    const adaptive = executor.execute(query, staleEnv);
    const staticRun = staticExecution(query, staleEnv);
    const oracle = oracleExecution(query, staleEnv);
    adaptiveWork += adaptive.work;
    staticWork += staticRun.work;
    oracleWork += oracle.work;
    // Exact cost accounting: work = sampling + wasted restart work + planner calls + one final plan.
    assert.strictEqual(
      adaptive.work,
      adaptive.samplingWork + adaptive.wastedWork + adaptive.plannerWork + planCost(adaptive.plan, adaptive.truth.users, adaptive.truth.orders),
      `cost accounting for ${JSON.stringify(query)}`,
    );
    assert.ok(adaptive.work >= planCost(adaptive.plan, adaptive.truth.users, adaptive.truth.orders), "work must cover the final plan at least once");
    assert.strictEqual(adaptive.wastedWork > 0, adaptive.decisions.some((decision) => decision.when === "build-phase" && decision.replanned));
    assert.ok(oracle.work <= staticRun.work + 1e-9, "the clairvoyant plan can never be worse than the stale one");
  }
  assert.ok(adaptiveWork < staticWork, `adaptive ${adaptiveWork} must beat stale-static ${staticWork}`);
  assert.ok(adaptiveWork < staticWork * 0.85, `the benefit must be substantial (ratio ${(adaptiveWork / staticWork).toFixed(3)})`);
  assert.ok(adaptiveWork <= oracleWork * 1.4, `regret must stay bounded (ratio ${(adaptiveWork / oracleWork).toFixed(3)})`);
  const repairedQuery = workload.find((query) => executor.execute(query, staleEnv).replans > 0);
  const repairedRun = executor.execute(repairedQuery, staleEnv);
  const repairedStatic = staticExecution(repairedQuery, staleEnv);
  assert.ok(repairedRun.work < repairedStatic.work, `a repaired query must get cheaper: ${repairedRun.work} vs ${repairedStatic.work}`);
  assert.notStrictEqual(sideLabel(repairedRun.plan), sideLabel(repairedStatic.plan), "the repair must change the physical plan");

  // --- fresh statistics: correct plans are left alone, only the sampling cost is added ------------------------
  let freshAdaptiveWork = 0;
  let freshStaticWork = 0;
  let freshReplans = 0;
  for (const query of workload) {
    const adaptive = executor.execute(query, freshEnv);
    const staticRun = staticExecution(query, freshEnv);
    freshAdaptiveWork += adaptive.work;
    freshStaticWork += staticRun.work;
    freshReplans += adaptive.replans;
    assert.strictEqual(rowSignature(adaptive.rows), rowSignature(staticRun.rows));
  }
  assert.ok(freshAdaptiveWork >= freshStaticWork);
  assert.ok(freshAdaptiveWork <= freshStaticWork + queriesBudget(workload.length, executor));
  assert.ok(freshReplans <= workload.length, "with fresh statistics a re-plan is at most one per query");

  // --- edge cases ---------------------------------------------------------------------------------------------
  const emptyRun = executor.execute({ filterA: { kind: "cmp", column: "age", op: ">", value: 1000 }, filterB: null }, staleEnv);
  assert.strictEqual(emptyRun.rows.length, 0);
  assert.ok(emptyRun.work > 0);
  assert.ok(emptyRun.replans <= executor.maxReplans);
  const single = makeEnvironment({ users: 1, orders: 1, staleRows: 1, seed: 5 });
  const oneRun = executor.execute({ filterA: null, filterB: null }, single);
  assert.strictEqual(oneRun.rows.length, 1);
  assert.deepStrictEqual(oneRun.truth, { users: 1, orders: 1 });
  const oversample = new AdaptiveExecutor({ sampleSize: 10_000 });
  const oversampleRun = oversample.execute({ filterA: null, filterB: null }, single);
  assert.strictEqual(oversampleRun.rows.length, 1);
  assert.ok(oversampleRun.work >= planCost(oversampleRun.plan, 1, 1));
  assert.throws(() => executor.execute(null, staleEnv), /query required/);
  assert.throws(() => makeEnvironment({ users: 0 }), /users must be/);
  assert.throws(() => makeEnvironment({ staleRows: 0 }), /staleRows/);

  // --- determinism -----------------------------------------------------------------------------------------------
  const first = executor.execute(workload[3], staleEnv);
  const second = executor.execute(workload[3], staleEnv);
  assert.strictEqual(first.work, second.work);
  assert.strictEqual(first.replans, second.replans);
  assert.deepStrictEqual(first.plan, second.plan);
  assert.deepStrictEqual(first.decisions, second.decisions);
  assert.strictEqual(rowSignature(first.rows), rowSignature(second.rows));
  const otherSeed = makeEnvironment({ staleRows: 10, seed: 857 });
  assert.strictEqual(otherSeed.tables.users.length, staleEnv.tables.users.length);
  assert.notDeepStrictEqual(otherSeed.tables.users[0], staleEnv.tables.users[0], "another seed shuffles differently");

  console.log(`workload: ${workload.length} queries, ${staleEnv.totals.users}+${staleEnv.totals.orders} rows, ${repaired} repairs, plan shapes ${[...planShapes].join(" ")}`);
  console.log(`work: adaptive=${adaptiveWork} stale-static=${staticWork} clairvoyant=${oracleWork}`);
  console.log(`economy: ${insideTolerance} queries left alone, ${outsideTolerance} re-planned after sampling`);
  console.log("PASS");
}

/** Upper bound on the extra work the adaptive path may add over the static one. */
function queriesBudget(count, executor) {
  return count * (2 * executor.sampleSize + executor.maxReplans * executor.reoptimizeCost);
}

module.exports = { AdaptiveExecutor, choosePlan, planCost, deviation, makeEnvironment, generateWorkload, staticExecution, oracleExecution, bruteForceJoin, runPlan, filteredInputs, estimatedCardinalities, assumedSelectivity, rowSignature, mulberry32, HASH_MEMORY, SPILL_PENALTY, PER_ROW_INDEX };

if (require.main === module) {
  selfCheck();
}
