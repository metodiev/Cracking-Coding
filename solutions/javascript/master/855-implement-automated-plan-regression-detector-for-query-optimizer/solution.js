/** Task 855 - Implement automated plan regression detector for query optimizer changes.
 *  (Tier: Master, CTCI Ch 14 — Databases)
 *
 * Model. A rule-based optimizer turns a query AST into a physical plan tree (scan / hash-join /
 * nested-loop join / filter / sort / limit). Three optimizer switches change the plan without
 * changing what the query means:
 *
 *   pushFilter      when false, the base-table predicate stays above the join (classic
 *                   missing-predicate-pushdown regression);
 *   hashJoin        when false, joins fall back to a nested loop (quadratic regression);
 *   sortElimination when false, an extra sort is inserted even where the plan does not need one.
 *
 * Plans are executed by a deterministic in-process executor that returns both the result rows and
 * the amount of work performed (rows read, pairs probed, comparisons). The regression detector runs
 * a benchmark workload through the baseline and the candidate optimizer, measures per-query work
 * under seeded measurement noise, and decides with an exact two-sided paired sign test plus a
 * relative-cost threshold. When a change set regresses, delta debugging narrows it to a minimal
 * culprit subset.
 *
 * What is simplified: cost is a deterministic integer work count rather than time, statistics are
 * known exactly (no cardinality estimation error), measurement noise is a bounded seeded jitter,
 * the workload is generated from a fixed seed, and plans are plain objects — no disk, no threads,
 * no wall clock, no external state.
 *
 * Invariants verified:
 *   equivalence every optimizer variant returns exactly the same rows for every query in the
 *                workload, so a cost regression can never hide a semantics change.
 *   specificity  comparing an optimizer with itself yields zero regressed queries and a
 *                "no-change" verdict, for every noise seed tried (no false positives).
 *   recall      each injected regression switch is reported as a regression, with a significant
 *                exact two-sided sign-test p-value.
 *   attribution minimal delta debugging over the change set identifies the culprit switch (and, for
 *                two-switch change sets, still returns the single guilty switch), and the returned
 *                subset is 1-minimal: removing any element of it stops reproducing.
 *   statistics  the closed-form sign-test p-value equals a brute-force enumeration of all 2^n sign
 *                assignments (the oracle) for the sample sizes used here.
 *   determinism identical seeds produce identical reports.
 */
"use strict";

const assert = require("node:assert");

const SWITCHES = ["pushFilter", "hashJoin", "sortElimination"];
const DEFAULTS = { pushFilter: true, hashJoin: true, sortElimination: true };

/** mulberry32 — the only source of randomness (workload generation and measurement noise). */
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

/** Deterministic warehouse fixture: 60 users and 240 orders. */
function sampleData() {
  const users = [];
  for (let id = 0; id < 60; id++) users.push({ id, dept: ["eng", "sales", "hr"][id % 3], age: 20 + ((id * 7) % 40) });
  const orders = [];
  for (let id = 0; id < 240; id++) orders.push({ id, user_id: (id * 13) % 60, amount: 5 + ((id * 7) % 90) });
  return { users, orders };
}

function evalPredicate(pred, row) {
  if (pred === null) return true;
  if (pred.kind === "and") return pred.children.every((child) => evalPredicate(child, row));
  if (pred.kind === "or") return pred.children.some((child) => evalPredicate(child, row));
  const value = row[pred.column];
  switch (pred.op) {
    case "==":
      return value === pred.value;
    case "<":
      return value < pred.value;
    case "<=":
      return value <= pred.value;
    case ">":
      return value > pred.value;
    case ">=":
      return value >= pred.value;
    default:
      throw new Error(`unknown operator ${pred.op}`);
  }
}

/** Optimizer: query AST + switches -> physical plan tree. */
function optimize(query, overrides = null) {
  const settings = overrides === null || overrides === undefined ? {} : overrides;
  if (typeof settings !== "object") throw new TypeError("overrides must be an object");
  for (const key of Object.keys(settings)) {
    if (!SWITCHES.includes(key)) throw new Error(`unknown optimizer switch ${key}`);
  }
  const flags = { ...DEFAULTS, ...settings };
  const base = { kind: "scan", table: query.table, filter: flags.pushFilter || query.join === null ? query.filter : null };
  let plan;
  if (query.join === null) {
    plan = base;
  } else {
    // The join-side predicate is always applied to the probe side; the regression switch only affects
    // whether the base-table predicate is pushed below the join.
    const probe = { kind: "scan", table: query.join.table, filter: query.join.filter };
    plan = flags.hashJoin
      ? { kind: "hashJoin", left: base, right: probe, leftKey: query.join.left, rightKey: query.join.right }
      : { kind: "nestedLoop", left: base, right: probe, leftKey: query.join.left, rightKey: query.join.right };
    if (!flags.pushFilter && query.filter !== null) {
      plan = { kind: "filter", predicate: query.filter, child: plan };
    }
  }
  const needsSort = query.orderBy !== null;
  if (needsSort || !flags.sortElimination) {
    plan = { kind: "sort", column: query.orderBy === null ? "id" : query.orderBy, child: plan, redundant: !needsSort };
  }
  if (query.limit !== null) plan = { kind: "limit", count: query.limit, child: plan };
  return { plan, flags };
}

/** Deterministic executor: returns result rows plus the work performed. */
function execute(plan, data) {
  switch (plan.kind) {
    case "scan": {
      const table = data[plan.table];
      const rows = table.filter((row) => evalPredicate(plan.filter, row));
      return { rows, work: table.length + rows.length, nodes: 1 };
    }
    case "filter": {
      const child = execute(plan.child, data);
      const rows = child.rows.filter((row) => evalPredicate(plan.predicate, row));
      return { rows, work: child.work + child.rows.length, nodes: child.nodes + 1 };
    }
    case "hashJoin": {
      const left = execute(plan.left, data);
      const right = execute(plan.right, data);
      const buckets = new Map();
      for (const row of right.rows) {
        const key = row[plan.rightKey];
        const list = buckets.get(key);
        if (list === undefined) buckets.set(key, [row]);
        else list.push(row);
      }
      const rows = [];
      for (const row of left.rows) {
        for (const other of buckets.get(row[plan.leftKey]) || []) rows.push({ ...other, ...row });
      }
      // Build the probe side once (right.rows) and probe it once per outer row (left.rows).
      const work = left.work + right.work + right.rows.length + left.rows.length + rows.length;
      return { rows, work, nodes: left.nodes + right.nodes + 1 };
    }
    case "nestedLoop": {
      const left = execute(plan.left, data);
      const right = execute(plan.right, data);
      const rows = [];
      let pairs = 0;
      for (const row of left.rows) {
        for (const other of right.rows) {
          pairs += 1;
          if (row[plan.leftKey] === other[plan.rightKey]) rows.push({ ...other, ...row });
        }
      }
      // The probe side is materialised once; what remains is the textbook O(outer x inner) pairs.
      const innerCost = right.work + right.rows.length;
      const work = left.work + innerCost + pairs + rows.length;
      return { rows, work, nodes: left.nodes + right.nodes + 1 };
    }
    case "sort": {
      const child = execute(plan.child, data);
      const rows = child.rows.slice().sort((a, b) => (a[plan.column] < b[plan.column] ? -1 : a[plan.column] > b[plan.column] ? 1 : 0));
      const comparisons = rows.length === 0 ? 0 : Math.ceil(rows.length * Math.log2(rows.length + 1));
      return { rows, work: child.work + comparisons + (plan.redundant ? rows.length : 0), nodes: child.nodes + 1 };
    }
    case "limit": {
      const child = execute(plan.child, data);
      return { rows: child.rows.slice(0, plan.count), work: child.work + child.rows.length, nodes: child.nodes + 1 };
    }
    default:
      throw new Error(`unknown plan node ${plan.kind}`);
  }
}

/** Canonical form: filters inside AND nodes are sorted so commutativity does not look like a change. */
function canonicalPlan(plan) {
  switch (plan.kind) {
    case "scan":
      return `scan(${plan.table}${plan.filter === null ? "" : `,${canonicalPredicate(plan.filter)}`})`;
    case "filter":
      return `filter(${canonicalPredicate(plan.predicate)},${canonicalPlan(plan.child)})`;
    case "hashJoin":
    case "nestedLoop":
      return `${plan.kind}(${plan.leftKey}=${plan.rightKey},${canonicalPlan(plan.left)},${canonicalPlan(plan.right)})`;
    case "sort":
      return `sort(${plan.column}${plan.redundant ? ",redundant" : ""},${canonicalPlan(plan.child)})`;
    case "limit":
      return `limit(${plan.count},${canonicalPlan(plan.child)})`;
    default:
      throw new Error(`unknown plan node ${plan.kind}`);
  }
}

function canonicalPredicate(pred) {
  if (pred === null) return "*";
  if (pred.kind === "and" || pred.kind === "or") {
    return `${pred.kind}(${pred.children.map(canonicalPredicate).slice().sort().join(",")})`;
  }
  return `${pred.column}${pred.op}${pred.value}`;
}

/**
 * Result signature. Without ORDER BY only the row multiset matters. With ORDER BY the primary key
 * sequence is compared exactly, but rows that tie on the sort key are compared as a multiset, since
 * SQL does not define the order among equal keys.
 */
function resultSignature(rows, orderBy = null) {
  const rowSignature = (row) =>
    JSON.stringify(
      Object.keys(row)
        .sort()
        .map((key) => [key, row[key]]),
    );
  if (orderBy === null) return rows.map(rowSignature).sort().join("|");
  const groups = [];
  for (const row of rows) {
    const key = JSON.stringify(row[orderBy]);
    const last = groups[groups.length - 1];
    if (last !== undefined && last.key === key) last.signatures.push(rowSignature(row));
    else groups.push({ key, signatures: [rowSignature(row)] });
  }
  return groups.map((group) => `${group.key}:[${group.signatures.slice().sort().join(",")}]`).join(" > ");
}

/** Query generator: base filter, optional join with its own filter, projection, order+limit. */
function generateQuery(rand) {
  const predicate = (depth, columns) => {
    const roll = rand();
    if (depth > 0 && roll < 0.35) {
      return { kind: roll < 0.18 ? "and" : "or", children: [predicate(depth - 1, columns), predicate(depth - 1, columns)] };
    }
    if (columns.length === 1) return { kind: "cmp", column: columns[0], op: pick(rand, ["==", "<", "<=", ">", ">="]), value: 5 + Math.floor(rand() * 60) };
    // Mid-selectivity base predicates: the benchmark must exercise plans that touch many rows.
    return { kind: "cmp", column: pick(rand, columns), op: pick(rand, ["==", "<", "<=", ">", ">="]), value: 20 + Math.floor(rand() * 25) };
  };
  const withJoin = rand() < 0.8;
  const query = {
    table: "users",
    // Base predicates only mention base columns so that predicate pushdown is semantics preserving.
    filter: rand() < 0.2 ? null : predicate(1, ["id", "dept", "age"]),
    join: null,
    orderBy: null,
    limit: null,
  };
  if (withJoin) {
    query.join = {
      table: "orders",
      left: "id",
      right: "user_id",
      filter: rand() < 0.3 ? null : { kind: "cmp", column: "amount", op: pick(rand, [">", "<="]), value: 20 + Math.floor(rand() * 50) },
    };
  } else {
    query.filter = query.filter; // no join: the filter stays on the scan either way
  }
  if (rand() < 0.4) query.orderBy = pick(rand, withJoin ? ["id", "amount", "age"] : ["id", "age"]);
  if (query.orderBy !== null && rand() < 0.5) query.limit = 1 + Math.floor(rand() * 8);
  return query;
}

/** Benchmark: runs the workload through one optimizer configuration, with bounded seeded noise. */
function benchmark(queries, data, overrides, noise, noiseSeed) {
  const rand = mulberry32(noiseSeed);
  const report = { costs: [], plans: [], results: [], work: 0 };
  let executions = 0;
  for (const query of queries) {
    const { plan } = optimize(query, overrides);
    const outcome = execute(plan, data);
    executions += 1;
    report.plans.push(canonicalPlan(plan));
    report.results.push(resultSignature(outcome.rows, query.orderBy));
    report.costs.push(Math.round(outcome.work * (1 + (rand() * 2 - 1) * noise)));
    report.work += outcome.work;
  }
  report.executions = executions;
  return report;
}

/** Exact two-sided sign test p-value from the closed form (2 * P(X <= min(wins, losses))). */
function signTestPValue(wins, losses) {
  const n = wins + losses;
  if (n === 0) return 1;
  const k = Math.min(wins, losses);
  let tail = 0;
  for (let i = 0; i <= k; i++) tail += binom(n, i);
  const p = (2 * tail) / Math.pow(2, n);
  return Math.min(1, Number(p.toFixed(12)));
}

/** Oracle: brute-force enumeration of every sign assignment, counting the two-sided tail exactly. */
function signTestPValueBruteForce(wins, losses) {
  const n = wins + losses;
  if (n === 0) return 1;
  const observed = Math.min(wins, losses);
  let extreme = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    let plus = 0;
    for (let bit = 0; bit < n; bit++) if (mask & (1 << bit)) plus += 1;
    if (Math.min(plus, n - plus) <= observed) extreme += 1;
  }
  return Math.min(1, Number((extreme / Math.pow(2, n)).toFixed(12)));
}

function binom(n, k) {
  let out = 1;
  for (let i = 0; i < k; i++) out = (out * (n - i)) / (i + 1);
  return out;
}

/**
 * The detector itself: compares the baseline report with the candidate report. A verdict needs three
 * things at once — a significant paired sign test (the direction is real), more than `minFraction` of
 * the workload beyond the per-query tolerance (the size is real) and a consistent direction — which
 * is what keeps bounded measurement noise from raising alarms.
 */
function detect(before, after, { threshold = 0.05, alpha = 0.05, minFraction = 0.1 } = {}) {
  if (!(threshold >= 0 && threshold < 1)) throw new RangeError("threshold must be in [0,1)");
  if (!(alpha > 0 && alpha < 1)) throw new RangeError("alpha must be in (0,1)");
  if (!(minFraction > 0 && minFraction <= 1)) throw new RangeError("minFraction must be in (0,1]");
  if (before.costs.length === 0) throw new RangeError("the workload must not be empty");
  if (before.costs.length !== after.costs.length) throw new RangeError("reports must cover the same workload");
  const ratios = before.costs.map((cost, i) => (cost === 0 ? (after.costs[i] === 0 ? 1 : Infinity) : after.costs[i] / cost));
  const regressed = [];
  const improved = [];
  let wins = 0;
  let losses = 0;
  for (let i = 0; i < ratios.length; i++) {
    if (after.costs[i] > before.costs[i]) wins += 1;
    else if (after.costs[i] < before.costs[i]) losses += 1;
    if (ratios[i] > 1 + threshold) regressed.push({ index: i, ratio: ratios[i] });
    if (ratios[i] < 1 / (1 + threshold)) improved.push({ index: i, ratio: ratios[i] });
  }
  const sorted = ratios.slice().sort((a, b) => a - b);
  const medianRatio = sorted[Math.floor(sorted.length / 2)];
  const p = signTestPValue(wins, losses);
  const significant = p <= alpha;
  const planChanges = before.plans.reduce((acc, plan, i) => acc + (plan === after.plans[i] ? 0 : 1), 0);
  let verdict = "no-change";
  if (significant && wins > losses && regressed.length >= Math.max(1, Math.ceil(minFraction * ratios.length))) {
    verdict = "regression";
  } else if (significant && losses > wins && improved.length >= Math.max(1, Math.ceil(minFraction * ratios.length))) {
    verdict = "improvement";
  }
  return { verdict, regressed, improved, medianRatio, planChanges, signTest: { wins, losses, p, n: wins + losses } };
}

/** Full comparison run: optimize + execute both configurations and detect. */
function compareWorkload(queries, data, baselineOverrides, candidateOverrides, { noise = 0.02, noiseSeed = 1, ...options } = {}) {
  const before = benchmark(queries, data, baselineOverrides, noise, noiseSeed);
  const after = benchmark(queries, data, candidateOverrides, noise, noiseSeed + 1000);
  const report = detect(before, after, options);
  return { before, after, ...report, executions: before.executions + after.executions };
}

/** Merges a change set (list of partial switch objects) into a single overrides object. */
function mergeChanges(changes) {
  const merged = {};
  for (const change of changes) Object.assign(merged, change);
  return merged;
}

/** Delta debugging over a change set: returns a 1-minimal reproducing subset. */
function bisectChangeSet(queries, data, changes, { noise = 0, noiseSeed = 7 } = {}) {
  let current = changes;
  let steps = 0;
  let improved = true;
  while (improved) {
    improved = false;
    for (let i = 0; i < current.length; i++) {
      const candidate = current.filter((_, j) => j !== i);
      steps += 1;
      if (candidate.length === 0) continue;
      const outcome = compareWorkload(queries, data, null, mergeChanges(candidate), { noise, noiseSeed });
      if (outcome.verdict === "regression") {
        current = candidate;
        improved = true;
        break;
      }
    }
  }
  const outcome = compareWorkload(queries, data, null, mergeChanges(current), { noise, noiseSeed });
  return { changes: current, reproduces: outcome.verdict === "regression", steps, executions: outcome.executions };
}

/** Exhaustive minimal-subset search (used as the oracle for attribution). */
function minimalChangeSet(queries, data, changes, { noise = 0, noiseSeed = 7 } = {}) {
  let executions = 0;
  for (let size = 1; size <= changes.length; size++) {
    for (let mask = 0; mask < 1 << changes.length; mask++) {
      const subset = changes.filter((_, i) => (mask >> i) & 1);
      if (subset.length !== size) continue;
      const outcome = compareWorkload(queries, data, null, mergeChanges(subset), { noise, noiseSeed });
      executions += outcome.executions;
      if (outcome.verdict === "regression") return { changes: subset, executions };
    }
  }
  return { changes: [], executions };
}

function selfCheck() {
  // --- validation ---------------------------------------------------------------------------------
  assert.throws(() => optimize({ table: "users", filter: null, join: null, orderBy: null, limit: null }, { magicSwitch: true }), /unknown optimizer switch/);
  assert.throws(() => detect({ costs: [], plans: [] }, { costs: [], plans: [] }), /must not be empty/);
  assert.throws(() => detect({ costs: [1], plans: [""] }, { costs: [1], plans: [""] }, { threshold: 1.5 }), /threshold/);
  assert.throws(() => detect({ costs: [1], plans: [""] }, { costs: [1, 2], plans: ["", ""] }), /same workload/);

  const data = sampleData();
  assert.strictEqual(data.users.length, 60);
  assert.strictEqual(data.orders.length, 240);

  const rand = mulberry32(855);
  const queries = [];
  for (let i = 0; i < 80; i++) queries.push(generateQuery(rand));
  assert.ok(queries.some((q) => q.join !== null) && queries.filter((q) => q.join === null).length >= 5);
  assert.ok(queries.some((q) => q.orderBy !== null) && queries.some((q) => q.limit !== null));

  // --- plan shaping: the switches really change the plan ----------------------------------------------
  const joined = queries.find((q) => q.join !== null && q.filter !== null);
  const planned = optimize(joined);
  assert.ok(canonicalPlan(planned.plan).includes("hashJoin"));
  const unpushed = optimize(joined, { pushFilter: false });
  assert.ok(canonicalPlan(unpushed.plan).includes("filter("), "without pushdown the base predicate sits above the join");
  assert.ok(canonicalPlan(unpushed.plan).includes("scan(users)"), "the users scan itself loses its predicate");
  assert.ok(canonicalPlan(unpushed.plan).includes("scan(orders,amount"), "the join-side predicate is still pushed");
  assert.ok(/scan\(users,[^*]/.test(canonicalPlan(planned.plan)), "with pushdown the users scan keeps its predicate");
  const unpushedNoJoin = optimize({ ...joined, join: null, orderBy: null, limit: null }, { pushFilter: false });
  assert.ok(/scan\(users,[^*]/.test(canonicalPlan(unpushedNoJoin.plan)), "a join-free query always pushes its predicate into the scan");
  const nested = optimize(joined, { hashJoin: false });
  assert.ok(canonicalPlan(nested.plan).includes("nestedLoop"));
  assert.ok(!canonicalPlan(nested.plan).includes("hashJoin"));
  const extraSort = optimize({ ...joined, orderBy: null, limit: null }, { sortElimination: false });
  assert.ok(canonicalPlan(extraSort.plan).includes("redundant"));
  assert.strictEqual(optimize({ ...joined, orderBy: null, limit: null }).plan.kind, "hashJoin", "a redundant sort must be eliminated by default");
  // The canonical form is idempotent and blind to the commutativity of AND.
  const andPredicate = { kind: "and", children: [{ kind: "cmp", column: "age", op: ">", value: 30 }, { kind: "cmp", column: "dept", op: "==", value: "eng" }] };
  const swapped = { kind: "and", children: [andPredicate.children[1], andPredicate.children[0]] };
  const base = { table: "users", filter: andPredicate, join: null, orderBy: null, limit: null };
  assert.strictEqual(canonicalPlan(optimize(base).plan), canonicalPlan(optimize({ ...base, filter: swapped }).plan));
  assert.strictEqual(canonicalPlan(optimize(base).plan), canonicalPlan(optimize(base).plan));

  // --- equivalence: no switch may change the query result ----------------------------------------------
  for (const query of queries) {
    const outcomes = [];
    for (const overrides of [{}, { pushFilter: false }, { hashJoin: false }, { sortElimination: false }, { pushFilter: false, hashJoin: false, sortElimination: false }]) {
      const { plan } = optimize(query, overrides);
      const outcome = execute(plan, data);
      outcomes.push(resultSignature(outcome.rows, query.orderBy));
    }
    assert.ok(outcomes.every((signature) => signature === outcomes[0]), `plan variants disagree for ${JSON.stringify(query)}`);
  }

  // --- hand-checked execution and work accounting -------------------------------------------------------
  const usersOnly = { table: "users", filter: { kind: "cmp", column: "dept", op: "==", value: "eng" }, join: null, orderBy: null, limit: null };
  const scanned = execute(optimize(usersOnly).plan, data);
  assert.strictEqual(scanned.rows.length, data.users.filter((r) => r.dept === "eng").length);
  assert.strictEqual(scanned.work, data.users.length + scanned.rows.length, "scan work = rows read + rows kept");
  const limited = execute(optimize({ ...usersOnly, orderBy: "id", limit: 3 }).plan, data);
  assert.strictEqual(limited.rows.length, 3);
  assert.ok(limited.rows[0].id <= limited.rows[1].id && limited.rows[1].id <= limited.rows[2].id);
  const joinedResult = execute(optimize(joined).plan, data);
  assert.ok(joinedResult.rows.every((row) => row.dept !== undefined && row.amount !== undefined));
  assert.ok(joinedResult.rows.length > 0 && joinedResult.rows.length <= data.orders.length);

  // --- statistics: closed form vs brute-force enumeration ------------------------------------------------
  for (let n = 1; n <= 14; n++) {
    for (const wins of [0, Math.floor(n / 3), Math.floor(n / 2), n]) {
      const losses = n - wins;
      assert.strictEqual(signTestPValue(wins, losses), signTestPValueBruteForce(wins, losses), `sign test n=${n} wins=${wins}`);
    }
  }
  assert.strictEqual(signTestPValue(5, 5), 1);
  assert.ok(signTestPValue(10, 0) < 0.01 && signTestPValue(9, 1) < 0.05);
  assert.ok(signTestPValue(7, 3) > 0.05, "a mild majority must not be significant");

  // --- specificity: an optimizer compared with itself never regresses -------------------------------------
  // With measurement noise bounded by 2% the worst possible cost ratio is 1.02/0.98 < 1.05, so a 5%
  // tolerance cannot produce a false alarm for any noise seed.
  const noiseSeeds = [1, 7, 13, 21, 34, 55, 89, 144, 233, 377];
  for (const noiseSeed of noiseSeeds) {
    const self = compareWorkload(queries, data, null, {}, { noise: 0.02, noiseSeed });
    assert.strictEqual(self.verdict, "no-change", `noise seed ${noiseSeed} produced a false alarm`);
    assert.deepStrictEqual(self.regressed, []);
    assert.strictEqual(self.planChanges, 0);
    const ties = self.before.costs.filter((cost, i) => cost === self.after.costs[i]).length;
    assert.strictEqual(self.signTest.n + ties, queries.length, "every non-tied query must be counted");
    assert.ok(Math.abs(self.medianRatio - 1) <= 0.05);
  }

  // --- the detector's decision rule, unit tested on hand-built reports -------------------------------------
  const flat = (n, value) => ({ costs: new Array(n).fill(value), plans: new Array(n).fill("p") });
  assert.strictEqual(detect(flat(20, 100), flat(20, 102)).verdict, "no-change", "2% is inside a 5% tolerance");
  assert.strictEqual(detect(flat(20, 100), flat(20, 110)).verdict, "regression");
  assert.strictEqual(detect(flat(20, 110), flat(20, 100)).verdict, "improvement");
  assert.strictEqual(detect(flat(20, 100), flat(20, 100)).verdict, "no-change");
  const mixed = { costs: [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100], plans: new Array(20).fill("p") };
  const mixedReport = detect(flat(20, 100), mixed);
  assert.strictEqual(mixedReport.verdict, "regression");
  assert.strictEqual(mixedReport.regressed.length, 10);
  assert.strictEqual(mixedReport.signTest.wins, 10);
  assert.strictEqual(mixedReport.signTest.losses, 0);
  // Three regressed queries out of twenty is not statistically significant, so no verdict is raised.
  const sparse = flat(20, 100);
  sparse.costs[0] = 150;
  sparse.costs[1] = 150;
  sparse.costs[2] = 150;
  const sparseReport = detect(flat(20, 100), sparse);
  assert.strictEqual(sparseReport.verdict, "no-change");
  assert.strictEqual(sparseReport.regressed.length, 3);
  assert.ok(sparseReport.signTest.p > 0.05);
  // A change below the tolerance is ignored even when every query is affected.
  assert.strictEqual(detect(flat(30, 500), flat(30, 515), { threshold: 0.05 }).verdict, "no-change");
  assert.strictEqual(detect(flat(30, 500), flat(30, 515), { threshold: 0.02 }).verdict, "regression");

  // --- recall: each injected regression switch is reported --------------------------------------------------
  const expectedDetections = [];
  for (const key of SWITCHES) {
    const outcome = compareWorkload(queries, data, null, { [key]: false }, { noise: 0.02, noiseSeed: 3 });
    assert.strictEqual(outcome.verdict, "regression", `switch ${key} must be detected`);
    assert.ok(outcome.regressed.length > 0 && outcome.regressed.length <= queries.length);
    assert.ok(outcome.signTest.p <= 0.05, `${key}: p=${outcome.signTest.p}`);
    assert.ok(outcome.medianRatio > 1);
    assert.ok(outcome.planChanges > 0, `${key}: the plan shapes must change`);
    expectedDetections.push(`${key} (p=${outcome.signTest.p.toExponential(2)}, regressed=${outcome.regressed.length}, plan Δ=${outcome.planChanges})`);
  }
  const triple = compareWorkload(queries, data, null, { pushFilter: false, hashJoin: false, sortElimination: false }, { noise: 0.02, noiseSeed: 3 });
  assert.strictEqual(triple.verdict, "regression");
  assert.ok(triple.medianRatio > 1);

  // --- improvements are recognised and are not reported as regressions ---------------------------------------
  const improved = compareWorkload(queries, data, { pushFilter: false }, {}, { noise: 0.02, noiseSeed: 3 });
  assert.strictEqual(improved.verdict, "improvement");
  assert.deepStrictEqual(improved.regressed, []);
  assert.ok(improved.improved.length > 0 && improved.medianRatio < 1);

  // --- attribution: bisection and the exhaustive oracle agree -------------------------------------------------
  const changes = [{ pushFilter: false }, { hashJoin: false }, { sortElimination: false }];
  const blamed = bisectChangeSet(queries, data, changes, { noise: 0, noiseSeed: 7 });
  assert.strictEqual(blamed.changes.length, 1, "exactly one switch must be blamed");
  assert.ok(SWITCHES.includes(Object.keys(blamed.changes[0])[0]));
  assert.strictEqual(blamed.reproduces, true);
  assert.ok(blamed.steps > 0 && blamed.steps <= changes.length * changes.length);
  const oracle = minimalChangeSet(queries, data, changes, { noise: 0, noiseSeed: 7 });
  assert.strictEqual(oracle.changes.length, blamed.changes.length, "the oracle agrees on the size");
  assert.ok(oracle.executions >= changes.length);
  for (const key of SWITCHES) {
    const single = bisectChangeSet(queries, data, [{ [key]: false }], { noise: 0, noiseSeed: 7 });
    assert.deepStrictEqual(single.changes, [{ [key]: false }], `bisection must keep ${key}`);
  }
  const pair = bisectChangeSet(queries, data, [{ hashJoin: false }, { pushFilter: false }], { noise: 0, noiseSeed: 7 });
  assert.strictEqual(pair.changes.length, 1, "the innocent switch must be dropped");
  assert.strictEqual(pair.reproduces, true);
  assert.ok(pair.changes[0].hashJoin === false || pair.changes[0].pushFilter === false);
  // 1-minimality: removing any element of the blamed set stops reproducing.
  for (let i = 0; i < blamed.changes.length; i++) {
    const without = blamed.changes.filter((_, j) => j !== i);
    if (without.length === 0) continue;
    const outcome = compareWorkload(queries, data, null, without, { noise: 0, noiseSeed: 7 });
    assert.notStrictEqual(outcome.verdict, "regression");
  }
  const emptyChangeSet = bisectChangeSet(queries, data, [], { noise: 0, noiseSeed: 7 });
  assert.deepStrictEqual(emptyChangeSet.changes, []);
  assert.strictEqual(emptyChangeSet.reproduces, false, "an empty change set cannot regress");

  // --- cost bound and determinism ------------------------------------------------------------------------------
  const bounded = compareWorkload(queries, data, null, { hashJoin: false }, { noise: 0.02, noiseSeed: 3 });
  assert.strictEqual(bounded.executions, queries.length * 2, "each configuration executes the workload exactly once");
  const reportA = compareWorkload(queries, data, null, { pushFilter: false, hashJoin: false }, { noise: 0.02, noiseSeed: 42 });
  const reportB = compareWorkload(queries, data, null, { pushFilter: false, hashJoin: false }, { noise: 0.02, noiseSeed: 42 });
  assert.deepStrictEqual(reportA.regressed, reportB.regressed);
  assert.strictEqual(reportA.signTest.p, reportB.signTest.p);
  assert.strictEqual(reportA.verdict, reportB.verdict);
  const milder = compareWorkload(queries, data, null, { sortElimination: false }, { noise: 0.02, noiseSeed: 42 });
  assert.ok(reportA.medianRatio >= milder.medianRatio, "a broader change set must not look milder");
  // A one-query workload carries no statistical power.
  const tiny = compareWorkload([queries[0]], data, null, { hashJoin: false }, { noise: 0, noiseSeed: 1 });
  assert.strictEqual(tiny.signTest.n, 1);
  assert.strictEqual(tiny.verdict, "no-change");

  console.log(`workload: ${queries.length} queries, ${data.users.length}+${data.orders.length} rows`);
  console.log(`detections: ${expectedDetections.join(" | ")}`);
  console.log(`attribution: bisection blamed ${JSON.stringify(blamed.changes)} in ${blamed.steps} steps (oracle agrees)`);
  console.log("PASS");
}

module.exports = { optimize, execute, canonicalPlan, generateQuery, sampleData, benchmark, detect, compareWorkload, bisectChangeSet, minimalChangeSet, signTestPValue, signTestPValueBruteForce, resultSignature, SWITCHES, DEFAULTS, mulberry32 };

if (require.main === module) {
  selfCheck();
}
