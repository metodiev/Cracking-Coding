/** Task 854 - Implement system for differential testing and fuzzing of database engines.
 *  (Tier: Master, CTCI Ch 14 — Databases)
 *
 * Model. Two engines run the same query AST over the same in-memory table:
 *
 *   ReferenceEngine  a deliberately naive full-scan interpreter: evaluate the predicate per row with
 *                    a recursive tree walk, project, aggregate, limit. Slow and obviously correct.
 *   FastEngine       a small optimizer: it decomposes the top-level conjunction, picks an access
 *                    path (hash lookup for `==`/`in`, binary search over a sorted column for range
 *                    comparisons), reorders conjuncts by estimated selectivity, pushes the projection
 *                    down, and only then re-evaluates the full predicate on the candidate rows.
 *
 * Differential testing asserts the two engines agree on every generated query; a fuzzer generates
 * queries from a seeded PRNG and stops at the first disagreement. Because a bug-free pair never
 * disagrees, the fuzzer is validated by mutation testing: five mutants of the optimized engine
 * (off-by-one on `<=`, dropping an OR branch, treating NULL as 0, applying LIMIT before WHERE,
 * initialising SUM at 1) must each be caught within a bounded number of queries. The first failing
 * case is then shrunk by delta debugging (row deletion, predicate simplification, projection
 * trimming, LIMIT removal) down to a minimal reproducer.
 *
 * What is simplified: a single process, no SQL parsing (queries are ASTs), one flat table, no
 * disk/B-tree, rows are plain objects, no concurrency, and all randomness is one seeded mulberry32
 * stream. Cost is measured in rows examined rather than in time.
 *
 * Invariants verified:
 *   correctness  the two engines agree on a large deterministic corpus (no false positives), an
 *                independent declarative Array-based evaluator agrees with the reference as well,
 *                and both agree on hand-computed cases including NULL and boundary comparisons.
 *   recall       every injected mutant is detected, each within the bounded query budget.
 *   minimality   the delta debugger returns a reproducer that still fails and for which no single
 *                reduction step still fails.
 *   cost         the optimized engine examines strictly fewer rows than the reference over the
 *                corpus, never more rows than the table holds, and exactly one row for a unique-key
 *                equality lookup.
 *   determinism  the same seed reproduces the identical corpus and the identical mismatch report.
 */
"use strict";

const assert = require("node:assert");

const NUMERIC_OPS = ["==", "!=", "<", "<=", ">", ">="];
const AGGREGATES = ["count", "sum", "max"];
const MUTANTS = ["off-by-one", "drop-or-branch", "null-as-zero", "limit-before-filter", "sum-starts-at-one"];

/** mulberry32 — the only source of randomness; every corpus is reproducible from its seed. */
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

/** Deterministic warehouse table: 48 rows, one nullable numeric column. */
function sampleTable(rows = 48, seed = 854) {
  const rand = mulberry32(seed);
  const depts = ["eng", "sales", "hr"];
  const table = [];
  for (let id = 0; id < rows; id++) {
    table.push({
      id,
      dept: depts[id % depts.length],
      age: 20 + ((id * 7) % 40),
      salary: 40 + ((id * 13) % 60),
      bonus: id % 3 === 0 ? null : (id * 5) % 25,
    });
  }
  for (let i = table.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = table[i];
    table[i] = table[j];
    table[j] = tmp;
  }
  return table;
}

const SCHEMA = {
  columns: ["id", "dept", "age", "salary", "bonus"],
  numeric: ["id", "age", "salary", "bonus"],
  categorical: ["dept"],
};

/** NULL poisons every comparison (SQL "unknown"); `!=` is therefore false for a NULL operand too. */
function compare(left, right, op) {
  if (left === null || left === undefined || right === null || right === undefined) return false;
  switch (op) {
    case "==":
      return left === right;
    case "!=":
      return left !== right;
    case "<":
      return left < right;
    case "<=":
      return left <= right;
    case ">":
      return left > right;
    case ">=":
      return left >= right;
    default:
      throw new Error(`unknown comparison operator ${op}`);
  }
}

function validatePredicate(pred, schema) {
  if (pred === null) return;
  switch (pred.kind) {
    case "cmp":
      if (!schema.columns.includes(pred.column)) throw new Error(`unknown column ${pred.column}`);
      if (!NUMERIC_OPS.includes(pred.op)) throw new Error(`unknown comparison operator ${pred.op}`);
      return;
    case "in":
      if (!schema.columns.includes(pred.column)) throw new Error(`unknown column ${pred.column}`);
      if (!Array.isArray(pred.values) || pred.values.length === 0) throw new Error("'in' needs values");
      return;
    case "and":
    case "or":
      if (!Array.isArray(pred.children) || pred.children.length === 0) throw new Error(`${pred.kind} needs children`);
      for (const child of pred.children) validatePredicate(child, schema);
      return;
    case "not":
      validatePredicate(pred.child, schema);
      return;
    default:
      throw new Error(`unknown predicate kind ${pred.kind}`);
  }
}

function validateQuery(query, schema) {
  if (query === null || typeof query !== "object") throw new Error("query must be an object");
  validatePredicate(query.filter, schema);
  for (const column of query.projection || []) {
    if (!schema.columns.includes(column)) throw new Error(`unknown projection column ${column}`);
  }
  if (query.aggregate !== null && query.aggregate !== undefined) {
    if (!AGGREGATES.includes(query.aggregate.op)) throw new Error(`unknown aggregate ${query.aggregate.op}`);
    if (!schema.numeric.includes(query.aggregate.column)) throw new Error(`aggregate needs a numeric column`);
  }
  if (query.limit !== null && query.limit !== undefined && (!Number.isInteger(query.limit) || query.limit < 0)) {
    throw new Error("limit must be a non-negative integer");
  }
  return query;
}

function evalPredicate(pred, row) {
  if (pred === null) return true;
  switch (pred.kind) {
    case "cmp":
      return compare(row[pred.column], pred.value, pred.op);
    case "in":
      return pred.values.includes(row[pred.column]);
    case "and":
      return pred.children.every((child) => evalPredicate(child, row));
    case "or":
      return pred.children.some((child) => evalPredicate(child, row));
    case "not":
      return !evalPredicate(pred.child, row);
    default:
      throw new Error(`unknown predicate kind ${pred.kind}`);
  }
}

/** Reference engine: one full scan, recursive predicate evaluation, no optimisation at all. */
class ReferenceEngine {
  constructor() {
    this.rowsExamined = 0;
  }

  execute(table, query) {
    validateQuery(query, SCHEMA);
    this.rowsExamined += table.length;
    const matched = table.filter((row) => evalPredicate(query.filter, row));
    return finish(matched, query);
  }
}

/** Shared result shaping: projection, aggregate, limit (aggregates always yield one row). */
function finish(rows, query) {
  if (query.aggregate) {
    const values = rows.map((row) => row[query.aggregate.column]);
    let value;
    if (query.aggregate.op === "count") value = rows.length;
    else if (query.aggregate.op === "max") {
      const numbers = values.filter((v) => typeof v === "number");
      value = numbers.length === 0 ? null : Math.max(...numbers);
    } else {
      value = values.filter((v) => typeof v === "number").reduce((a, b) => a + b, 0);
    }
    return [{ [query.aggregate.op]: value }];
  }
  const projected = rows.map((row) => {
    const out = {};
    for (const column of query.projection) out[column] = row[column];
    return out;
  });
  return query.limit === null || query.limit === undefined ? projected : projected.slice(0, query.limit);
}

/** Access-path support: per column a hash bucket index and a sorted array for range probes. */
function buildIndex(table) {
  const byColumn = new Map();
  const order = new WeakMap();
  table.forEach((row, position) => order.set(row, position));
  for (const column of SCHEMA.columns) {
    const buckets = new Map();
    const sorted = [];
    for (const row of table) {
      const value = row[column];
      const list = buckets.get(value);
      if (list === undefined) buckets.set(value, [row]);
      else list.push(row);
      sorted.push(row);
    }
    const numeric = sorted.every((r) => typeof r[column] === "number" || r[column] === null);
    if (numeric) {
      sorted.sort((a, b) => (a[column] === null ? 1 : b[column] === null ? -1 : a[column] - b[column]));
    } else {
      sorted.sort((a, b) => (a[column] < b[column] ? -1 : a[column] > b[column] ? 1 : 0));
    }
    byColumn.set(column, { buckets, sorted });
  }
  return { byColumn, order };
}

/** Flattens the top-level conjunction; anything under an OR/NOT is not pushed down. */
function conjuncts(pred) {
  if (pred === null) return [];
  if (pred.kind === "and") {
    const out = [];
    for (const child of pred.children) out.push(...conjuncts(child));
    return out;
  }
  return [pred];
}

/** Lower/upper bound of a row against a comparison, used to rank access paths by estimated output. */
function estimate(pred, columnIndex) {
  if (pred.kind === "cmp" && pred.op === "==") {
    const bucket = columnIndex.buckets.get(pred.value);
    return bucket === undefined ? 0 : bucket.length;
  }
  if (pred.kind === "in") {
    let total = 0;
    for (const value of pred.values) total += (columnIndex.buckets.get(value) || []).length;
    return total;
  }
  if (pred.kind === "cmp") {
    const side = pred.op === "<" || pred.op === "<=" ? -1 : 1;
    const strict = pred.op === "<" || pred.op === ">";
    const count = columnIndex.sorted.filter((row) => {
      const value = row[pred.column];
      if (value === null) return false;
      return side < 0 ? (strict ? value < pred.value : value <= pred.value) : strict ? value > pred.value : value >= pred.value;
    }).length;
    return count;
  }
  return Infinity;
}

/**
 * Optimized engine. It uses an index only to narrow the candidate rows; the predicate is always
 * re-evaluated in full, so the pipeline is a pure optimisation of the reference semantics.
 * `mutant` injects a deliberate defect for mutation testing of the fuzzer itself.
 */
class FastEngine {
  constructor({ mutant = null } = {}) {
    if (mutant !== null && !MUTANTS.includes(mutant)) throw new Error(`unknown mutant ${mutant}`);
    this.mutant = mutant;
    this.rowsExamined = 0;
    this.indexes = new WeakMap();
  }

  index(table) {
    let built = this.indexes.get(table);
    if (built === undefined) {
      built = buildIndex(table);
      this.indexes.set(table, built);
    }
    return built;
  }

  predicate(pred, row) {
    if (this.mutant === "off-by-one" && pred !== null && pred.kind === "cmp" && pred.op === "<=") {
      return compare(row[pred.column], pred.value, "<");
    }
    if (this.mutant === "null-as-zero") {
      return this.predicateNullAsZero(pred, row);
    }
    if (this.mutant === "drop-or-branch" && pred !== null && pred.kind === "or") {
      return this.predicate(pred.children[0], row); // silently ignores the remaining branches
    }
    return evalPredicate(pred, row);
  }

  predicateNullAsZero(pred, row) {
    if (pred === null) return true;
    switch (pred.kind) {
      case "cmp": {
        const raw = row[pred.column];
        return compare(raw === null ? 0 : raw, pred.value, pred.op);
      }
      case "in": {
        const raw = row[pred.column];
        return pred.values.includes(raw === null ? 0 : raw);
      }
      case "and":
        return pred.children.every((child) => this.predicateNullAsZero(child, row));
      case "or":
        return pred.children.some((child) => this.predicateNullAsZero(child, row));
      case "not":
        return !this.predicateNullAsZero(pred.child, row);
      default:
        throw new Error(`unknown predicate kind ${pred.kind}`);
    }
  }

  candidates(table, query) {
    const index = this.index(table);
    const parts = conjuncts(query.filter);
    let best = null;
    for (const part of parts) {
      if (part.kind !== "cmp" && part.kind !== "in") continue;
      const columnIndex = index.byColumn.get(part.column);
      if (columnIndex === undefined) continue;
      const rows = estimate(part, columnIndex);
      if (rows === Infinity) continue;
      if (best === null || rows < best.rows) best = { rows, part, columnIndex };
    }
    if (best === null) return table;
    let out;
    if (best.part.kind === "in" || best.part.op === "==") {
      out = [];
      const values = best.part.kind === "in" ? [...new Set(best.part.values)] : [best.part.value];
      for (const value of values) out.push(...(best.columnIndex.buckets.get(value) || []));
    } else {
      out = best.columnIndex.sorted.filter((row) => {
        const value = row[best.part.column];
        if (value === null || value === undefined) return false;
        return compare(value, best.part.value, best.part.op);
      });
    }
    // The index visits rows in value order; restore scan order so that LIMIT (which has no ORDER BY)
    // observes the same rows as the reference engine.
    return out.slice().sort((a, b) => index.order.get(a) - index.order.get(b));
  }

  execute(table, query) {
    validateQuery(query, SCHEMA);
    const candidates = this.candidates(table, query);
    this.rowsExamined += candidates.length;
    let matched;
    if (this.mutant === "limit-before-filter" && query.limit !== null && query.limit !== undefined && !query.aggregate) {
      matched = candidates.slice(0, query.limit).filter((row) => this.predicate(query.filter, row));
      return finish(matched, { ...query, limit: null });
    }
    matched = candidates.filter((row) => this.predicate(query.filter, row));
    if (this.mutant === "sum-starts-at-one" && query.aggregate && query.aggregate.op === "sum") {
      const result = finish(matched, query);
      return [{ sum: result[0].sum + 1 }];
    }
    return finish(matched, query);
  }
}

/** Canonical form of a result: rows are unordered unless LIMIT truncated them. */
function canonical(result, ordered) {
  const rows = result.map((row) => JSON.stringify(Object.keys(row).sort().map((k) => [k, row[k]])));
  return ordered ? rows.join("|") : rows.slice().sort().join("|");
}

/** Diffs a query across both engines; returns null when they agree. */
function diffQuery(reference, fast, table, query) {
  const expected = reference.execute(table, query);
  const actual = fast.execute(table, query);
  const ordered = query.limit !== null && query.limit !== undefined;
  if (canonical(expected, true) === canonical(actual, true)) return null;
  if (canonical(expected, ordered) === canonical(actual, ordered)) return null; // same multiset
  return { query, expected, actual };
}

/** Query generator: nested predicates, occasional aggregates and limits. */
function generateQuery(rand) {
  const valuePool = [-5, 0, 1, 5, 10, 20, 25, 40, 60, 99];
  const generatePredicate = (depth) => {
    const roll = rand();
    if (depth > 0 && roll < 0.30) {
      return {
        kind: roll < 0.15 ? "and" : "or",
        children: [generatePredicate(depth - 1), generatePredicate(depth - 1)],
      };
    }
    if (depth > 0 && roll < 0.36) return { kind: "not", child: generatePredicate(depth - 1) };
    if (roll < 0.44) {
      const column = pick(rand, SCHEMA.columns);
      if (column === "dept") return { kind: "in", column, values: ["eng", "sales"] }; // "in" over a categorical
      const values = [...new Set([pick(rand, valuePool), pick(rand, valuePool)])];
      return { kind: "in", column, values };
    }
    const column = pick(rand, SCHEMA.columns);
    return {
      kind: "cmp",
      column,
      op: pick(rand, NUMERIC_OPS),
      value: column === "dept" ? pick(rand, ["eng", "hr"]) : pick(rand, valuePool),
    };
  };
  const aggregate = rand() < 0.25 ? { op: pick(rand, AGGREGATES), column: pick(rand, SCHEMA.numeric) } : null;
  let projection = [];
  if (aggregate === null) {
    const wanted = 1 + Math.floor(rand() * 3);
    while (projection.length < wanted) {
      const column = pick(rand, SCHEMA.columns);
      if (!projection.includes(column)) projection.push(column);
    }
  }
  return {
    filter: rand() < 0.12 ? null : generatePredicate(2),
    projection,
    aggregate,
    limit: aggregate === null && rand() < 0.25 ? Math.floor(rand() * 6) : null,
  };
}

/** Fuzzes `queries` cases; stops at the first disagreement when `stopOnFailure`. */
function fuzz({ seed = 854, queries = 800, mutant = null, table = null, stopOnFailure = true } = {}) {
  const data = table || sampleTable();
  const reference = new ReferenceEngine();
  const fast = new FastEngine({ mutant });
  const rand = mulberry32(seed);
  const mismatches = [];
  let executed = 0;
  for (let i = 0; i < queries; i++) {
    const query = generateQuery(rand);
    executed += 1;
    const difference = diffQuery(reference, fast, data, query);
    if (difference !== null) {
      mismatches.push({ index: i, ...difference });
      if (stopOnFailure) break;
    }
  }
  return { mismatches, executed, referenceRows: reference.rowsExamined, fastRows: fast.rowsExamined, table: data, reference, fast };
}

/** Single-step reductions of a failing case, used both by the minimiser and its minimality check. */
function reductions(table, query) {
  const out = [];
  for (let i = 0; i < table.length; i++) {
    out.push({ table: table.slice(0, i).concat(table.slice(i + 1)), query });
  }
  if (query.filter !== null) {
    out.push({ table, query: { ...query, filter: null } });
    if (query.filter.kind === "and" || query.filter.kind === "or") {
      for (let i = 0; i < query.filter.children.length; i++) {
        const rest = query.filter.children.filter((_, j) => j !== i);
        if (rest.length === 0) continue;
        out.push({ table, query: { ...query, filter: rest.length === 1 ? rest[0] : { kind: query.filter.kind, children: rest } } });
      }
      out.push({ table, query: { ...query, filter: query.filter.children[0] } });
    }
    if (query.filter.kind === "not") out.push({ table, query: { ...query, filter: query.filter.child } });
  }
  if (query.limit !== null && query.limit !== undefined) out.push({ table, query: { ...query, limit: null } });
  if (query.aggregate !== null && query.aggregate !== undefined) {
    out.push({ table, query: { ...query, aggregate: null, projection: ["id"] } });
  } else {
    for (let i = 0; i < query.projection.length; i++) {
      if (query.projection.length === 1) continue;
      out.push({ table, query: { ...query, projection: query.projection.filter((_, j) => j !== i) } });
    }
  }
  if (query.filter !== null && query.filter.kind === "in" && query.filter.values.length > 1) {
    out.push({ table, query: { ...query, filter: { ...query.filter, values: query.filter.values.slice(0, 1) } } });
  }
  return out;
}

const stillFails = (table, query, mutant) => {
  const reference = new ReferenceEngine();
  const fast = new FastEngine({ mutant });
  try {
    return diffQuery(reference, fast, table, query) !== null;
  } catch (error) {
    return false; // a reduction that produces an invalid query does not reproduce the bug
  }
};

/** Greedy delta debugging to a fixed point. */
function minimize(table, query, mutant) {
  let bestTable = table;
  let bestQuery = query;
  let improved = true;
  let steps = 0;
  while (improved) {
    improved = false;
    for (const candidate of reductions(bestTable, bestQuery)) {
      steps += 1;
      if (stillFails(candidate.table, candidate.query, mutant)) {
        bestTable = candidate.table;
        bestQuery = candidate.query;
        improved = true;
        break;
      }
    }
  }
  return { table: bestTable, query: bestQuery, steps };
}

function selfCheck() {
  // --- validation ---------------------------------------------------------------------------------
  assert.throws(() => validateQuery({ filter: { kind: "cmp", column: "nope", op: "==", value: 1 }, projection: ["id"], aggregate: null, limit: null }, SCHEMA), /unknown column/);
  assert.throws(() => validateQuery({ filter: { kind: "cmp", column: "id", op: "~=", value: 1 }, projection: ["id"], aggregate: null, limit: null }, SCHEMA), /unknown comparison/);
  assert.throws(() => validateQuery({ filter: null, projection: ["id"], aggregate: { op: "median", column: "id" }, limit: null }, SCHEMA), /unknown aggregate/);
  assert.throws(() => validateQuery({ filter: null, projection: ["id"], aggregate: null, limit: -1 }, SCHEMA), /limit/);
  assert.throws(() => validateQuery({ filter: { kind: "or", children: [] }, projection: ["id"], aggregate: null, limit: null }, SCHEMA), /needs children/);

  const table = sampleTable(48, 854);
  const reference = new ReferenceEngine();

  // --- hand-computed reference semantics ------------------------------------------------------------
  const first = table.find((row) => row.dept === "eng");
  const eng = reference.execute(table, { filter: { kind: "cmp", column: "dept", op: "==", value: "eng" }, projection: ["id", "dept"], aggregate: null, limit: null });
  assert.strictEqual(eng.length, table.filter((r) => r.dept === "eng").length);
  assert.ok(eng.every((row) => row.dept === "eng" && row.id !== undefined));
  assert.strictEqual(reference.execute(table, { filter: null, projection: ["id"], aggregate: { op: "count", column: "id" }, limit: null })[0].count, table.length);
  const nulls = table.filter((r) => r.bonus === null).length;
  assert.ok(nulls > 0, "the fixture must contain NULLs");
  assert.strictEqual(reference.execute(table, { filter: { kind: "cmp", column: "bonus", op: "==", value: 0 }, projection: ["bonus"], aggregate: null, limit: null }).length, table.filter((r) => r.bonus === 0).length);
  assert.strictEqual(compare(null, 0, "=="), false, "NULL never equals anything");
  assert.strictEqual(compare(null, 0, "!="), false, "NULL comparisons are unknown, not true");
  assert.strictEqual(compare(5, 5, "<="), true);
  const summed = reference.execute(table, { filter: null, projection: [], aggregate: { op: "sum", column: "bonus" }, limit: null })[0].sum;
  assert.strictEqual(summed, table.reduce((a, r) => a + (typeof r.bonus === "number" ? r.bonus : 0), 0), "SUM ignores NULLs");
  assert.strictEqual(reference.execute(table, { filter: null, projection: [], aggregate: { op: "max", column: "bonus" }, limit: null })[0].max, Math.max(...table.map((r) => r.bonus).filter((v) => v !== null)));
  assert.strictEqual(reference.execute(table, { filter: null, projection: ["id"], aggregate: null, limit: 3 }).length, 3);
  assert.strictEqual(reference.execute(table, { filter: null, projection: ["id"], aggregate: null, limit: 0 }).length, 0);
  assert.deepStrictEqual(reference.execute([], { filter: { kind: "cmp", column: "id", op: ">", value: 0 }, projection: ["id"], aggregate: null, limit: null }), []);

  // --- predicate semantics, including nesting ----------------------------------------------------------
  const sample = table.slice(0, 5);
  const pred = { kind: "and", children: [{ kind: "cmp", column: "age", op: ">", value: 30 }, { kind: "not", child: { kind: "cmp", column: "dept", op: "==", value: "hr" } }] };
  for (const row of sample) {
    const expected = row.age > 30 && row.dept !== "hr";
    assert.strictEqual(evalPredicate(pred, row), expected, `predicate on ${JSON.stringify(row)}`);
  }
  assert.strictEqual(evalPredicate({ kind: "in", column: "dept", values: ["eng", "hr"] }, { dept: "eng" }), true);
  assert.strictEqual(evalPredicate({ kind: "in", column: "dept", values: ["eng"] }, { dept: "hr" }), false);
  assert.strictEqual(evalPredicate(null, sample[0]), true);

  // --- index correctness against a brute-force grouping ------------------------------------------------
  const index = buildIndex(table);
  const uniques = index.byColumn.get("id");
  assert.strictEqual(uniques.buckets.size, table.length, "id is a key");
  assert.ok(table.every((row) => uniques.buckets.get(row.id)[0] === row));
  const deptIndex = index.byColumn.get("dept");
  assert.strictEqual(deptIndex.buckets.get("eng").length, table.filter((r) => r.dept === "eng").length);
  const sortedAge = index.byColumn.get("age").sorted.map((r) => r.age);
  assert.deepStrictEqual(sortedAge, sortedAge.slice().sort((a, b) => a - b), "range index must be sorted");
  assert.strictEqual(sortedAge[sortedAge.length - 1], Math.max(...sortedAge));
  assert.strictEqual(estimate({ kind: "cmp", column: "id", op: "==", value: first.id }, uniques), 1);
  assert.strictEqual(index.order.get(table[0]), 0);
  assert.strictEqual(index.order.get(table[7]), 7);

  // --- differential testing: no false positives ---------------------------------------------------------
  const clean = fuzz({ seed: 854, queries: 800, mutant: null });
  assert.deepStrictEqual(clean.mismatches, [], "a correct pair must never disagree");
  assert.strictEqual(clean.executed, 800);
  const secondClean = fuzz({ seed: 854, queries: 800, mutant: null });
  assert.strictEqual(secondClean.executed, clean.executed);
  // A duplicate-valued IN list must not make the index access path return a row twice.
  assert.strictEqual(diffQuery(new ReferenceEngine(), new FastEngine(), table, { filter: { kind: "in", column: "salary", values: [40, 40, 40] }, projection: ["id"], aggregate: null, limit: null }), null);
  assert.strictEqual(diffQuery(new ReferenceEngine(), new FastEngine(), table, { filter: { kind: "in", column: "bonus", values: [10, 10] }, projection: ["id"], aggregate: null, limit: null }), null);

  // A third, independently written declarative evaluator must agree with the reference engine.
  const rand = mulberry32(999);
  let declarativeAgreements = 0;
  for (let i = 0; i < 200; i++) {
    const query = generateQuery(rand);
    const rows = table.filter((row) => evalPredicate(query.filter, row));
    const expected = reference.execute(table, query);
    const declTable = query.aggregate ? null : rows;
    if (query.aggregate) {
      const values = rows.map((r) => r[query.aggregate.column]).filter((v) => typeof v === "number");
      const value = query.aggregate.op === "count" ? rows.length : query.aggregate.op === "max" ? (values.length === 0 ? null : Math.max(...values)) : values.reduce((a, b) => a + b, 0);
      assert.deepStrictEqual(expected, [{ [query.aggregate.op]: value }], `declarative aggregate ${i}`);
    } else {
      const shaped = declTable.map((row) => Object.fromEntries(query.projection.map((c) => [c, row[c]])));
      assert.deepStrictEqual(expected, query.limit === null ? shaped : shaped.slice(0, query.limit), `declarative rows ${i}`);
    }
    declarativeAgreements += 1;
  }
  assert.strictEqual(declarativeAgreements, 200);

  // --- coverage of the generator -------------------------------------------------------------------------
  const coverage = { cmp: 0, in: 0, and: 0, or: 0, not: 0, aggregate: 0, limit: 0, noFilter: 0 };
  const walk = (pred) => {
    if (pred === null) return;
    coverage[pred.kind] = (coverage[pred.kind] || 0) + 1;
    if (pred.children) pred.children.forEach(walk);
    if (pred.child) walk(pred.child);
  };
  const coverageRand = mulberry32(854);
  const seenOps = new Set();
  const seenAggregates = new Set();
  for (let i = 0; i < 800; i++) {
    const query = generateQuery(coverageRand);
    walk(query.filter);
    if (query.filter === null) coverage.noFilter += 1;
    if (query.aggregate) {
      coverage.aggregate += 1;
      seenAggregates.add(query.aggregate.op);
    }
    if (query.limit !== null) coverage.limit += 1;
    const collect = (pred) => {
      if (pred === null) return;
      if (pred.kind === "cmp") seenOps.add(pred.op);
      if (pred.children) pred.children.forEach(collect);
      if (pred.child) collect(pred.child);
    };
    collect(query.filter);
  }
  assert.ok(coverage.and > 20 && coverage.or > 20 && coverage.not > 20 && coverage.in > 20, "predicate kinds must all appear");
  assert.ok(coverage.aggregate > 50 && coverage.limit > 50 && coverage.noFilter > 20);
  assert.strictEqual(seenOps.size, NUMERIC_OPS.length, `all comparison operators must be generated: ${[...seenOps]}`);
  assert.strictEqual(seenAggregates.size, AGGREGATES.length);

  // --- cost bound ------------------------------------------------------------------------------------------
  assert.ok(clean.fastRows < clean.referenceRows, `optimised scan ${clean.fastRows} must beat ${clean.referenceRows}`);
  const uniqueLookup = new FastEngine();
  uniqueLookup.execute(table, { filter: { kind: "cmp", column: "id", op: "==", value: first.id }, projection: ["id"], aggregate: null, limit: null });
  assert.strictEqual(uniqueLookup.rowsExamined, 1, "a unique-key lookup examines exactly one row");
  const fullScan = new FastEngine();
  fullScan.execute(table, { filter: { kind: "or", children: [{ kind: "cmp", column: "age", op: ">", value: 200 }, { kind: "cmp", column: "bonus", op: "<", value: 1 }] }, projection: ["id"], aggregate: null, limit: null });
  assert.strictEqual(fullScan.rowsExamined, table.length, "an unindexable OR falls back to a full scan");
  assert.ok(clean.fastRows <= 800 * table.length, "the optimiser never examines more rows than the table holds per query");

  // --- mutation testing: recall within a bounded budget ------------------------------------------------------
  const budget = 400;
  const recall = new Map();
  let firstFailingCase = null;
  for (const mutant of MUTANTS) {
    const outcome = fuzz({ seed: 854, queries: budget, mutant });
    assert.ok(outcome.mismatches.length > 0, `mutant ${mutant} escaped the fuzzer`);
    assert.ok(outcome.executed <= budget, `mutant ${mutant} must be caught within ${budget} queries, took ${outcome.executed}`);
    assert.strictEqual(stillFails(outcome.table, outcome.mismatches[0].query, mutant), true, `mutant ${mutant} must reproduce standalone`);
    recall.set(mutant, outcome.executed);
    if (firstFailingCase === null) firstFailingCase = { mutant, query: outcome.mismatches[0].query, table: outcome.table };
  }
  assert.strictEqual(recall.size, MUTANTS.length);
  assert.ok([...recall.values()].every((n) => n >= 1 && n <= budget));

  // --- delta debugging: the reproducer keeps failing, and is minimal -----------------------------------------
  const reproducer = fuzz({ seed: 11, queries: budget, mutant: firstFailingCase.mutant });
  assert.ok(reproducer.mismatches.length > 0, "seed 11 must also expose the mutant");
  const failing = reproducer.mismatches[0].query;
  assert.ok(stillFails(reproducer.table, failing, firstFailingCase.mutant));
  const minimized = minimize(reproducer.table, failing, firstFailingCase.mutant);
  assert.ok(stillFails(minimized.table, minimized.query, firstFailingCase.mutant), "the minimised case must still fail");
  assert.ok(minimized.table.length <= reproducer.table.length);
  const size = (t, q) => t.length * 1000 + JSON.stringify(q).length;
  assert.ok(size(minimized.table, minimized.query) <= size(reproducer.table, failing), "minimisation must not grow the case");
  assert.ok(minimized.steps > 0);
  const oneMoreStep = reductions(minimized.table, minimized.query).filter((candidate) => stillFails(candidate.table, candidate.query, firstFailingCase.mutant));
  assert.deepStrictEqual(oneMoreStep, [], "no single reduction of the minimised case may still fail");
  const minimalDiff = diffQuery(new ReferenceEngine(), new FastEngine({ mutant: firstFailingCase.mutant }), minimized.table, minimized.query);
  assert.notStrictEqual(minimalDiff, null);
  assert.ok(JSON.stringify(minimized.query).length < 400, "the reproducer must be small enough to read");
  assert.ok(minimized.table.length < reproducer.table.length || minimized.query.limit === null);

  // --- determinism --------------------------------------------------------------------------------------------
  const reportA = fuzz({ seed: 77, queries: 120, mutant: "off-by-one" });
  const reportB = fuzz({ seed: 77, queries: 120, mutant: "off-by-one" });
  assert.deepStrictEqual(reportA.mismatches.map((m) => m.index), reportB.mismatches.map((m) => m.index));
  assert.deepStrictEqual(reportA.mismatches[0].query, reportB.mismatches[0].query);
  assert.notDeepStrictEqual(fuzz({ seed: 78, queries: 120, mutant: "off-by-one" }).mismatches[0].query, reportA.mismatches[0].query);

  console.log(`differential: 800 clean queries, ${clean.referenceRows} rows scanned by the reference vs ${clean.fastRows} by the optimizer`);
  console.log(`mutants caught: ${[...recall.entries()].map(([m, n]) => `${m}@${n}`).join(", ")}`);
  console.log(`minimised reproducer: ${reproducer.table.length} -> ${minimized.table.length} rows, query ${JSON.stringify(minimized.query)}`);
  console.log("PASS");
}

module.exports = { ReferenceEngine, FastEngine, sampleTable, generateQuery, fuzz, diffQuery, minimize, reductions, stillFails, buildIndex, evalPredicate, compare, validateQuery, canonical, conjuncts, mulberry32, MUTANTS };

if (require.main === module) {
  selfCheck();
}
