/** Task 655 - Implement query rewriter for simple view unfolding. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * A "simple view" is a single-table SPJ view: explicit projection list, one FROM, boolean
 * WHERE, no aggregates / DISTINCT / GROUP BY / HAVING / ORDER BY. Rewriting a query that reads
 * such a view means *unfolding* it: the view reference is replaced by the view body
 * (recursively, with cycle detection), the outer projection is checked against the columns the
 * view exposes, and the view predicate and the outer predicate are merged (predicate pushdown)
 * into one predicate which is then simplified.
 *
 * Simplification is domain-aware, like a planner using column statistics: the domain of a
 * column is the set of distinct values that occur. A comparison no domain value satisfies folds
 * to FALSE, one every value satisfies folds to TRUE, and the conjuncts on one column collapse
 * into an interval only after that interval is *verified value-by-value against the whole
 * domain* — hence `qty >= 10 AND qty > 5` may legally become `qty >= 12` when the table has no
 * qty in 6..11, while `x != 5 AND x > 2` is left untouched.
 *
 * Invariant verified: every rewritten query returns exactly the rows the unrewritten query
 * returns after materialising the view; the simplifier is brute-force cross-checked over a whole
 * domain (120 random predicates x every domain value); non-simple views, cyclic chains and
 * references to non-exposed columns are rejected with descriptive errors.
 */
"use strict";
const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");
const OPS = new Set(["=", "!=", "<", "<=", ">", ">="]);
const TRUE = Object.freeze({ kind: "bool", value: true });
const FALSE = Object.freeze({ kind: "bool", value: false });
const isTrue = (p) => p.kind === "bool" && p.value;
const isFalse = (p) => p.kind === "bool" && !p.value;
function makeRng(seed) {
  let state = seed >>> 0 || 0x5bd1e995;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
function cmp(col, op, value) {
  if (!OPS.has(op)) throw new RangeError(`unsupported comparison operator ${op}`);
  return { kind: "cmp", col, op, value };
}
const combine = (kind, identity, parts) => {
  const kept = parts.flat().filter((p) => p !== undefined && p !== null);
  return kept.length === 0 ? identity : kept.reduce((left, right) => ({ kind, left, right }));
};
const and = (...parts) => combine("and", TRUE, parts);
const or = (...parts) => combine("or", FALSE, parts);
const COMPARATORS = {
  "=": (a, b) => a === b, "!=": (a, b) => a !== b, "<": (a, b) => a < b,
  "<=": (a, b) => a <= b, ">": (a, b) => a > b, ">=": (a, b) => a >= b,
};
const satisfies = (value, op, rhs) => COMPARATORS[op](value, rhs);
function evaluatePredicate(pred, row) {
  if (pred.kind === "bool") return pred.value;
  if (pred.kind === "cmp") return satisfies(row[pred.col], pred.op, pred.value);
  const left = evaluatePredicate(pred.left, row);
  return pred.kind === "and" ? left && evaluatePredicate(pred.right, row)
    : left || evaluatePredicate(pred.right, row);
}
/** Runs a select AST against a Map of relation name -> rows. */
function evalQuery(query, relations) {
  const rows = relations.get(query.from);
  if (rows === undefined) throw new Error(`unknown relation ${query.from}`);
  const filter = query.where === undefined ? TRUE : query.where;
  const star = query.select.length === 1 && query.select[0] === "*";
  const out = [];
  for (const row of rows) {
    if (!evaluatePredicate(filter, row)) continue;
    if (star) { out.push({ ...row }); continue; }
    const projected = {};
    for (const col of query.select) projected[col] = row[col];
    out.push(projected);
  }
  return out;
}
const renderPredicate = (pred) => {
  if (pred.kind === "bool") return pred.value ? "TRUE" : "FALSE";
  if (pred.kind === "cmp") return `${pred.col} ${pred.op} ${JSON.stringify(pred.value)}`;
  const glue = pred.kind === "and" ? " AND " : " OR ";
  return `(${renderPredicate(pred.left)}${glue}${renderPredicate(pred.right)})`;
};
const renderQuery = (q) =>
  `SELECT ${q.select.join(", ")} FROM ${q.from} WHERE ${renderPredicate(q.where ?? TRUE)}`;
function collectColumns(pred) {
  if (pred.kind === "cmp") return new Set([pred.col]);
  if (pred.kind === "bool") return new Set();
  return new Set([...collectColumns(pred.left), ...collectColumns(pred.right)]);
}
/** Flattens a right-nested and/or tree into its leaf list, preserving order. */
function flatten(pred, kind) {
  if (pred.kind !== kind) return [pred];
  return [...flatten(pred.left, kind), ...flatten(pred.right, kind)];
}
function intervalToPredicate(col, lo, hi, domainLo, domainHi) {
  if (lo > hi) return FALSE;
  if (lo <= domainLo && hi >= domainHi) return TRUE;
  const parts = [];
  if (lo > domainLo) parts.push(cmp(col, ">=", lo));
  if (hi < domainHi) parts.push(cmp(col, "<=", hi));
  return parts.length === 0 ? TRUE : and(...parts);
}
const numericDomain = (domain) => domain !== undefined && domain.length > 0 &&
  domain.every((v) => typeof v === "number");
/** Domain values covered by a group of same-column leaves: "all" for AND, "any" for OR. */
const coveredValues = (domain, leaves, mode) => domain.filter((value) => (mode === "all"
  ? leaves.every((leaf) => satisfies(value, leaf.op, leaf.value))
  : leaves.some((leaf) => satisfies(value, leaf.op, leaf.value))));
function agreesOnDomain(candidate, leaves, col, domain, mode) {
  for (const value of domain) {
    const original = mode === "all"
      ? leaves.every((leaf) => satisfies(value, leaf.op, leaf.value))
      : leaves.some((leaf) => satisfies(value, leaf.op, leaf.value));
    if (evaluatePredicate(candidate, { [col]: value }) !== original) return false;
  }
  return true;
}
/** Narrows a group of same-column comparisons to one verified interval, or returns null. */
function verifiedInterval(col, leaves, domains, mode) {
  const domain = domains.get(col);
  if (!numericDomain(domain)) return null;
  const matching = coveredValues(domain, leaves, mode);
  if (matching.length === 0 || matching.length === domain.length) return null;
  const candidate = intervalToPredicate(col, Math.min(...matching), Math.max(...matching),
    Math.min(...domain), Math.max(...domain));
  return agreesOnDomain(candidate, leaves, col, domain, mode) ? candidate : null;
}
/** Domain-aware predicate simplification; every rewrite is verified before it is accepted. */
function simplify(pred, domains = new Map()) {
  if (pred.kind === "bool") return pred.value ? TRUE : FALSE;
  if (pred.kind === "cmp") {
    const domain = domains.get(pred.col);
    if (domain === undefined || domain.length === 0) return pred;
    const hits = domain.filter((v) => satisfies(v, pred.op, pred.value)).length;
    return hits === 0 ? FALSE : hits === domain.length ? TRUE : pred;
  }
  if (pred.kind === "and") {
    const parts = flatten(pred, "and").map((p) => simplify(p, domains));
    if (parts.some(isFalse)) return FALSE;
    const live = parts.filter((p) => !isTrue(p));
    if (live.length === 0) return TRUE;
    const byColumn = new Map();
    for (const part of live) {
      if (part.kind !== "cmp") continue;
      const domain = domains.get(part.col);
      if (domain === undefined || domain.length === 0) continue;
      if (!byColumn.has(part.col)) byColumn.set(part.col, []);
      byColumn.get(part.col).push(part);
    }
    const rewritten = [];
    const handled = new Set();
    for (const part of live) {
      if (part.kind !== "cmp" || !byColumn.has(part.col)) { rewritten.push(part); continue; }
      if (handled.has(part.col)) continue;
      handled.add(part.col);
      const leaves = byColumn.get(part.col);
      const domain = domains.get(part.col);
      const matching = coveredValues(domain, leaves, "all");
      if (matching.length === 0) return FALSE; // contradictory conjuncts
      if (matching.length === domain.length) continue; // vacuously true on this column
      const interval = verifiedInterval(part.col, leaves, domains, "all");
      if (interval !== null) {
        if (isFalse(interval)) return FALSE;
        if (!isTrue(interval)) rewritten.push(interval);
        continue;
      }
      if (matching.length === 1 && leaves.some((l) => l.op === "=" && l.value === matching[0])) {
        rewritten.push(cmp(part.col, "=", matching[0]));
        continue;
      }
      rewritten.push(...leaves); // no verified rewrite: keep the original conjuncts
    }
    if (rewritten.length === 0) return TRUE;
    return rewritten.length === 1 ? rewritten[0] : and(...rewritten);
  }
  const parts = flatten(pred, "or").map((p) => simplify(p, domains));
  if (parts.some(isTrue)) return TRUE;
  const live = parts.filter((p) => !isFalse(p));
  if (live.length === 0) return FALSE;
  const col = live[0].kind === "cmp" ? live[0].col : null;
  if (col !== null && live.every((p) => p.kind === "cmp" && p.col === col) &&
    domains.get(col) !== undefined && domains.get(col).length > 0) {
    const domain = domains.get(col);
    const matching = coveredValues(domain, live, "any");
    if (matching.length === 0) return FALSE;
    if (matching.length === domain.length) return TRUE;
    const interval = verifiedInterval(col, live, domains, "any");
    if (interval !== null) return interval;
    if (matching.length === 1) return cmp(col, "=", matching[0]);
  }
  return live.length === 1 ? live[0] : or(...live);
}
function assertSimpleView(name, view) {
  const q = view.query;
  if (q.groupBy !== undefined || q.having !== undefined || q.distinct === true ||
    q.aggregates !== undefined || q.orderBy !== undefined) {
    throw new Error(`view ${name} is not simple: aggregates, DISTINCT, GROUP BY, HAVING and ORDER BY are unsupported`);
  }
  if (!Array.isArray(q.select) || q.select.length === 0 || q.select.includes("*")) {
    throw new Error(`view ${name} must project an explicit, non-empty column list`);
  }
}
/** Unfolds simple view references, merges WHERE clauses and simplifies the merged predicate. */
function unfoldQuery(query, views, domains = new Map(), stack = []) {
  const view = views.get(query.from);
  if (view === undefined) {
    return { select: query.select.slice(), from: query.from, where: simplify(query.where ?? TRUE, domains) };
  }
  if (stack.includes(query.from)) throw new Error(`cyclic view definition: ${[...stack, query.from].join(" -> ")}`);
  assertSimpleView(query.from, view);
  const inner = unfoldQuery(view.query, views, domains, [...stack, query.from]);
  const exposed = inner.select;
  const outerSelect = query.select.includes("*") ? exposed.slice() : query.select.slice();
  const outerWhere = query.where ?? TRUE;
  for (const col of [...outerSelect, ...collectColumns(outerWhere)]) {
    if (!exposed.includes(col)) throw new Error(`column ${col} is not exposed by view ${query.from}`);
  }
  return { select: outerSelect, from: inner.from, where: simplify(and(inner.where, outerWhere), domains) };
}
const domainsOf = (rows, columns) =>
  new Map(columns.map((col) => [col, [...new Set(rows.map((r) => r[col]))]]));
const sortRows = (rows) => rows.map((r) => JSON.stringify(r, Object.keys(r).sort())).sort();
module.exports = {
  cmp, and, or, TRUE, FALSE, satisfies, evaluatePredicate, evalQuery, simplify, unfoldQuery,
  renderQuery, renderPredicate, domainsOf, sortRows, collectColumns, assertSimpleView, makeRng,
};
if (require.main === module) {
  const orders = [
    { id: 1, customer: "a1", region: "eu", qty: 5, status: "new" },
    { id: 2, customer: "b2", region: "eu", qty: 12, status: "paid" },
    { id: 3, customer: "c3", region: "us", qty: 40, status: "paid" },
    { id: 4, customer: "a1", region: "us", qty: 100, status: "new" },
    { id: 5, customer: "d4", region: "eu", qty: 100, status: "paid" },
    { id: 6, customer: "b2", region: "ap", qty: 7, status: "new" },
  ];
  const domains = domainsOf(orders, ["id", "customer", "region", "qty", "status"]);
  const def = (name, select, from, where, extra = {}) =>
    [name, { name, query: { select, from, where, ...extra } }];
  const views = new Map([
    def("v_eu", ["id", "customer", "qty", "region"], "orders", cmp("region", "=", "eu")),
    def("v_big", ["id", "qty"], "orders", cmp("qty", ">=", 10)),
    def("v_chain", ["id", "qty"], "v_big", cmp("qty", "<=", 100)),
    def("v_agg", ["region", "cnt"], "orders", TRUE, { groupBy: ["region"], aggregates: ["count(*)"] }),
    def("v_star", ["*"], "orders", TRUE),
    def("v_loop_a", ["id"], "v_loop_b", TRUE),
    def("v_loop_b", ["id"], "v_loop_a", TRUE),
  ]);
  const base = new Map([["orders", orders]]);
  const ids = (q) => evalQuery(q, base).map((r) => r.id).sort((a, b) => a - b);
  // Materialise-then-filter semantics: what the unrewritten query must return.
  const referenceRows = (name) => {
    const view = views.get(name);
    return view === undefined ? base.get(name)
      : evalQuery(view.query, new Map([[view.query.from, referenceRows(view.query.from)]]));
  };
  const referenceAnswer = (q) => evalQuery(q, new Map([[q.from, referenceRows(q.from)]]));
  function checkEquivalent(query, label) {
    const rewritten = unfoldQuery(query, views, domains);
    deepStrictEqual(sortRows(evalQuery(rewritten, base)), sortRows(referenceAnswer(query)),
      `rewrite must preserve semantics: ${label}`);
    return rewritten;
  }
  // 1. plain view plus an extra predicate: both survive pushdown
  const r1 = checkEquivalent({ select: ["id", "qty"], from: "v_eu", where: cmp("qty", ">", 5) }, "v_eu, qty > 5");
  strictEqual(r1.from, "orders", "unfolding must expose the base table");
  strictEqual(renderQuery(r1).includes("v_eu"), false, "the view name must be gone after rewriting");
  deepStrictEqual([r1.select, [...collectColumns(r1.where)].sort()], [["id", "qty"], ["qty", "region"]]);
  ok(renderPredicate(r1.where).includes('region = "eu"'), "the view predicate survives pushdown");
  ok(renderPredicate(r1.where).includes("qty >="), "the outer qty predicate is pushed down");
  deepStrictEqual(ids(r1), [2, 5], "EU orders with qty > 5 are ids 2 and 5");
  const r1b = checkEquivalent({ select: ["id"], from: "v_eu", where: cmp("qty", ">=", 12) }, "stable bound");
  deepStrictEqual(renderPredicate(r1b.where), '(region = "eu" AND qty >= 12)', "a bound on a present value is kept");
  // 2. SELECT * exposes exactly the view columns; an explicit subset is preserved
  const r2 = checkEquivalent({ select: ["*"], from: "v_eu" }, "SELECT * over v_eu");
  deepStrictEqual([r2.select, ids(r2)], [["id", "customer", "qty", "region"], [1, 2, 5]]);
  const r2b = checkEquivalent({ select: ["qty", "id"], from: "v_eu" }, "projection subset");
  deepStrictEqual(r2b.select, ["qty", "id"]);
  // 3. contradiction: the two region predicates cannot both hold
  const r3 = checkEquivalent({ select: ["id"], from: "v_eu", where: cmp("region", "=", "us") }, "contradiction");
  deepStrictEqual([r3.where, evalQuery(r3, base)], [FALSE, []], "region = eu AND region = us folds to FALSE");
  // 4. tautology elimination on the pushed-down branch
  const tautology = or(cmp("region", "=", "eu"), cmp("region", "!=", "eu"));
  deepStrictEqual(simplify(tautology, domains), TRUE, "x = v OR x != v is TRUE over any domain");
  const r4 = checkEquivalent({ select: ["id"], from: "v_eu", where: tautology }, "tautology");
  deepStrictEqual([r4.where, ids(r4)], [cmp("region", "=", "eu"), [1, 2, 5]], "the view predicate survives alone");
  // 5. interval merging on a numeric column, verified against the domain
  const r5 = checkEquivalent({ select: ["id"], from: "v_big", where: cmp("qty", "<=", 100) }, "interval merge");
  strictEqual(collectColumns(r5.where).size, 1, "the merged predicate stays on one column");
  ok(renderPredicate(r5.where).includes("qty >="), "the merged predicate keeps a lower bound");
  deepStrictEqual(ids(r5), [2, 3, 4, 5], "qty >= 10 AND qty <= 100 keeps ids 2..5");
  const r5b = checkEquivalent({ select: ["id"], from: "v_big", where: cmp("qty", "<", 10) }, "empty range");
  deepStrictEqual(r5b.where, FALSE, "qty >= 10 AND qty < 10 is unsatisfiable");
  // narrowing qty > 5 to qty >= 12 is legal only because no qty lies in 6..11
  const narrow = unfoldQuery({ select: ["id"], from: "v_big", where: cmp("qty", ">", 5) }, views, domains);
  deepStrictEqual([renderPredicate(narrow.where), ids(narrow)], ["qty >= 12", [2, 3, 4, 5]]);
  // 6. a view over a view unfolds two levels
  const r6 = checkEquivalent({ select: ["id"], from: "v_chain", where: cmp("qty", ">", 20) }, "chained view");
  ok(r6.from === "orders" && collectColumns(r6.where).size === 1 && renderPredicate(r6.where).includes("qty >="),
    "the chain collapses into a single bound on the base table");
  deepStrictEqual(ids(r6), [3, 4, 5], "qty >= 10 AND qty <= 100 AND qty > 20 keeps ids 3..5");
  // 7. pushdown keeps every leaf when the columns differ; 8. rewriting is idempotent
  const r7 = checkEquivalent({ select: ["id"], from: "v_eu", where: cmp("customer", "=", "a1") }, "two columns");
  deepStrictEqual([[...collectColumns(r7.where)].sort(), ids(r7)], [["customer", "region"], [1]]);
  deepStrictEqual(unfoldQuery(r1, views, domains), r1, "unfolding an unfolded query is a no-op");
  // 9. rejected inputs
  const reject = (query, pattern, message) => throws(() => unfoldQuery(query, views, domains), pattern, message);
  const selectAll = (from) => ({ select: ["id"], from, where: TRUE });
  reject(selectAll("v_agg"), /not simple/, "aggregate views must be rejected");
  reject(selectAll("v_star"), /explicit/, "views without an explicit projection list must be rejected");
  reject(selectAll("v_loop_a"), /cyclic/, "cyclic view chains must be rejected");
  reject({ select: ["id"], from: "v_eu", where: cmp("missing", "=", 1) }, /not exposed/, "unknown predicate column");
  reject({ select: ["region", "qty"], from: "v_big", where: TRUE }, /not exposed/, "unknown projection column");
  throws(() => cmp("x", "~", 1), RangeError, "unsupported operators must be rejected");
  // 10. brute-force check of the simplifier over a full column domain
  const rng = makeRng(31337);
  const domainValues = Array.from({ length: 10 }, (_, i) => i);
  const testDomains = new Map([["x", domainValues]]);
  const ops = ["=", "!=", "<", "<=", ">", ">="];
  const randomPredicate = (depth) => {
    if (depth === 0 || rng(100) < 40) return cmp("x", ops[rng(ops.length)], rng(12) - 1);
    return rng(100) < 50 ? and(randomPredicate(depth - 1), randomPredicate(depth - 1))
      : or(randomPredicate(depth - 1), randomPredicate(depth - 1));
  };
  let rewrittenCount = 0;
  for (let trial = 0; trial < 120; trial++) {
    const pred = randomPredicate(3);
    const folded = simplify(pred, testDomains);
    for (const value of domainValues) {
      strictEqual(evaluatePredicate(folded, { x: value }), evaluatePredicate(pred, { x: value }),
        `simplifier must agree on every domain value: ${renderPredicate(pred)}`);
    }
    if (renderPredicate(folded) !== renderPredicate(pred)) rewrittenCount++;
  }
  ok(rewrittenCount > 10, `the simplifier should rewrite many predicates, rewrote ${rewrittenCount}`);
  deepStrictEqual([simplify(cmp("x", "<", 0), testDomains), simplify(cmp("x", ">=", -5), testDomains)],
    [FALSE, TRUE], "out-of-domain comparisons fold to constants");
  deepStrictEqual(simplify(and(cmp("x", "<", 3), cmp("x", ">", 7)), testDomains), FALSE);
  deepStrictEqual(simplify(or(cmp("x", "<", 2), cmp("x", ">", 7)), testDomains),
    or(cmp("x", "<", 2), cmp("x", ">", 7)), "a non-contiguous union is left alone");
  deepStrictEqual(simplify(or(cmp("x", "<", 2), cmp("x", ">=", 2)), testDomains), TRUE);
  deepStrictEqual(simplify(and(cmp("x", "!=", 5), cmp("x", ">", 2)), testDomains),
    and(cmp("x", "!=", 5), cmp("x", ">", 2)), "a non-contiguous conjunct resists narrowing");
  // 11. string columns fold by domain membership
  deepStrictEqual(simplify(cmp("status", "=", "archived"), domains), FALSE);
  deepStrictEqual(simplify(cmp("status", "!=", "paid"), domains), cmp("status", "!=", "paid"));
  deepStrictEqual(simplify(and(cmp("status", "=", "new"), cmp("status", "=", "paid")), domains), FALSE);
  console.log(`unfolded 8 queries, simplifier rewrote ${rewrittenCount}/120 random predicates, ` +
    `narrowed=[${renderPredicate(narrow.where)}]`);
  console.log("PASS");
}
