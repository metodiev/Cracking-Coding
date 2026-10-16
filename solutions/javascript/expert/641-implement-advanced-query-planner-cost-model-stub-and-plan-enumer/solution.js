/** Task 641 - Implement advanced query planner cost model stub and plan enumeration. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMPLIFIED (the "stub"): the catalog carries row counts, distinct-value counts and per-column
 * histograms, but no real tuples, no buffer pool, no I/O calibration and no parallelism. Costs are
 * therefore counted in *pages read*, using documented textbook formulas rather than measured constants:
 *
 *   pages(rows)             = ceil(rows / rowsPerPage)
 *   scan(T)                 = pages(rows(T))                     (a base table scan)
 *   selectivity(T, p)       = 1 / distinct(T.c)                  for "= c"
 *                           = histogram fraction                for < <= > >=
 *   rows(T, p1..pk)         = rows(T) * product selectivity(T, pi)     (independence assumption)
 *   rows(L join R, j)       = rows(L) * rows(R) * 1 / max(distinct(L.c), distinct(R.c))
 *   nested loop             = cost(L) + rows(L) * cost(R)             (R is re-read per outer row)
 *   hash join               = cost(L) + cost(R) + 3 * (pages(rows(L)) + pages(rows(R)))
 *   sort-merge join         = cost(L) + cost(R) + 2 * sort(pages(rows(L))) + sort(pages(rows(R)))
 *
 * THE INVARIANT VERIFIED: because the output-row estimate of a join depends only on the *subset* of
 * relations it contains (every join predicate is applied exactly once and the leaf selectivities are
 * fixed), the cost of extending a plan depends only on the sub-plan's cost and row estimate - exactly
 * Selinger's precondition for "keep the cheapest plan per subset". The checker therefore compares the
 * dynamic-programming planner against an exhaustive enumeration of *every* bushy join tree with every
 * method assignment and asserts the two best costs agree, while reporting how many partial merges the DP
 * had to evaluate instead of materialising every complete plan.
 *
 * WHAT ELSE IS VERIFIED: equality selectivity is 1/distinct; range selectivity matches the true row count
 * when the histogram has one bucket per value, and is bounded by one bucket's worth of rows for coarse
 * histograms; conjunction selectivity is the product and stays inside [0, 1]; histogram counts must sum
 * to the row count; a fully covered range is 100% and an empty range 0%; disconnected join graphs,
 * unknown relations/columns and unknown join methods are rejected; and the best plan's reported cost and
 * row estimate can be recomputed from the plan tree itself.
 */
"use strict";

class Catalog {
  constructor(rowsPerPage = 40) {
    this.rowsPerPage = rowsPerPage;
    this.tables = new Map();
  }
  addTable(name, rows, columns) {
    if (this.tables.has(name)) throw new Error(`table ${name} is already in the catalog`);
    const entry = { name, rows, columns: new Map() };
    for (const column of columns) {
      if (column.histogram) {
        const total = column.histogram.reduce((sum, bucket) => sum + bucket.count, 0);
        if (total !== rows) throw new Error(`histogram of ${name}.${column.name} sums to ${total}, not ${rows}`);
        for (const bucket of column.histogram) {
          if (!(bucket.count > 0)) throw new Error(`empty histogram bucket in ${name}.${column.name}`);
        } }
      entry.columns.set(column.name, column);
    }
    this.tables.set(name, entry);
    return entry;
  }
  table(name) {
    const entry = this.tables.get(name);
    if (!entry) throw new Error(`unknown table ${name}`);
    return entry;
  }
  rows(name) {
    return this.table(name).rows;
  }
  column(table, column) {
    const entry = this.table(table).columns.get(column);
    if (!entry) throw new Error(`unknown column ${table}.${column}`);
    return entry;
  }
  pages(rows) {
    return Math.max(1, Math.ceil(rows / this.rowsPerPage));
  }
  /** Range selectivity: the histogram's bucket counts are spread uniformly over each bucket's width. */
  selectivity(tableName, predicate) {
    const table = this.table(tableName);
    const column = table.columns.get(predicate.column);
    if (!column) throw new Error(`unknown column ${tableName}.${predicate.column}`);
    if (predicate.op === "=") {
      if (!(column.distinct > 0)) throw new Error(`column ${tableName}.${predicate.column} has no distinct count`);
      return 1 / column.distinct;
    }
    if (!["<", "<=", ">", ">="].includes(predicate.op)) throw new Error(`unknown operator ${predicate.op}`);
    if (!column.histogram) throw new Error(`column ${tableName}.${predicate.column} has no histogram`);
    let matching = 0;
    for (const bucket of column.histogram) {
      const width = bucket.hi - bucket.lo + 1;
      let covered;
      if (predicate.op === "<") covered = Math.min(bucket.hi, predicate.value - 1) - bucket.lo + 1;
      else if (predicate.op === "<=") covered = Math.min(bucket.hi, predicate.value) - bucket.lo + 1;
      else if (predicate.op === ">") covered = bucket.hi - Math.max(bucket.lo, predicate.value + 1) + 1;
      else covered = bucket.hi - Math.max(bucket.lo, predicate.value) + 1;
      matching += bucket.count * (Math.max(0, Math.min(covered, width)) / width);
    }
    return matching / table.rows;
  }
  /** Independence assumption; the result never leaves [0, 1]. */
  conjunctionSelectivity(tableName, predicates = []) {
    let selectivity = 1;
    for (const predicate of predicates) selectivity *= this.selectivity(tableName, predicate);
    return Math.min(1, Math.max(0, selectivity));
  } }

function leafPlan(query, tableName) {
  return { kind: "leaf", table: tableName, predicates: query.filters[tableName] || [], query };
}

function joinSelectivity(catalog, predicate) {
  const left = catalog.column(predicate.left.table, predicate.left.column);
  const right = catalog.column(predicate.right.table, predicate.right.column);
  return 1 / Math.max(left.distinct, right.distinct);
}

function joinPlan(method, left, right, predicates, selectivity) {
  if (!["NL", "HASH", "SM"].includes(method)) throw new Error(`unknown join method ${method}`);
  return { kind: "join", method, left, right, predicates, selectivity, query: left.query };
}

function planTableSet(plan) {
  if (!plan.tableSet) {
    plan.tableSet = plan.kind === "leaf"
      ? [plan.table]
      : [...planTableSet(plan.left), ...planTableSet(plan.right)].sort();
  }
  return plan.tableSet;
}

function subsetTables(query, mask) {
  return query.tables.filter((_, index) => (mask & (1 << index)) !== 0);
}

function joinPredicatesBetween(query, mask, table) {
  return query.joins.filter((j) => {
    const leftIn = (mask & (1 << query.tables.indexOf(j.left.table))) !== 0;
    const rightIn = (mask & (1 << query.tables.indexOf(j.right.table))) !== 0;
    return (leftIn && j.right.table === table) || (rightIn && j.left.table === table);
  });
}

/** Row estimate of a plan: a canonical product over its *subset* of relations, so it never depends on
 * the shape of the tree - that is what makes "cheapest plan per subset" valid for the dynamic program. */
function planRows(plan, catalog) {
  const tables = planTableSet(plan);
  let rows = 1;
  for (const table of tables) {
    rows *= catalog.rows(table) * catalog.conjunctionSelectivity(table, plan.query.filters[table] || []);
  }
  for (const join of plan.query.joins) {
    if (tables.includes(join.left.table) && tables.includes(join.right.table)) {
      rows *= joinSelectivity(catalog, join);
    } }
  return Math.max(1, rows);
}

function sortCost(pages) {
  return 2 * pages * Math.ceil(Math.log2(Math.max(2, pages)));
}

function planCost(plan, catalog) {
  if (plan.kind === "leaf") return catalog.pages(catalog.rows(plan.table));
  const left = planCost(plan.left, catalog);
  const right = planCost(plan.right, catalog);
  const leftPages = catalog.pages(planRows(plan.left, catalog));
  const rightPages = catalog.pages(planRows(plan.right, catalog));
  if (plan.method === "NL") return left + planRows(plan.left, catalog) * right;
  if (plan.method === "HASH") return left + right + 3 * (leftPages + rightPages);
  return left + right + sortCost(leftPages) + sortCost(rightPages) + leftPages + rightPages;
}

function describePlan(plan) {
  if (plan.kind === "leaf") {
    const filters = plan.predicates.map((p) => `${p.column}${p.op}${p.value ?? ""}`).join(",");
    return `Scan(${plan.table}${filters ? ` [${filters}]` : ""})`;
  }
  return `${plan.method}(${describePlan(plan.left)}, ${describePlan(plan.right)})`;
}

function checkQuery(query) {
  for (const table of query.tables) {
    if (query.tables.indexOf(table) !== query.tables.lastIndexOf(table)) {
      throw new Error(`table ${table} is listed twice`);
    } }
  const index = new Map(query.tables.map((table, position) => [table, position]));
  const reach = (mask, table) => query.joins.some((join) => {
    const leftIn = (mask & (1 << index.get(join.left.table))) !== 0;
    const rightIn = (mask & (1 << index.get(join.right.table))) !== 0;
    return (leftIn && join.right.table === table) || (rightIn && join.left.table === table);
  });
  const full = (1 << query.tables.length) - 1;
  const reachable = new Set([1]);
  let changed = true;
  while (changed) { // grow every subset that can be joined from an already reachable one
    changed = false;
    for (const mask of [...reachable]) {
      for (const table of query.tables) {
        if ((mask & (1 << index.get(table))) !== 0) continue;
        if (!reach(mask, table)) continue;
        const next = mask | (1 << index.get(table));
        if (!reachable.has(next)) {
          reachable.add(next);
          changed = true;
        } }
    } }
  if (!reachable.has(full)) throw new Error("the join graph is disconnected");
  return query;
}

/** Every bushy join tree over every connected split, with every method: exponential by design. */
function exhaustivePlans(query, catalog, methods = ["NL", "HASH", "SM"]) {
  checkQuery(query);
  const cache = new Map();
  const plansFor = (mask) => {
    if (cache.has(mask)) return cache.get(mask);
    const tables = subsetTables(query, mask);
    let plans;
    if (tables.length === 1) {
      plans = [leafPlan(query, tables[0])];
    } else {
      plans = [];
      for (let sub = (mask - 1) & mask; sub > 0; sub = (sub - 1) & mask) {
        const rest = mask & ~sub;
        if (rest === 0) continue;
        const predicates = query.joins.filter((j) => {
          const left = query.tables.indexOf(j.left.table);
          const right = query.tables.indexOf(j.right.table);
          return ((sub >> left) & 1) !== ((sub >> right) & 1) && ((mask >> left) & 1) && ((mask >> right) & 1);
        });
        if (predicates.length === 0) continue; // a cross product is not a candidate in this planner
        for (const left of plansFor(sub)) for (const right of plansFor(rest)) {
          for (const method of methods) {
            plans.push(joinPlan(method, left, right, predicates,
              predicates.reduce((s, j) => Math.min(s, joinSelectivity(catalog, j)), 1)));
          } }
      } }
    cache.set(mask, plans);
    return plans;
  };
  const all = plansFor((1 << query.tables.length) - 1);
  let best = null;
  for (const plan of all) {
    const cost = planCost(plan, catalog);
    if (!best || cost < best.cost) best = { plan, cost };
  }
  return { all, best, candidates: all.length };
}

/** Selinger's dynamic program: keep only the cheapest plan per connected subset. */
function selingerDp(query, catalog, methods = ["NL", "HASH", "SM"]) {
  checkQuery(query);
  const best = new Map();
  let merges = 0;
  for (let index = 0; index < query.tables.length; index++) {
    const plan = leafPlan(query, query.tables[index]);
    best.set(1 << index, { plan, cost: planCost(plan, catalog) });
  }
  const full = (1 << query.tables.length) - 1;
  for (let mask = 1; mask <= full; mask++) {
    if ((mask & (mask - 1)) === 0) continue; // singletons are already placed
    let winner = null;
    for (let sub = (mask - 1) & mask; sub > 0; sub = (sub - 1) & mask) {
      const rest = mask & ~sub;
      if (rest === 0 || !best.has(sub) || !best.has(rest)) continue;
      const predicates = query.joins.filter((j) => {
        const left = query.tables.indexOf(j.left.table);
        const right = query.tables.indexOf(j.right.table);
        return ((sub >> left) & 1) !== ((sub >> right) & 1) && ((mask >> left) & 1) && ((mask >> right) & 1);
      });
      if (predicates.length === 0) continue;
      const selectivity = predicates.reduce((s, j) => Math.min(s, joinSelectivity(catalog, j)), 1);
      for (const method of methods) {
        merges++;
        const plan = joinPlan(method, best.get(sub).plan, best.get(rest).plan, predicates, selectivity);
        const cost = planCost(plan, catalog);
        if (!winner || cost < winner.cost) winner = { plan, cost };
      } }
    if (winner) best.set(mask, winner);
  }
  return { plan: best.get(full).plan, cost: best.get(full).cost, merges };
}

module.exports = { Catalog, planRows, planCost, describePlan, exhaustivePlans, selingerDp, joinSelectivity, sortCost };

if (require.main === module) {
  const assert = require("node:assert");
  const catalog = new Catalog(40);
  // A value-per-bucket histogram makes range selectivity exact, so it can be checked against real rows.
  const counts = new Map([[1, 5], [2, 10], [3, 4], [4, 1], [5, 20], [6, 15], [7, 2], [8, 3]]);
  const histogram = [...counts].map(([value, count]) => ({ lo: value, hi: value, count }));
  const total = [...counts.values()].reduce((a, b) => a + b, 0);
  catalog.addTable("small", total, [
    { name: "id", distinct: total, histogram }, ]);
  catalog.addTable("big", 4000, [{ name: "id", distinct: 800 }]);
  catalog.addTable("mid", 500, [{ name: "id", distinct: 50 }]);
  assert.throws(() => catalog.addTable("broken", 7, [{ name: "id", distinct: 7, histogram: [{ lo: 1, hi: 1, count: 3 }] }]),
    /sums to 3, not 7/);
  const actualRows = [...counts.entries()].flatMap(([value, count]) => Array.from({ length: count }, () => value));
  for (const predicate of [
    { column: "id", op: "<", value: 4 }, { column: "id", op: "<=", value: 4 },
    { column: "id", op: ">", value: 5 }, { column: "id", op: ">=", value: 5 },
    { column: "id", op: "<", value: 1 }, { column: "id", op: ">", value: 8 },
    { column: "id", op: ">=", value: 0 }, { column: "id", op: "<=", value: 8 }, ]) {
    const truth = actualRows.filter((value) => predicate.op === "<" ? value < predicate.value
      : predicate.op === "<=" ? value <= predicate.value
        : predicate.op === ">" ? value > predicate.value : value >= predicate.value).length;
    assert.strictEqual(catalog.selectivity("small", predicate) * total, truth,
      `${predicate.op}${predicate.value} must match the true row count`);
  }
  assert.strictEqual(catalog.selectivity("big", { column: "id", op: "=", value: 3 }), 1 / 800);
  // A coarse histogram cannot be exact, but its error is bounded by one bucket's worth of rows.
  const coarse = new Catalog(40);
  coarse.addTable("skewed", 1000, [{
    name: "v", distinct: 100, histogram: [{ lo: 0, hi: 99, count: 100 }, { lo: 100, hi: 199, count: 900 }],
  }]);
  const coarseEstimate = coarse.selectivity("skewed", { column: "v", op: "<", value: 150 });
  assert.ok(Math.abs(coarseEstimate * 1000 - 550) <= 100 + 1e-9, "uniform-within-bucket interpolation");
  assert.strictEqual(coarse.selectivity("skewed", { column: "v", op: "<", value: 0 }), 0);
  assert.strictEqual(coarse.selectivity("skewed", { column: "v", op: ">=", value: 0 }), 1);
  assert.strictEqual(coarse.conjunctionSelectivity("skewed", [
    { column: "v", op: ">=", value: 0 }, { column: "v", op: ">=", value: 0 }, ]), 1);
  assert.throws(() => coarse.selectivity("skewed", { column: "v", op: "~=", value: 1 }), /unknown operator/);
  assert.throws(() => coarse.selectivity("skewed", { column: "nope", op: "=", value: 1 }), /unknown column/);
  assert.throws(() => coarse.selectivity("ghost", { column: "v", op: "=", value: 1 }), /unknown table/);
  // A three-relation chain and a four-relation star.
  const chain = {
    tables: ["small", "mid", "big"], filters: { small: [{ column: "id", op: "<=", value: 4 }] }, joins: [
      { left: { table: "small", column: "id" }, right: { table: "mid", column: "id" } },
      { left: { table: "mid", column: "id" }, right: { table: "big", column: "id" } }, ],
  };
  const star = {
    tables: ["big", "mid", "small", "other"], filters: { other: [{ column: "id", op: "=", value: 1 }] }, joins: [
      { left: { table: "big", column: "id" }, right: { table: "mid", column: "id" } },
      { left: { table: "big", column: "id" }, right: { table: "small", column: "id" } },
      { left: { table: "big", column: "id" }, right: { table: "other", column: "id" } }, ],
  };
  catalog.addTable("other", 90, [{ name: "id", distinct: 90 }]);
  // Cost model sanity: the hash join must beat the nested loop on a large inner side. `small` has 60
  // rows (two pages) of which the filter keeps 20, and `big` occupies 100 pages.
  const leaf = leafPlan(chain, "big");
  assert.strictEqual(planCost(leaf, catalog), 100);
  assert.strictEqual(catalog.pages(catalog.rows("small")), 2);
  assert.strictEqual(planRows(leafPlan(chain, "small"), catalog), 20);
  const hash = joinPlan("HASH", leafPlan(chain, "small"), leaf, chain.joins.slice(0, 1), 1 / 800);
  const loop = joinPlan("NL", leafPlan(chain, "small"), leaf, chain.joins.slice(0, 1), 1 / 800);
  assert.strictEqual(planCost(hash, catalog), 2 + 100 + 3 * (1 + 100));
  assert.strictEqual(planCost(loop, catalog), 2 + 20 * 100);
  assert.ok(planCost(hash, catalog) < planCost(loop, catalog), "hash join wins on a big inner side");
  // Exhaustive enumeration versus the Selinger dynamic program.
  const pair = {
    tables: ["small", "mid"], filters: { small: [{ column: "id", op: "<=", value: 4 }] },
    joins: [{ left: { table: "small", column: "id" }, right: { table: "mid", column: "id" } }],
  };
  const pairEx = exhaustivePlans(pair, catalog);
  const pairDp = selingerDp(pair, catalog);
  assert.strictEqual(pairEx.candidates, 6, "two ordered splits, three methods each");
  assert.strictEqual(pairDp.merges, 6, "nothing to prune with a single join");
  assert.strictEqual(pairDp.cost, pairEx.best.cost);
  const chainEx = exhaustivePlans(chain, catalog);
  const chainDp = selingerDp(chain, catalog);
  assert.ok(chainEx.candidates > chainDp.merges, "the DP prunes the three-relation chain");
  assert.strictEqual(chainDp.cost, chainEx.best.cost, "the DP finds the exhaustive optimum");
  assert.strictEqual(planCost(chainDp.plan, catalog), chainDp.cost, "the reported cost matches the tree");
  const starEx = exhaustivePlans(star, catalog);
  const starDp = selingerDp(star, catalog);
  assert.strictEqual(starDp.cost, starEx.best.cost, "the DP also matches on a four-relation star");
  assert.ok(starDp.merges < starEx.candidates, "the DP never materialises every complete plan");
  assert.strictEqual(planRows(starDp.plan, catalog), planRows(starEx.best.plan, catalog),
    "output rows depend only on the subset, never on the shape");
  // The best plan starts from the filtered relation: the star's only selective access path is `other`.
  assert.match(describePlan(starDp.plan), /Scan\(other \[id=1\]\)/);
  // Rejections: unknown methods, duplicated relations and disconnected graphs.
  assert.throws(() => joinPlan("MERGE", leaf, leaf, [], 1), /unknown join method/);
  assert.throws(() => exhaustivePlans({ tables: ["big", "big"], filters: {}, joins: [] }, catalog), /listed twice/);
  assert.throws(() => selingerDp({ tables: ["big", "other"], filters: {}, joins: [] }, catalog), /disconnected/);
  assert.throws(() => exhaustivePlans({ tables: ["big", "mid"], filters: {}, joins: [] }, catalog), /disconnected/);
  // Randomised catalogs: the DP must equal exhaustive enumeration on every generated query.
  let seed = 641;
  const rng = (bound) => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return (seed >>> 8) % bound;
  };
  let queries = 0;
  for (let trial = 0; trial < 30; trial++) {
    const local = new Catalog(40);
    const rows = [100 + rng(4000), 100 + rng(4000), 100 + rng(4000)];
    for (let i = 0; i < 3; i++) local.addTable(`t${i}`, rows[i], [{ name: "id", distinct: 2 + rng(rows[i]) }]);
    const candidates = [
      { left: { table: "t0", column: "id" }, right: { table: "t1", column: "id" } },
      { left: { table: "t1", column: "id" }, right: { table: "t2", column: "id" } },
      { left: { table: "t0", column: "id" }, right: { table: "t2", column: "id" } }, ];
    const joins = candidates.filter(() => rng(2) === 0);
    for (const required of [candidates[0], candidates[1]]) { // keep the chain connected, without duplicates
      const present = joins.some((j) => j.left.table === required.left.table && j.right.table === required.right.table);
      if (!present) joins.push(required);
    }
    const query = { tables: ["t0", "t1", "t2"], filters: { t0: [{ column: "id", op: "=", value: 1 }] }, joins };
    const exhaustive = exhaustivePlans(query, local);
    const dp = selingerDp(query, local);
    assert.strictEqual(dp.cost, exhaustive.best.cost, `trial ${trial}: DP must match enumeration`);
    assert.ok(dp.merges <= exhaustive.candidates);
    for (const plan of exhaustive.all) {
      assert.ok(planCost(plan, local) > 0 && planRows(plan, local) >= 1, "every enumerated plan is usable");
    }
    queries++;
  }
  console.log(`planner: ${queries} randomised queries, star DP ${starDp.merges} merges vs ${starEx.candidates} plans`);
  console.log("PASS");
}
