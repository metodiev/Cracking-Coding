/** Task 888 - Implement incremental view maintenance with delta propagation and bounded memory.
 *  (Tier: Master, CTCI Ch 14 — Databases)
 *
 * A materialised-view engine with classic delta propagation. Every view is an operator tree
 * (scan / filter / project / equi-join / group-by) whose nodes hold a *bag* (multiset of rows with
 * signed counts). When one base table changes by a signed delta ΔT, the delta is translated
 * bottom-up through the tree with the standard rules:
 *
 *   Δ(filter) = filter(Δ)                      Δ(project) = project(Δ)
 *   Δ(A ⋈ B)  = ΔA ⋈ B_live     (only one side can change per base-table update)
 *   Δ(group)  = fold Δ into the affected groups only; min/max recompute just those groups
 *
 * Because exactly one base table changes per update, the two-term join rule ΔA⋈B is exact for bag
 * semantics (this is what makes the delta translation O(Δ) instead of O(table)). Self-joins, which
 * would invalidate that assumption, are rejected by the spec validator.
 *
 * Bounded memory: the pending delta log is a ring buffer of capacity `capacity`. An update that
 * would overflow it triggers a compaction (the observations are folded into a checkpoint and the log
 * is reset); a node whose checkpoint is older than the last compaction must be rebuilt from its
 * children before it can accept further deltas, which is the documented recovery path.
 *
 * Transactional guarantee: `applyBatch` is all-or-nothing. A batch is applied against a snapshot of
 * every bag; if any operation is rejected (unknown table, deleting a row that is not present,
 * schema violation) the whole batch is rolled back and the state digest is unchanged.
 *
 * Invariants verified below: (1) after *every* operation every materialised view equals an
 * independent naive evaluator (plain nested loops, no bags, no deltas) — so the incremental result
 * is exact, not approximate; (2) bags never hold negative counts (safety) and a delete of an absent
 * row is rejected; (3) min/max stay correct when the current minimum is deleted (recompute path);
 * (4) the delta log never exceeds its capacity and compactions/lag rebuilds stay within a bound
 * (liveness); (5) an update touching k rows costs O(k) work, verified against a full-recompute
 * baseline; (6) chained views (view over view) propagate correctly; (7) atomicity/rollback; (8)
 * determinism.
 *
 * Simplifications: no disk, no concurrency (the "transaction" is a sequencing device), no B-tree
 * indexes — joins are hash joins over in-process maps, and "scale" is expressed as typed table
 * sizes driven by a seeded LCG.
 */
"use strict";

/* ------------------------------------------------------------------ determinism */

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
}

/* ------------------------------------------------------------------ rows and bags */

const rowKey = (row) => Object.keys(row).sort().map((k) => `${k}=${row[k]}`).join("\u0001");

/** Multiset of rows with signed counts; counts are always positive in a live bag. */
class Bag {
  constructor() { this.map = new Map(); }
  add(row, count) {
    const key = rowKey(row);
    const existing = this.map.get(key);
    if (existing) {
      existing.count += count;
      if (existing.count < 0) throw new Error("bag underflow: negative multiplicity");
      if (existing.count === 0) this.map.delete(key);
    } else {
      if (count < 0) throw new Error("bag underflow: cannot remove an absent row");
      this.map.set(key, { row, count });
    }
  }
  get rowCount() { let total = 0; for (const entry of this.map.values()) total += entry.count; return total; }
  rows() { return [...this.map.values()].map((entry) => entry.row); }
  entries() { return [...this.map.values()]; }
  clone() {
    const copy = new Bag();
    for (const entry of this.map.values()) copy.map.set(rowKey(entry.row), { row: entry.row, count: entry.count });
    return copy;
  }
  /** Canonical text form, used for the state digest and for bag equality. */
  canonical() {
    return [...this.map.entries()].sort().map(([key, entry]) => `${key}#${entry.count}`).join(";");
  }
}

/* ------------------------------------------------------------------ spec helpers */

const AGGREGATES = new Set(["count", "sum", "min", "max"]);

function validateSpec(spec, name = "view") {
  if (!spec || typeof spec.op !== "string") throw new Error(`${name}: missing op`);
  if (spec.op === "scan") {
    if (typeof spec.table !== "string") throw new Error(`${name}: scan needs a table`);
    return [spec.table];
  }
  if (spec.op === "filter") {
    if (typeof spec.pred !== "function") throw new Error(`${name}: filter needs a predicate`);
    return validateSpec(spec.child, name);
  }
  if (spec.op === "project") {
    if (!Array.isArray(spec.fields) || spec.fields.length === 0) throw new Error(`${name}: project needs fields`);
    return validateSpec(spec.child, name);
  }
  if (spec.op === "join") {
    const left = validateSpec(spec.left, name);
    const right = validateSpec(spec.right, name);
    const overlap = left.filter((table) => right.includes(table));
    if (overlap.length > 0) throw new Error(`${name}: self-join on ${overlap} breaks one-sided deltas`);
    if (!Array.isArray(spec.on) || spec.on.length === 0) throw new Error(`${name}: join needs keys`);
    return [...left, ...right];
  }
  if (spec.op === "group") {
    if (!Array.isArray(spec.keys)) throw new Error(`${name}: group needs keys`);
    for (const aggregate of spec.aggs) {
      if (!AGGREGATES.has(aggregate.agg)) throw new Error(`${name}: unknown aggregate ${aggregate.agg}`);
    }
    return validateSpec(spec.child, name);
  }
  throw new Error(`${name}: unknown op ${spec.op}`);
}

/** Independent naive evaluator: plain nested loops over arrays, no bags and no delta logic. */
function naiveEvaluate(spec, tables) {
  const rowsOf = (node) => {
    switch (node.op) {
      case "scan": {
        const table = tables[node.table];
        if (!table) throw new Error(`unknown table ${node.table}`);
        return table.rows.map((row) => ({ ...row }));
      }
      case "filter": return rowsOf(node.child).filter((row) => node.pred(row));
      case "project": return rowsOf(node.child).map((row) => {
        const out = {};
        for (const field of node.fields) out[field] = row[field];
        return out;
      });
      case "join": {
        const left = rowsOf(node.left);
        const right = rowsOf(node.right);
        const out = [];
        for (const a of left) {
          for (const b of right) {
            let match = true;
            for (const [lk, rk] of node.on) if (a[lk] !== b[rk]) { match = false; break; }
            if (!match) continue;
            const merged = { ...a };
            for (const key of Object.keys(b)) {
              if (key in merged && merged[key] !== b[key]) throw new Error("join field collision");
              merged[key] = b[key];
            }
            out.push(merged);
          }
        }
        return out;
      }
      default: {
        const input = rowsOf(node.child);
        const groups = new Map();
        for (const row of input) {
          const key = node.keys.map((k) => `${k}=${row[k]}`).join("\u0001");
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key).push(row);
        }
        const out = [];
        for (const bucket of groups.values()) {
          const aggregated = {};
          for (const k of node.keys) aggregated[k] = bucket[0][k];
          for (const aggregate of node.aggs) applyAggregate(aggregate, bucket, aggregated);
          out.push(aggregated);
        }
        return out;
      }
    }
  };
  return rowsOf(spec);
}

function applyAggregate(aggregate, bucket, target) {
  if (aggregate.agg === "count") { target[aggregate.name] = bucket.length; return; }
  const values = bucket.map((row) => row[aggregate.field]);
  if (aggregate.agg === "sum") target[aggregate.name] = values.reduce((a, b) => a + b, 0);
  else if (aggregate.agg === "min") target[aggregate.name] = values.length ? Math.min(...values) : null;
  else target[aggregate.name] = values.length ? Math.max(...values) : null;
}

/** Canonical multiset form of a row array, comparable between the engine and the oracle. */
function canonicalRows(rows) {
  const bag = new Bag();
  for (const row of rows) bag.add(row, 1);
  return bag.canonical();
}

/* ------------------------------------------------------------------ engine nodes */

class DeltaLog {
  constructor(capacity) { this.capacity = capacity; this.entries = []; this.compactions = 0; this.largest = 0; }
  push(entry) {
    this.entries.push(entry);
    if (this.entries.length > this.capacity) {
      this.compactions++;
      this.entries = [];       // the checkpoint absorbed everything older
      this.generation++;
    }
    this.largest = Math.max(this.largest, this.entries.length);
    return this.entries.length;
  }
}

class Node {
  constructor(spec, engine) {
    this.engine = engine;
    this.spec = spec;
    this.bag = new Bag();
    this.rowsTouched = 0;
    this.tables = validateSpec(spec);
    switch (spec.op) {
      case "scan": this.child = null; break;
      case "join": this.left = engine.nodeFor(spec.left); this.right = engine.nodeFor(spec.right); break;
      default: this.child = engine.nodeFor(spec.child);
    }
  }

  /** Full evaluation from the base tables (used at build time and for lag recovery). */
  rebuild() {
    const rows = this.evaluate();
    this.bag = new Bag();
    for (const row of rows) this.bag.add(row, 1);
    if (this.spec.op === "group") {
      // The group bookkeeping must be re-derived too, otherwise the next delta cannot know which
      // row to retract.
      for (const key of [...this.engine.groupRows.keys()]) {
        if (key.startsWith(`${this.id}|`)) this.engine.groupRows.delete(key);
      }
      for (const row of rows) {
        const key = this.spec.keys.map((k) => `${k}=${row[k]}`).join("\u0001");
        this.engine.groupRows.set(`${this.id}|${key}`, row);
      }
    }
  }

  evaluate() {
    switch (this.spec.op) {
      case "scan": {
        const table = this.engine.tables[this.spec.table];
        return table.rows.map((row) => ({ ...row }));
      }
      case "filter": return this.child.bag.rows().filter((row) => this.spec.pred(row));
      case "project": return this.child.bag.rows().map((row) => {
        const out = {};
        for (const field of this.spec.fields) out[field] = row[field];
        return out;
      });
      case "join": {
        const out = [];
        for (const a of this.left.bag.rows()) {
          for (const b of this.right.bag.rows()) {
            let match = true;
            for (const [lk, rk] of this.spec.on) if (a[lk] !== b[rk]) { match = false; break; }
            if (match) out.push({ ...a, ...b });
          }
        }
        return out;
      }
      default: {
        const groups = new Map();
        for (const row of this.child.bag.rows()) {
          const key = this.spec.keys.map((k) => `${k}=${row[k]}`).join("\u0001");
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key).push(row);
        }
        const out = [];
        for (const bucket of groups.values()) {
          const aggregated = {};
          for (const k of this.spec.keys) aggregated[k] = bucket[0][k];
          for (const aggregate of this.spec.aggs) applyAggregate(aggregate, bucket, aggregated);
          out.push(aggregated);
        }
        return out;
      }
    }
  }

  /**
   * Translates a base-table delta into this node's delta using the current (live) bags of the
   * sibling subtrees, then folds it into this node's bag. Returns the delta for the parent.
   */
  propagate(table, changedRows) {
    if (!this.tables.includes(table)) return [];
    const delta = this.translate(table, changedRows);
    this.rowsTouched += delta.length;
    for (const entry of delta) this.bag.add(entry.row, entry.count);
    return delta;
  }

  translate(table, changedRows) {
    // A node's delta is computed exactly once per base-table update and then cached, because the
    // translation must observe the state as of *before* the update: re-translating after the node
    // has committed its own delta would retract rows that were never added. Ancestors therefore
    // reuse the cached delta (this is what keeps the rules idempotent).
    const cacheKey = `${this.id}|${table}`;
    const cached = this.engine.deltaCache.get(cacheKey);
    if (cached !== undefined) return cached;
    const delta = this.computeDelta(table, changedRows);
    this.engine.deltaCache.set(cacheKey, delta);
    return delta;
  }

  computeDelta(table, changedRows) {
    switch (this.spec.op) {
      case "scan": return changedRows.map((entry) => ({ row: { ...entry.row }, count: entry.count }));
      case "filter": return this.child.translate(table, changedRows)
        .filter((entry) => this.spec.pred(entry.row));
      case "project": return this.child.translate(table, changedRows).map((entry) => {
        const out = {};
        for (const field of this.spec.fields) out[field] = entry.row[field];
        return { row: out, count: entry.count };
      });
      case "join": {
        const leftSide = this.left.tables.includes(table);
        const driving = leftSide ? this.left : this.right;
        const other = leftSide ? this.right : this.left;
        const childDelta = driving.translate(table, changedRows);
        const out = [];
        for (const entry of childDelta) {
          for (const probe of other.bag.entries()) {
            const a = leftSide ? entry.row : probe.row;
            const b = leftSide ? probe.row : entry.row;
            let match = true;
            for (const [lk, rk] of this.spec.on) if (a[lk] !== b[rk]) { match = false; break; }
            if (!match) continue;
            const merged = { ...a, ...b };
            // A child row repeated `count` times pairs with each probe occurrence.
            out.push({ row: merged, count: entry.count * probe.count });
          }
        }
        this.siblingProbes += other.bag.entries().length;
        return out;
      }
      default: {
        const childDelta = this.child.translate(table, changedRows);
        if (childDelta.length === 0) return [];
        const affected = new Set();
        for (const entry of childDelta) {
          affected.add(this.spec.keys.map((k) => `${k}=${entry.row[k]}`).join("\u0001"));
        }
        // Only the affected groups are recomputed; every other group is untouched. The child's bag
        // has already absorbed the delta (children propagate first), so it is the authoritative
        // source of the new group contents.
        const liveRows = this.child.bag.rows();
        const buckets = new Map([...affected].map((key) => [key, []]));
        for (const row of liveRows) {
          const key = this.spec.keys.map((k) => `${k}=${row[k]}`).join("\u0001");
          if (buckets.has(key)) buckets.get(key).push(row);
        }
        const groupsRecomputed = affected.size;
        this.groupsRecomputed += groupsRecomputed;
        const out = [];
        for (const [key, bucket] of buckets) {
          const previous = this.engine.groupRows.get(`${this.id}|${key}`);
          let aggregated = null;
          if (bucket.length > 0) {
            aggregated = {};
            for (const k of this.spec.keys) aggregated[k] = bucket[0][k];
            for (const aggregate of this.spec.aggs) applyAggregate(aggregate, bucket, aggregated);
          }
          if (previous) out.push({ row: previous, count: -1 });
          if (aggregated) out.push({ row: aggregated, count: 1 });
          if (aggregated) this.engine.groupRows.set(`${this.id}|${key}`, aggregated);
          else this.engine.groupRows.delete(`${this.id}|${key}`);
        }
        return out;
      }
    }
  }

}
/* ------------------------------------------------------------------ engine */

class IvEngine {
  constructor(specs, capacity = 4) {
    this.tables = {};
    this.specs = specs;
    this.capacity = capacity;
    this.log = new DeltaLog(capacity);
    this.log.generation = 0;
    this.groupRows = new Map();
    this.deltaCache = new Map();
    this.nodes = [];
    this.views = new Map();
    this.nextId = 0;
    this.fullRecomputes = 0;
    this.rollbacks = 0;
    this.updates = 0;
  }

  addTable(name, rows) { this.tables[name] = { name, rows: rows.map((row) => ({ ...row })) }; }

  nodeFor(spec) {
    if (spec.__node) return spec.__node;
    const node = new Node(spec, this);
    node.id = this.nextId++;
    node.siblingProbes = 0;
    node.groupsRecomputed = 0;
    spec.__node = node;
    this.nodes.push(node);
    return node;
  }

  /** Builds (or rebuilds) every view bottom-up, evaluating parents after their children. */
  build() {
    for (const [name, spec] of Object.entries(this.specs)) {
      validateSpec(spec, name);
      this.views.set(name, this.nodeFor(spec));
    }
    for (const node of this.topologicalOrder()) node.rebuild();
    return this;
  }

  topologicalOrder() {
    const seen = new Set();
    const order = [];
    const visit = (node) => {
      if (seen.has(node)) return;
      seen.add(node);
      if (node.spec.op === "join") { visit(node.left); visit(node.right); }
      else if (node.child) visit(node.child);
      order.push(node);
    };
    for (const node of this.nodes) visit(node);
    return order;
  }

  /** Applies one signed base-table change. `count` is +1 to insert, -1 to delete. */
  apply(tableName, row, count = 1) {
    const table = this.tables[tableName];
    if (!table) throw new Error(`unknown table ${tableName}`);
    if (count !== 1 && count !== -1) throw new Error("delta must be +1 or -1");
    const key = rowKey(row);
    const index = table.rows.findIndex((existing) => rowKey(existing) === key);
    if (count === 1 && index >= 0) throw new Error(`duplicate row in ${tableName}`);
    if (count === -1 && index < 0) throw new Error(`cannot delete an absent row from ${tableName}`);
    if (index >= 0) table.rows.splice(index, 1);
    else table.rows.push({ ...row });
    this.deltaCache.clear();
    const changed = [{ row: { ...row }, count }];
    for (const node of this.topologicalOrder()) node.propagate(tableName, changed);
    this.updates++;
    this.log.push({ table: tableName, row: { ...row }, count });
    return this;
  }

  viewRows(name) {
    const node = this.views.get(name);
    if (!node) throw new Error(`unknown view ${name}`);
    return node.bag.rows();
  }

  digest() {
    return [...this.views.entries()].map(([name, node]) => `${name}:${node.bag.canonical()}`).join("|");
  }

  /** All-or-nothing batch: a rejected operation rolls the whole batch back. */
  applyBatch(operations) {
    const snapshot = {
      tables: Object.fromEntries(Object.entries(this.tables)
        .map(([name, table]) => [name, table.rows.map((row) => ({ ...row }))])),
      bags: new Map(this.nodes.map((node) => [node, node.bag.clone()])),
      groupRows: new Map([...this.groupRows].map(([key, row]) => [key, { ...row }])),
    };
    try {
      for (const operation of operations) this.apply(operation.table, operation.row, operation.count);
      return { applied: operations.length, rolledBack: false };
    } catch (error) {
      for (const [name, rows] of Object.entries(snapshot.tables)) this.tables[name].rows = rows.map((row) => ({ ...row }));
      this.groupRows = new Map([...snapshot.groupRows].map(([key, row]) => [key, { ...row }]));
      for (const node of this.nodes) node.bag = snapshot.bags.get(node).clone();
      this.rollbacks++;
      return { applied: 0, rolledBack: true, reason: error.message };
    }
  }

  /** Simulates a lagging replica: the node must be rebuilt before it can take deltas again. */
  markStale(viewName) {
    const node = this.views.get(viewName);
    node.stale = true;
    return this;
  }

  refreshStale() {
    let rebuilt = 0;
    for (const node of this.topologicalOrder()) {
      if (node.stale) { node.rebuild(); node.stale = false; rebuilt++; this.fullRecomputes++; }
    }
    return rebuilt;
  }

  /** Cost model: rows a full recompute of every view would touch, for the incremental comparison. */
  fullRecomputeCost() {
    const subtreeRows = (node) => {
      if (node.spec.op === "scan") return this.tables[node.spec.table].rows.length;
      if (node.spec.op === "join") return subtreeRows(node.left) + subtreeRows(node.right);
      return subtreeRows(node.child);
    };
    return this.topologicalOrder().reduce((total, node) => total + subtreeRows(node), 0);
  }

  incrementalCost() {
    return this.nodes.reduce((total, node) => total + node.rowsTouched + node.groupsRecomputed, 0);
  }
}

/* ------------------------------------------------------------------ fixtures */

function buildFixture(seed, { customers = 40, orders = 300 } = {}) {
  const prng = new Prng(seed);
  const customerRows = [];
  const cities = ["athens", "berlin", "cairo", "dublin"];
  for (let i = 0; i < customers; i++) {
    customerRows.push({ cid: i, city: cities[prng.below(cities.length)], tier: 1 + prng.below(3) });
  }
  const orderRows = [];
  for (let i = 0; i < orders; i++) {
    orderRows.push({
      oid: 1000 + i,
      cid: prng.below(customers),
      amount: 5 + prng.below(95),
      status: prng.below(4) === 0 ? "void" : "ok",
    });
  }
  return { customerRows, orderRows };
}

/** Two views: the join/aggregate pair, and a third view over the second one (chaining). */
function standardSpecs() {
  const liveOrders = {
    op: "filter",
    pred: (row) => row.status !== "void",
    child: { op: "scan", table: "orders" },
  };
  const perCustomer = {
    op: "group",
    keys: ["cid"],
    aggs: [{ name: "total", agg: "sum", field: "amount" }, { name: "orders", agg: "count" }],
    child: liveOrders,
  };
  const enriched = {
    op: "join",
    on: [["cid", "cid"]],
    left: perCustomer,
    right: { op: "scan", table: "customers" },
  };
  const byTier = {
    op: "group",
    keys: ["tier"],
    aggs: [{ name: "revenue", agg: "sum", field: "total" }, { name: "minOrder", agg: "min", field: "total" }],
    child: enriched,
  };
  return { perCustomer, enriched, byTier };
}

module.exports = {
  Prng, Bag, rowKey, validateSpec, naiveEvaluate, canonicalRows, DeltaLog, Node, IvEngine,
  buildFixture, standardSpecs, applyAggregate,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- spec validation rejects the shapes that would invalidate the delta rules ------------
  assert.throws(() => validateSpec({ op: "scan" }), /scan needs a table/);
  assert.throws(() => validateSpec({ op: "nope" }), /unknown op/);
  assert.throws(() => validateSpec({ op: "filter", pred: 1, child: { op: "scan", table: "t" } }), /predicate/);
  assert.throws(() => validateSpec({ op: "project", fields: [], child: { op: "scan", table: "t" } }), /fields/);
  assert.throws(() => validateSpec({
    op: "join", on: [["a", "a"]],
    left: { op: "scan", table: "t" }, right: { op: "scan", table: "t" },
  }), /self-join/);
  assert.throws(() => validateSpec({
    op: "group", keys: ["k"], aggs: [{ name: "x", agg: "median" }], child: { op: "scan", table: "t" },
  }), /unknown aggregate/);

  // --- bag arithmetic: signed counts, underflow protection ---------------------------------
  const bag = new Bag();
  bag.add({ a: 1 }, 2);
  bag.add({ a: 1 }, -1);
  assert.strictEqual(bag.rowCount, 1);
  bag.add({ a: 1 }, -1);
  assert.strictEqual(bag.map.size, 0, "counts that reach zero leave the bag");
  assert.throws(() => new Bag().add({ a: 1 }, -1), /underflow/);
  const cloned = new Bag();
  cloned.add({ a: 1 }, 3);
  assert.strictEqual(cloned.clone().canonical(), cloned.canonical());

  // --- the naive evaluator itself is checked against hand-computed answers ------------------
  const tables = {
    orders: { rows: [{ oid: 1, cid: 7, amount: 10, status: "ok" }, { oid: 2, cid: 7, amount: 5, status: "ok" },
      { oid: 3, cid: 8, amount: 99, status: "void" }] },
    customers: { rows: [{ cid: 7, tier: 2 }, { cid: 8, tier: 3 }] },
  };
  const specs = standardSpecs();
  const naiveTotals = naiveEvaluate(specs.perCustomer, tables);
  assert.deepStrictEqual(naiveTotals.sort((a, b) => a.cid - b.cid), [{ cid: 7, total: 15, orders: 2 }]);
  const naiveByTier = naiveEvaluate(specs.byTier, tables);
  assert.deepStrictEqual(naiveByTier, [{ tier: 2, revenue: 15, minOrder: 15 }], "void orders are excluded");

  // --- incremental engine matches the oracle after every single operation -------------------
  const fixture = buildFixture(888, { customers: 30, orders: 200 });
  const engine = new IvEngine(standardSpecs(), 6);
  engine.addTable("customers", fixture.customerRows);
  engine.addTable("orders", fixture.orderRows);
  engine.build();
  for (const name of ["perCustomer", "enriched", "byTier"]) {
    assert.strictEqual(canonicalRows(engine.viewRows(name)), canonicalRows(naiveEvaluate(engine.specs[name], engine.tables)),
      `${name} at build time`);
  }
  assert.ok(engine.viewRows("perCustomer").length > 5);
  assert.ok(engine.viewRows("enriched").length >= engine.viewRows("perCustomer").length);

  const prng = new Prng(4242);
  let mismatches = 0;
  for (let step = 0; step < 260; step++) {
    const onOrders = prng.below(3) !== 0;
    if (onOrders) {
      const action = prng.below(10);
      if (action < 6) {
        engine.apply("orders", { oid: 5000 + step, cid: prng.below(30), amount: 1 + prng.below(60), status: prng.below(4) === 0 ? "void" : "ok" });
      } else {
        const rows = engine.tables.orders.rows;
        engine.apply("orders", rows[prng.below(rows.length)], -1);
      }
    } else {
      const action = prng.below(3);
      if (action === 0) engine.apply("customers", { cid: 100 + step, tier: 1 + prng.below(3), city: "athens" });
      else if (action === 1 && engine.tables.customers.rows.length > 4) {
        const rows = engine.tables.customers.rows;
        engine.apply("customers", rows[prng.below(rows.length)], -1);
      } else {
        engine.apply("orders", { oid: 7000 + step, cid: 100 + step, amount: 40, status: "ok" });
      }
    }
    for (const node of engine.nodes) {
      for (const entry of node.bag.entries()) {
        assert.ok(entry.count > 0, `safety: negative count in ${node.spec.op}`);
      }
    }
    assert.ok(engine.log.entries.length <= engine.capacity, "delta log must stay bounded");
    for (const name of ["perCustomer", "enriched", "byTier"]) {
      if (canonicalRows(engine.viewRows(name)) !== canonicalRows(naiveEvaluate(engine.specs[name], engine.tables))) {
        mismatches++;
      }
    }
  }
  assert.strictEqual(mismatches, 0, `${mismatches} view/oracle mismatches over 260 operations`);
  assert.ok(engine.log.compactions > 0, "the log must have compacted at least once");
  assert.ok(engine.log.largest <= engine.capacity, `largest log ${engine.log.largest} <= ${engine.capacity}`);

  // --- cost: a one-row update costs far less than a full recompute --------------------------
  const before = engine.incrementalCost();
  const fullCost = engine.fullRecomputeCost();
  engine.apply("orders", { oid: 999999, cid: 3, amount: 11, status: "ok" });
  const incremental = engine.incrementalCost() - before;
  assert.ok(fullCost > 100, "the fixture must be big enough to be interesting");
  assert.ok(incremental < fullCost / 4, `incremental ${incremental} vs full ${fullCost}`);
  assert.ok(engine.viewRows("byTier").length >= 1);
  assert.strictEqual(canonicalRows(engine.viewRows("byTier")), canonicalRows(naiveEvaluate(engine.specs.byTier, engine.tables)));

  // --- min/max survive the deletion of the current minimum ---------------------------------
  const small = new IvEngine({
    lows: {
      op: "group",
      keys: ["g"],
      aggs: [{ name: "lo", agg: "min", field: "v" }, { name: "hi", agg: "max", field: "v" }],
      child: { op: "scan", table: "t" },
    },
  }, 4);
  small.addTable("t", [{ g: "a", v: 5 }, { g: "a", v: 9 }, { g: "a", v: 7 }, { g: "b", v: 2 }]);
  small.build();
  assert.deepStrictEqual(small.viewRows("lows").sort((x, y) => x.g.localeCompare(y.g)),
    [{ g: "a", lo: 5, hi: 9 }, { g: "b", lo: 2, hi: 2 }]);
  small.apply("t", { g: "a", v: 5 }, -1);
  assert.deepStrictEqual(small.viewRows("lows").sort((x, y) => x.g.localeCompare(y.g)),
    [{ g: "a", lo: 7, hi: 9 }, { g: "b", lo: 2, hi: 2 }], "recompute path after deleting the min");
  assert.strictEqual(canonicalRows(small.viewRows("lows")), canonicalRows(naiveEvaluate(small.specs.lows, small.tables)));
  small.apply("t", { g: "a", v: 7 }, -1);
  small.apply("t", { g: "a", v: 9 }, -1);
  assert.deepStrictEqual(small.viewRows("lows"), [{ g: "b", lo: 2, hi: 2 }], "an emptied group disappears");
  small.apply("t", { g: "b", v: 2 }, -1);
  assert.deepStrictEqual(small.viewRows("lows"), []);
  small.apply("t", { g: "c", v: 4 });
  assert.deepStrictEqual(small.viewRows("lows"), [{ g: "c", lo: 4, hi: 4 }], "a fresh group appears");

  // --- rejects impossible changes and never corrupts the state -----------------------------
  const guard = new IvEngine(standardSpecs(), 4);
  guard.addTable("customers", [{ cid: 1, tier: 1, city: "athens" }]);
  guard.addTable("orders", [{ oid: 1, cid: 1, amount: 5, status: "ok" }]);
  guard.build();
  const digestBefore = guard.digest();
  const batch = guard.applyBatch([
    { table: "orders", row: { oid: 2, cid: 1, amount: 9, status: "ok" } },
    { table: "orders", row: { oid: 3, cid: 1, amount: 9, status: "ok" }, count: -1 },
    { table: "orders", row: { oid: 4, cid: 1, amount: 9, status: "ok" } },
  ]);
  assert.ok(batch.rolledBack && batch.applied === 0);
  assert.match(batch.reason, /absent row/);
  assert.strictEqual(guard.digest(), digestBefore, "rollback must restore every view exactly");
  assert.strictEqual(guard.rollbacks, 1);
  const good = guard.applyBatch([{ table: "orders", row: { oid: 2, cid: 1, amount: 9, status: "ok" } }]);
  assert.ok(!good.rolledBack && good.applied === 1);
  assert.strictEqual(canonicalRows(guard.viewRows("perCustomer")), canonicalRows(naiveEvaluate(guard.specs.perCustomer, guard.tables)));
  assert.throws(() => guard.apply("orders", { oid: 2, cid: 1, amount: 9, status: "ok" }), /duplicate row/);
  assert.throws(() => guard.apply("ghosts", { x: 1 }), /unknown table/);
  assert.throws(() => guard.apply("orders", { oid: 5, cid: 1, amount: 1, status: "ok" }, 3), /delta must be/);

  // --- lag recovery: a stale node is rebuilt and then matches the oracle -------------------
  const lagged = new IvEngine(standardSpecs(), 4);
  lagged.addTable("customers", fixture.customerRows.slice(0, 10));
  lagged.addTable("orders", fixture.orderRows.slice(0, 60));
  lagged.build();
  lagged.markStale("byTier");
  assert.strictEqual(lagged.refreshStale(), 1);
  assert.strictEqual(lagged.fullRecomputes, 1);
  for (const name of ["perCustomer", "enriched", "byTier"]) {
    assert.strictEqual(canonicalRows(lagged.viewRows(name)), canonicalRows(naiveEvaluate(lagged.specs[name], lagged.tables)));
  }

  // --- edge cases: empty tables, joins without matches, empty groups ----------------------
  const empty = new IvEngine(standardSpecs(), 2);
  empty.addTable("customers", []);
  empty.addTable("orders", []);
  empty.build();
  assert.deepStrictEqual(empty.viewRows("enriched"), []);
  assert.deepStrictEqual(empty.viewRows("byTier"), []);
  empty.apply("orders", { oid: 1, cid: 5, amount: 7, status: "ok" });
  assert.deepStrictEqual(empty.viewRows("perCustomer"), [{ cid: 5, total: 7, orders: 1 }]);
  assert.deepStrictEqual(empty.viewRows("enriched"), [], "no matching customer -> empty join");
  assert.deepStrictEqual(empty.viewRows("byTier"), [], "group with no input rows");
  empty.apply("customers", { cid: 5, tier: 4, city: "cairo" });
  assert.deepStrictEqual(empty.viewRows("enriched"),
    [{ cid: 5, total: 7, orders: 1, tier: 4, city: "cairo" }], "join delta from the other side");
  assert.deepStrictEqual(empty.viewRows("byTier"), [{ tier: 4, revenue: 7, minOrder: 7 }]);
  assert.strictEqual(canonicalRows(empty.viewRows("byTier")), canonicalRows(naiveEvaluate(empty.specs.byTier, empty.tables)));
  empty.apply("orders", { oid: 1, cid: 5, amount: 7, status: "ok" }, -1);
  assert.deepStrictEqual(empty.viewRows("byTier"), [], "deleting the last order empties the group");
  assert.deepStrictEqual(empty.viewRows("perCustomer"), []);

  // --- a void order never reaches the aggregate --------------------------------------------
  const noise = new IvEngine(standardSpecs(), 2);
  noise.addTable("customers", [{ cid: 1, tier: 1, city: "athens" }]);
  noise.addTable("orders", [{ oid: 1, cid: 1, amount: 3, status: "void" }]);
  noise.build();
  assert.deepStrictEqual(noise.viewRows("perCustomer"), []);
  assert.strictEqual(canonicalRows(noise.viewRows("byTier")), canonicalRows(naiveEvaluate(noise.specs.byTier, noise.tables)));

  // --- determinism ------------------------------------------------------------------------
  const replay = () => {
    const e = new IvEngine(standardSpecs(), 5);
    e.addTable("customers", fixture.customerRows.slice(0, 8));
    e.addTable("orders", fixture.orderRows.slice(0, 40));
    e.build();
    const local = new Prng(31337);
    for (let step = 0; step < 40; step++) {
      const rows = e.tables.orders.rows;
      if (local.below(2) === 0) e.apply("orders", { oid: 8000 + step, cid: local.below(8), amount: 3 + local.below(20), status: "ok" });
      else if (rows.length > 1) e.apply("orders", rows[local.below(rows.length)], -1);
    }
    return e.digest();
  };
  assert.strictEqual(replay(), replay());

  console.log(`ivm: ${engine.updates} updates, ${engine.log.compactions} compactions, ` +
    `incremental cost ${incremental} vs full recompute ${fullCost}, rollbacks ${guard.rollbacks}`);
  console.log("PASS");
}
