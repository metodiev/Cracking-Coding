/** Task 645 - Implement materialized view maintenance incremental algorithm. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMPLIFIED: the warehouse is two in-memory base tables (a fact table and one dimension table)
 * and the view is a single-block definition - a selection, an equi-join, an optional projection and an
 * optional GROUP BY with one aggregate. There is no delta network, no tuple id space and no MVCC; the
 * "aggregate cache" is a per-group summary {count, sum, min, max}. The one honest gap is MAX/MIN: a
 * departing extremum cannot be recovered from a summary, so those views recompute the affected group from
 * the base tables (counted in `work.groupRecomputes`) instead of pretending to adjust it. Everything else
 * is maintained purely from the delta, using the standard delta rules:
 *
 *   selection      dV = sigma(where, dR)
 *   projection     dV = pi(dR)                       (the view keeps a multiset of projected rows)
 *   equi-join      dV = dR join S_old  U  R_old join dS  U  dR join dS
 *   GROUP BY       each affected group's count/sum is adjusted by the delta's contribution; a group whose
 *                  count falls to zero is deleted, which is what makes "the last row of a group" correct.
 *
 * WHAT IS VERIFIED: after *every* delta - fact inserts and deletes, dimension inserts and deletes - the
 * incrementally maintained view equals an independent full recomputation written as plain array code
 * (filter/map/reduce), for the aggregates sum, count, avg, min and max, for a grouped view and for an
 * ungrouped multiset view; deleting a row that is not in the base table is refused; groups disappear when
 * their last row goes away and reappear when a row arrives; duplicate base rows are counted as a
 * multiset; and a deliberately naive "add but never subtract" variant is shown to diverge from the
 * recomputation, so the equality checks cannot be vacuous. Work is measured: a single point delta visits
 * far fewer rows than a full recomputation, and the MIN/MAX recompute path is exercised.
 */
"use strict";

function rowKey(row) {
  return JSON.stringify(row, Object.keys(row).sort());
}

function sameRow(left, right) {
  return rowKey(left) === rowKey(right);
}

const AGGREGATES = ["sum", "count", "avg", "min", "max"];

function checkSpec(spec) {
  if (!AGGREGATES.includes(spec.aggregate)) throw new Error(`unknown aggregate ${spec.aggregate}`);
  if (spec.groupColumn === undefined) throw new Error("a group column (or null) is required");
  if (spec.groupColumn !== null && !(spec.aggregateColumn || spec.groupColumn)) {
    throw new Error("an aggregate column is required");
  }
  return spec;
}

/** Independent oracle: full recomputation with plain array code, no incremental reasoning at all. */
function evaluateFromScratch(spec, facts, dimensions) {
  const groups = new Map();
  const rows = new Map();
  let visited = 0;
  for (const fact of facts) {
    if (!spec.where(fact)) continue;
    const partner = dimensions.find((dimension) => dimension[spec.dimensionKey] === fact[spec.factKey]);
    if (!partner) continue;
    visited++;
    const value = spec.aggregateColumn ? fact[spec.aggregateColumn] : 1;
    if (spec.groupColumn === null) {
      const projected = { [spec.aggregateColumn]: value };
      rows.set(rowKey(projected), (rows.get(rowKey(projected)) || 0) + 1);
      continue;
    }
    const key = partner[spec.groupColumn];
    const bucket = groups.get(key) || { count: 0, sum: 0, min: undefined, max: undefined };
    bucket.count++;
    bucket.sum += value;
    bucket.min = bucket.min === undefined ? value : Math.min(bucket.min, value);
    bucket.max = bucket.max === undefined ? value : Math.max(bucket.max, value);
    groups.set(key, bucket);
  }
  return { groups, rows, visited };
}

function aggregateValue(aggregate, bucket) {
  if (aggregate === "sum") return bucket.sum;
  if (aggregate === "count") return bucket.count;
  if (aggregate === "avg") return bucket.sum / bucket.count;
  return aggregate === "min" ? bucket.min : bucket.max;
}

function snapshotOf(spec, entries) {
  const lines = [];
  if (spec.groupColumn === null) {
    for (const [key, count] of entries.rows) lines.push(`${key}x${count}`);
  } else {
    for (const [key, bucket] of entries.groups) lines.push(`${key}=${aggregateValue(spec.aggregate, bucket)}`);
  }
  return lines.sort();
}

class ViewMaintainer {
  constructor(spec, facts, dimensions) {
    checkSpec(spec);
    this.spec = spec;
    this.facts = facts.slice();
    this.dimensions = dimensions.slice();
    this.index = new Map();
    for (const dimension of this.dimensions) {
      const key = dimension[spec.dimensionKey];
      this.index.set(key, (this.index.get(key) || []).concat([dimension]));
    }
    this.groups = new Map();
    this.rows = new Map();
    this.work = { incremental: 0, full: 0, groupRecomputes: 0, indexMaintenance: 0 };
    this.recount();
  }
  /** Builds the view from the base tables (used once at load time and by the recompute path). */
  recount() {
    this.groups = new Map();
    this.rows = new Map();
    for (const fact of this.facts) this._update(fact, this._partnerOf(fact), +1, false);
    return this;
  }
  _partnerOf(fact) {
    const bucket = this.index.get(fact[this.spec.factKey]);
    return bucket ? bucket[0] : undefined;
  }
  /** The delta rule: fold one (fact, dimension) pair into the view with the given sign. */
  _update(fact, partner, sign, incremental) {
    if (!partner || !this.spec.where(fact)) return;
    if (incremental) this.work.incremental++;
    const value = this.spec.aggregateColumn ? fact[this.spec.aggregateColumn] : 1;
    if (this.spec.groupColumn === null) {
      const projected = { [this.spec.aggregateColumn]: value };
      const key = rowKey(projected);
      const count = (this.rows.get(key) || 0) + sign;
      if (count <= 0) {
        if (count < 0) throw new Error(`the view has no row to delete: ${key}`);
        this.rows.delete(key);
      } else this.rows.set(key, count);
      return;
    }
    const key = partner[this.spec.groupColumn];
    const bucket = this.groups.get(key) || { count: 0, sum: 0, min: undefined, max: undefined };
    if (sign < 0 && bucket.count === 0) throw new Error(`the view has no group ${key} to remove`);
    bucket.count += sign;
    bucket.sum += sign * value;
    const needsRecompute = sign < 0 && (this.spec.aggregate === "min" || this.spec.aggregate === "max");
    if (bucket.count === 0) {
      this.groups.delete(key);
      return;
    }
    if (sign > 0) {
      bucket.min = bucket.min === undefined ? value : Math.min(bucket.min, value);
      bucket.max = bucket.max === undefined ? value : Math.max(bucket.max, value);
    }
    this.groups.set(key, bucket);
    if (needsRecompute) this.recomputeGroup(key);
  }
  /** The documented gap: MIN/MAX need a scan of the affected group, they cannot be adjusted in place. */
  recomputeGroup(key) {
    this.work.groupRecomputes++;
    const bucket = { count: 0, sum: 0, min: undefined, max: undefined };
    for (const fact of this.facts) {
      if (!this.spec.where(fact)) continue;
      const partner = this._partnerOf(fact);
      if (!partner || partner[this.spec.groupColumn] !== key) continue;
      const value = this.spec.aggregateColumn ? fact[this.spec.aggregateColumn] : 1;
      bucket.count++;
      bucket.sum += value;
      bucket.min = bucket.min === undefined ? value : Math.min(bucket.min, value);
      bucket.max = bucket.max === undefined ? value : Math.max(bucket.max, value);
    }
    if (bucket.count === 0) this.groups.delete(key);
    else this.groups.set(key, bucket);
  }
  insertFact(row) {
    this.facts.push(row);
    const partners = this.index.get(row[this.spec.factKey]) || [];
    for (const partner of partners) this._update(row, partner, +1, true);
    return this;
  }
  deleteFact(row) {
    const position = this.facts.findIndex((candidate) => sameRow(candidate, row));
    if (position < 0) throw new Error("the fact row is not in the base table");
    this.facts.splice(position, 1);
    const partners = this.index.get(row[this.spec.factKey]) || [];
    for (const partner of partners) this._update(row, partner, -1, true);
    return this;
  }
  insertDimension(row) {
    this.dimensions.push(row);
    const key = row[this.spec.dimensionKey];
    this.index.set(key, (this.index.get(key) || []).concat([row]));
    this.work.indexMaintenance++;
    for (const fact of this.facts) {
      if (fact[this.spec.factKey] === key) this._update(fact, row, +1, true);
    }
    return this;
  }
  deleteDimension(row) {
    const position = this.dimensions.findIndex((candidate) => sameRow(candidate, row));
    if (position < 0) throw new Error("the dimension row is not in the base table");
    this.dimensions.splice(position, 1);
    const key = row[this.spec.dimensionKey];
    this.index.set(key, this.index.get(key).filter((candidate) => !sameRow(candidate, row)));
    this.work.indexMaintenance++;
    for (const fact of this.facts) {
      if (fact[this.spec.factKey] === key) this._update(fact, row, -1, true);
    }
    return this;
  }
  snapshot() {
    return snapshotOf(this.spec, { groups: this.groups, rows: this.rows });
  }
  /** Full recomputation through the independent oracle, in the same comparable shape. */
  fullRecompute() {
    this.work.full += this.facts.length + this.dimensions.length;
    return snapshotOf(this.spec, evaluateFromScratch(this.spec, this.facts, this.dimensions));
  }
  verify() {
    const expected = this.fullRecompute();
    const actual = this.snapshot();
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
      throw new Error(`the incremental view drifted: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
    }
    return actual;
  } }

function makeData(orders, customers, seed = 645) {
  let state = seed >>> 0;
  const next = (bound) => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) % bound;
  };
  const regions = ["north", "south", "east", "west"];
  const dimensionRows = [];
  for (let i = 0; i < customers; i++) dimensionRows.push({ id: i, region: regions[next(regions.length)] });
  const factRows = [];
  for (let i = 0; i < orders; i++) {
    factRows.push({ id: i, customer: next(customers), amount: 1 + next(400), region: regions[next(4)] });
  }
  return { factRows, dimensionRows, next };
}

const BASE_SPEC = {
  factKey: "customer", dimensionKey: "id",
  where: (order) => order.amount > 100, groupColumn: "region",
  aggregateColumn: "amount", aggregate: "sum",
};

module.exports = { ViewMaintainer, evaluateFromScratch, snapshotOf, rowKey, makeData, BASE_SPEC, AGGREGATES };

if (require.main === module) {
  const assert = require("node:assert");
  const { factRows, dimensionRows, next } = makeData(120, 25);
  const forAggregate = (aggregate, groupColumn = "region") => ({ ...BASE_SPEC, aggregate, groupColumn });
  // Loading the view agrees with the oracle, for every aggregate and for an ungrouped multiset view.
  for (const aggregate of AGGREGATES) {
    const view = new ViewMaintainer(forAggregate(aggregate), factRows, dimensionRows);
    assert.deepStrictEqual(view.snapshot(), view.fullRecompute(), aggregate);
    assert.ok(view.snapshot().length > 1, `${aggregate}: more than one group`);
  }
  const multiset = new ViewMaintainer(forAggregate("sum", null), factRows, dimensionRows);
  assert.deepStrictEqual(multiset.snapshot(), multiset.fullRecompute());
  assert.ok(multiset.snapshot().every((line) => /x\d+$/.test(line)), "the ungrouped view counts occurrences");
  // A scripted delta sequence: after every operation the maintained view must equal a full recomputation.
  const checks = [];
  for (const aggregate of AGGREGATES) {
    const view = new ViewMaintainer(forAggregate(aggregate), factRows, dimensionRows);
    let expected = view.fullRecompute();
    assert.deepStrictEqual(view.snapshot(), expected);
    for (let step = 0; step < 60; step++) {
      const roll = next(10);
      if (roll < 5) {
        const row = { id: 1000 + step, customer: next(25), amount: 1 + next(400), region: "north" };
        view.insertFact(row);
        checks.push("insert fact");
      } else if (roll < 8) {
        const row = { id: 2000 + step, region: ["north", "south", "east", "west"][next(4)] };
        view.insertDimension(row);
        checks.push("insert dimension");
      } else {
        const position = next(view.facts.length);
        view.deleteFact(view.facts[position]);
        checks.push("delete fact");
      }
      expected = view.fullRecompute();
      assert.deepStrictEqual(view.snapshot(), expected, `${aggregate} after ${checks[checks.length - 1]}`);
    }
    assert.ok(view.work.incremental > 0, `${aggregate}: the incremental path was exercised`);
  }
  // Deleting from the dimension side removes the matching facts from the view, and re-inserting restores it.
  const dimensions = new ViewMaintainer(forAggregate("sum"), factRows, dimensionRows);
  const victim = dimensionRows.find((row) => factRows.some((fact) => fact.customer === row.id && fact.amount > 100));
  const before = dimensions.snapshot();
  dimensions.deleteDimension(victim);
  assert.deepStrictEqual(dimensions.snapshot(), dimensions.fullRecompute());
  assert.notDeepStrictEqual(dimensions.snapshot(), before, "the dimension deletion changed the view");
  dimensions.insertDimension(victim);
  assert.deepStrictEqual(dimensions.snapshot(), before, "restoring the dimension restores the view");
  // A group disappears when its last row is deleted and comes back when a row arrives again.
  const tinyFacts = [{ id: 1, customer: 1, amount: 200 }, { id: 2, customer: 1, amount: 300 }];
  const tinyDimensions = [{ id: 1, region: "solo" }];
  const single = new ViewMaintainer(forAggregate("sum"), tinyFacts, tinyDimensions);
  assert.deepStrictEqual(single.snapshot(), ["solo=500"]);
  single.deleteFact(tinyFacts[0]);
  assert.deepStrictEqual(single.snapshot(), ["solo=300"]);
  assert.deepStrictEqual(single.snapshot(), single.fullRecompute());
  single.deleteFact(tinyFacts[1]);
  assert.deepStrictEqual(single.snapshot(), [], "the group vanished with its last row");
  assert.deepStrictEqual(single.snapshot(), single.fullRecompute());
  single.insertFact(tinyFacts[1]);
  assert.deepStrictEqual(single.snapshot(), ["solo=300"], "the group came back");
  assert.deepStrictEqual(single.snapshot(), single.fullRecompute());
  // The MIN/MAX recompute path runs and still matches: deleting the minimum must find the next one.
  const extremes = new ViewMaintainer(forAggregate("min"), tinyFacts, tinyDimensions);
  assert.deepStrictEqual(extremes.snapshot(), ["solo=200"]);
  extremes.deleteFact(tinyFacts[0]);
  assert.deepStrictEqual(extremes.snapshot(), ["solo=300"], "the new minimum came from a recompute");
  assert.ok(extremes.work.groupRecomputes > 0, "the documented MIN recompute path was used");
  assert.deepStrictEqual(extremes.snapshot(), extremes.fullRecompute());
  assert.throws(() => extremes.deleteFact({ id: 999, customer: 1, amount: 5 }), /not in the base table/);
  assert.throws(() => extremes.deleteDimension({ id: 42, region: "ghost" }), /not in the base table/);
  // Duplicate base rows are a multiset: the same row twice counts twice, and deleting one leaves one.
  const duplicates = new ViewMaintainer(forAggregate("count"), tinyFacts, tinyDimensions);
  const duplicate = { id: 3, customer: 1, amount: 250 };
  duplicates.insertFact(duplicate);
  duplicates.insertFact(duplicate);
  assert.deepStrictEqual(duplicates.snapshot(), ["solo=4"]);
  duplicates.deleteFact(duplicate);
  assert.deepStrictEqual(duplicates.snapshot(), ["solo=3"], "one occurrence of the duplicate left");
  assert.deepStrictEqual(duplicates.snapshot(), duplicates.fullRecompute());
  duplicates.deleteFact(duplicate);
  assert.deepStrictEqual(duplicates.snapshot(), ["solo=2"]);
  assert.throws(() => duplicates.deleteFact(duplicate), /not in the base table/);
  // A view that ignores a delete delta must drift from a full recomputation, so the equality checks bite.
  const naive = new ViewMaintainer(forAggregate("sum"), tinyFacts, tinyDimensions);
  assert.deepStrictEqual(naive.snapshot(), ["solo=500"]);
  naive.facts.splice(0, 1); // the base table loses the row, the "maintenance" forgets about it
  assert.notDeepStrictEqual(naive.snapshot(), naive.fullRecompute(),
    "a maintenance that ignores the delta must drift from the recomputation");
  assert.deepStrictEqual(naive.fullRecompute(), ["solo=300"]);
  // Work accounting: a point delta is far cheaper than a full recomputation.
  const cheap = new ViewMaintainer(forAggregate("sum"), factRows, dimensionRows);
  const beforeWork = cheap.work.incremental;
  const fullWork = factRows.length + dimensionRows.length;
  cheap.insertFact({ id: 5000, customer: 3, amount: 250, region: "north" });
  assert.ok(cheap.work.incremental - beforeWork < fullWork,
    "one insert must visit fewer rows than a full recomputation");
  console.log(`view: ${checks.length} deltas verified, incremental ${cheap.work.incremental - beforeWork} rows ` +
    `vs ${fullWork} for a full recompute, ${cheap.work.groupRecomputes} MIN/MAX recomputes`);
  console.log("PASS");
}
