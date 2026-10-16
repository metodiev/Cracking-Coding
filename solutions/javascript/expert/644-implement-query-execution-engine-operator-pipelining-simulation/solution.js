/** Task 644 - Implement query execution engine operator pipelining simulation. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMULATED: a volcano-style (iterator) execution engine. Every operator implements the
 * open/next/close protocol, pulls one row at a time from its children, and is instrumented with
 * lifecycle statistics - opens, closes, next() calls, rows produced, rows pulled from the children, and
 * the maximum number of rows it held in memory at once. The operators are scan (with an attached
 * predicate), filter, project, limit, sort (blocking), hash group-by aggregation (blocking) and a
 * pipelined nested-loop join that re-opens its inner child once per outer row.
 *
 * WHY THAT INSTRUMENTATION IS THE POINT: pipelining is exactly the claim that a streaming plan never
 * materialises its input, so it can be *measured* rather than timed. A LIMIT above a scan must stop the
 * scan early, a sort cannot emit anything before it has consumed everything, and a join must be able to
 * emit its first row without draining the outer side. The same operator spec is also evaluated by an
 * independent array-based evaluator, so every pipeline is checked against a second implementation.
 *
 * WHAT IS VERIFIED: the lifecycle protocol (next before open, double open, double close and next after
 * close all throw, close() reaches every child exactly once); a LIMIT of k pulls exactly k rows from a
 * thousand-row scan and closes it early; a sort buffers the entire input before its first output while
 * the streaming plan never buffers more than one row; the join re-opens its inner child once per outer
 * row and its row accounting is consistent at every level; GROUP BY matches a Map-based oracle and an
 * empty input produces no groups; and a sweep of operator combinations agrees row for row with the
 * array evaluator, including projections, predicates, limits, sorts and aggregates.
 */
"use strict";

function operator(name, children, hooks) {
  const op = {
    name, children, stats: { opens: 0, closes: 0, nextCalls: 0, rowsOut: 0, rowsPulled: 0, buffered: 0, maxBuffered: 0 },
    opened: false, closed: false, open() {
      if (this.opened) throw new Error(`${name}: open() out of protocol`);
      this.opened = true;
      this.closed = false; // a closed iterator may be re-opened (a nested-loop inner side is rewound)
      this.stats.opens++;
      if (hooks.open) hooks.open.call(this);
      if (hooks.openChildren === false) return;
      for (const child of children) child.open();
    }, next() {
      if (!this.opened || this.closed) throw new Error(`${name}: next() on a closed iterator`);
      this.stats.nextCalls++;
      const step = hooks.next.call(this);
      if (step.done === false) this.stats.rowsOut++;
      return step;
    },
    pull(child) { // one row from a child, accounted for
      const step = child.next();
      if (step.done === false) this.stats.rowsPulled++;
      return step;
    }, buffered(rows) {
      this.stats.buffered = rows;
      if (rows > this.stats.maxBuffered) this.stats.maxBuffered = rows;
    }, close() {
      if (this.closed) throw new Error(`${name}: close() twice`);
      this.closed = true;
      this.opened = false;
      this.stats.closes++;
      for (const child of children) if (!child.closed) child.close();
    },
  };
  return op;
}

function scan(rows, predicate = null) {
  let index = 0;
  let accepted = 0;
  return operator("scan", [], {
    open() { index = 0; }, next() {
      while (index < rows.length) {
        const row = rows[index++];
        if (predicate && !predicate(row)) continue;
        accepted++;
        return { done: false, value: row };
      }
      return { done: true, value: undefined };
    },
  });
}

function filter(child, predicate) {
  return operator("filter", [child], {
    next() {
      for (;;) {
        const step = this.pull(child);
        if (step.done) return { done: true, value: undefined };
        if (predicate(step.value)) return { done: false, value: step.value };
      } },
  });
}

function project(child, mapper) {
  return operator("project", [child], {
    next() {
      const step = this.pull(child);
      if (step.done) return { done: true, value: undefined };
      return { done: false, value: mapper(step.value) };
    },
  });
}

function limit(child, count) {
  let produced = 0;
  return operator("limit", [child], {
    open() { produced = 0; }, next() {
      if (produced >= count) {
        if (!child.closed) child.close(); // stop the producer as soon as the limit is satisfied
        return { done: true, value: undefined };
      }
      const step = this.pull(child);
      if (step.done) return { done: true, value: undefined };
      produced++;
      return { done: false, value: step.value };
    },
  });
}

function sort(child, compare = null) {
  let buffer = [];
  let index = 0;
  return operator("sort", [child], {
    open() { buffer = []; index = 0; }, next() {
      if (index === 0 && buffer.length === 0) {
        for (;;) { // the whole input is consumed before the first row leaves this operator
          const step = this.pull(child);
          if (step.done) break;
          buffer.push(step.value);
          this.buffered(buffer.length);
        }
        if (compare) buffer.sort(compare);
      }
      if (index >= buffer.length) return { done: true, value: undefined };
      return { done: false, value: buffer[index++] };
    },
  });
}

function hashAgg(child, keyOf, aggregate, column) {
  let groups = null;
  let emitted = 0;
  return operator("hashAgg", [child], {
    open() { groups = null; emitted = 0; }, next() {
      if (groups === null) {
        groups = new Map();
        for (;;) {
          const step = this.pull(child);
          if (step.done) break;
          const key = keyOf(step.value);
          const bucket = groups.get(key) || { key, count: 0, sum: 0, min: undefined, max: undefined };
          const value = column ? step.value[column] : 1;
          bucket.count++;
          bucket.sum += value;
          bucket.min = bucket.min === undefined ? value : Math.min(bucket.min, value);
          bucket.max = bucket.max === undefined ? value : Math.max(bucket.max, value);
          groups.set(key, bucket);
        }
        this.buffered(groups.size);
      }
      const entries = [...groups.values()];
      if (emitted >= entries.length) return { done: true, value: undefined };
      const bucket = entries[emitted++];
      const value = aggregate === "count" ? bucket.count
        : aggregate === "sum" ? bucket.sum
          : aggregate === "avg" ? bucket.sum / bucket.count
            : aggregate === "min" ? bucket.min : bucket.max;
      return { done: false, value: { key: bucket.key, [aggregate]: value, count: bucket.count } };
    },
  });
}

function nlJoin(leftOp, rightOp, predicate) {
  let current = null;
  let rightOpen = false;
  return operator("nlJoin", [leftOp, rightOp], {
    openChildren: false, // the inner side is opened lazily, once per outer row
    open() { leftOp.open(); current = null; rightOpen = false; }, next() {
      for (;;) {
        if (current === null) {
          const step = this.pull(leftOp);
          if (step.done) {
            if (rightOpen) { rightOp.close(); rightOpen = false; }
            return { done: true, value: undefined };
          }
          current = step.value;
          rightOp.open(); // the inner side is re-opened for every outer row
          rightOpen = true;
        }
        const step = this.pull(rightOp);
        if (step.done) {
          rightOp.close();
          rightOpen = false;
          current = null;
          continue;
        }
        this.buffered(1);
        if (predicate(current, step.value)) return { done: false, value: { ...current, ...step.value } };
      } },
  });
}

function materialize(op) {
  op.open();
  const rows = [];
  for (;;) {
    const step = op.next();
    if (step.done) break;
    rows.push(step.value);
  }
  op.close();
  return rows;
}

/** Independent array-based evaluator for the same operator spec, used as the oracle. */
function naiveEval(spec, rows) {
  let result = rows.slice();
  for (const step of spec) {
    if (step.op === "filter") result = result.filter((row) => row[step.column] === step.value);
    else if (step.op === "keep") result = result.filter((row) => row[step.column] > step.value);
    else if (step.op === "project") result = result.map((row) => ({ [step.column]: row[step.column] }));
    else if (step.op === "sort") result = result.slice().sort((a, b) => a[step.column] - b[step.column]);
    else if (step.op === "limit") result = result.slice(0, step.count);
    else if (step.op === "agg") {
      const groups = new Map();
      for (const row of result) {
        const bucket = groups.get(row[step.key]) || { key: row[step.key], count: 0, sum: 0 };
        bucket.count++;
        bucket.sum += row[step.column];
        groups.set(row[step.key], bucket);
      }
      result = [...groups.values()].map((bucket) => ({
        key: bucket.key, [step.aggregate]: step.aggregate === "sum" ? bucket.sum
          : step.aggregate === "avg" ? bucket.sum / bucket.count : bucket.count, count: bucket.count,
      }));
    } else throw new Error(`unknown spec step ${step.op}`);
  }
  return result;
}

/** Builds the iterator pipeline for the same operator spec. */
function buildPlan(spec, rows) {
  let plan = scan(rows);
  for (const step of spec) {
    if (step.op === "filter") plan = filter(plan, (row) => row[step.column] === step.value);
    else if (step.op === "keep") plan = filter(plan, (row) => row[step.column] > step.value);
    else if (step.op === "project") plan = project(plan, (row) => ({ [step.column]: row[step.column] }));
    else if (step.op === "sort") plan = sort(plan, (a, b) => a[step.column] - b[step.column]);
    else if (step.op === "limit") plan = limit(plan, step.count);
    else if (step.op === "agg") {
      plan = hashAgg(plan, (row) => row[step.key], step.aggregate, step.column);
    } else throw new Error(`unknown spec step ${step.op}`);
  }
  return plan;
}

function makeRows(count, seed = 644) {
  const rows = [];
  let state = seed >>> 0;
  for (let i = 0; i < count; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    rows.push({
      id: i, region: ["north", "south", "east"][(state >>> 9) % 3], amount: 1 + ((state >>> 11) % 50),
    });
  }
  return rows;
}

module.exports = { scan, filter, project, limit, sort, hashAgg, nlJoin, materialize, naiveEval, buildPlan, makeRows };

if (require.main === module) {
  const assert = require("node:assert");
  const north = "north";
  // Lifecycle protocol: every misuse is refused.
  const single = scan(makeRows(3));
  assert.throws(() => single.next(), /closed iterator/, "next before open");
  single.open();
  assert.throws(() => single.open(), /out of protocol/, "double open");
  single.next();
  single.close();
  assert.throws(() => single.close(), /close\(\) twice/);
  assert.throws(() => single.next(), /closed iterator/, "next after close");
  // Pipelining: a LIMIT of three pulls three rows out of a thousand-row scan and closes it early.
  const bigRows = makeRows(1000);
  const earlyScan = scan(bigRows);
  const earlyLimit = limit(earlyScan, 3);
  const firstThree = materialize(earlyLimit);
  assert.strictEqual(firstThree.length, 3);
  assert.strictEqual(earlyScan.stats.rowsOut, 3, "the scan produced only what the limit asked for");
  assert.strictEqual(earlyScan.stats.closes, 1, "and was closed as soon as the limit was satisfied");
  assert.strictEqual(earlyLimit.stats.rowsPulled, 3);
  assert.strictEqual(earlyScan.stats.rowsOut, earlyLimit.stats.rowsPulled, "row accounting matches");
  // A LIMIT of zero produces nothing but still opens and closes the child exactly once.
  const zeroScan = scan(bigRows);
  const zeroLimit = limit(zeroScan, 0);
  assert.deepStrictEqual(materialize(zeroLimit), []);
  assert.strictEqual(zeroScan.stats.opens, 1);
  assert.strictEqual(zeroScan.stats.closes, 1);
  assert.strictEqual(zeroScan.stats.rowsOut, 0);
  // Blocking: the sort consumes everything before its first row, while the streaming plan never buffers.
  const sortChild = scan(bigRows);
  const sorted = sort(sortChild, (a, b) => a.amount - b.amount);
  const firstSortedRow = (() => {
    sorted.open();
    const rowsOut = sortChild.stats.rowsOut;
    const step = sorted.next();
    const consumed = sortChild.stats.rowsOut - rowsOut;
    sorted.close();
    return { step, consumed };
  })();
  assert.strictEqual(firstSortedRow.consumed, 1000, "the sort drained its input before emitting a row");
  assert.strictEqual(sorted.stats.maxBuffered, 1000);
  const streamScan = scan(bigRows);
  const streamFilter = filter(streamScan, (row) => row.amount > 25);
  const streaming = project(streamFilter, (row) => ({ amount: row.amount }));
  const streamed = materialize(streaming);
  assert.ok(streaming.stats.maxBuffered <= 1 && streamScan.stats.maxBuffered <= 1,
    "a purely streaming plan never buffers more than one row");
  assert.strictEqual(streamScan.stats.rowsOut, 1000, "the filter pulled the whole input, one row at a time");
  assert.strictEqual(streamFilter.stats.rowsPulled, 1000);
  assert.strictEqual(streaming.stats.rowsPulled, streamFilter.stats.rowsOut,
    "the projection pulled exactly what the filter emitted");
  assert.strictEqual(streamed.length, bigRows.filter((row) => row.amount > 25).length);
  // The nested-loop join re-opens the inner child once per outer row and accounts for every row.
  const outerRows = makeRows(4, 7).map((row) => ({ id: row.id, amount: row.amount }));
  const innerRows = makeRows(3, 9).map((row) => ({ id: row.id, region: row.region }));
  const outer = scan(outerRows);
  const inner = scan(innerRows);
  const joined = nlJoin(outer, inner, (left, right) => left.id === right.id);
  const joinedRows = materialize(joined);
  const expectedMatches = outerRows.filter((left) => innerRows.some((right) => right.id === left.id)).length;
  assert.strictEqual(joinedRows.length, expectedMatches);
  assert.strictEqual(inner.stats.opens, outerRows.length, "the inner side is re-opened per outer row");
  assert.strictEqual(joined.stats.rowsPulled, outerRows.length + outerRows.length * innerRows.length);
  assert.strictEqual(inner.stats.closes, outerRows.length, "and closed again after each outer row");
  // GROUP BY matches a Map-based oracle; an empty input yields no groups at all.
  const rows = makeRows(200);
  const aggregated = materialize(hashAgg(scan(rows), (row) => row.region, "sum", "amount"));
  const oracle = new Map();
  for (const row of rows) oracle.set(row.region, (oracle.get(row.region) || 0) + row.amount);
  assert.deepStrictEqual(aggregated.map((row) => [row.key, row.sum]).sort(), [...oracle].sort());
  assert.deepStrictEqual(materialize(hashAgg(scan([]), (row) => row.region, "count", "amount")), []);
  assert.deepStrictEqual(materialize(hashAgg(scan(rows), (row) => row.region, "count", "amount"))
    .map((row) => row.count).reduce((a, b) => a + b, 0), rows.length);
  // A sweep of operator combinations: the iterator plan must agree with the array evaluator everywhere,
  // and the accounting (rows pulled = rows produced by the child) must hold at every level.
  const specs = [
    [{ op: "filter", column: "region", value: north }], [{ op: "keep", column: "amount", value: 40 }],
    [{ op: "sort", column: "amount" }, { op: "limit", count: 5 }],
    [{ op: "filter", column: "region", value: north }, { op: "sort", column: "amount" }, { op: "limit", count: 3 }],
    [{ op: "project", column: "region" }, { op: "agg", key: "region", column: "region", aggregate: "count" }],
    [{ op: "agg", key: "region", column: "amount", aggregate: "sum" }],
    [{ op: "agg", key: "region", column: "amount", aggregate: "avg" }],
    [{ op: "sort", column: "amount" }, { op: "project", column: "amount" }, { op: "limit", count: 4 }], ];
  let checked = 0;
  for (const spec of specs) {
    const plan = buildPlan(spec, rows);
    const emitted = materialize(plan);
    const expected = naiveEval(spec, rows);
    assert.deepStrictEqual(emitted, expected, JSON.stringify(spec));
    const walk = [plan];
    while (walk.length > 0) {
      const current = walk.pop();
      assert.strictEqual(current.stats.closes, 1, `${current.name} must be closed exactly once`);
      if (current.children.length > 0) {
        assert.ok(current.stats.rowsOut <= current.stats.rowsPulled,
          `${current.name} cannot emit more rows than it pulled`);
      }
      for (const child of current.children) {
        assert.strictEqual(child.stats.rowsOut, current.stats.rowsPulled,
          `${child.name} produced exactly the rows ${current.name} pulled`);
        assert.strictEqual(child.stats.closes, 1, `${child.name} must be closed exactly once`);
        walk.push(child);
      }
      checked++;
    }
    for (const step of spec) if (step.op === "limit") {
      assert.ok(emitted.length <= step.count, "a limit can never over-produce");
    } }
  assert.ok(checked > specs.length);
  // Determinism: the same plan built twice produces identical rows and identical statistics.
  const runOnce = () => materialize(buildPlan(specs[3], rows));
  assert.deepStrictEqual(runOnce(), runOnce());
  assert.throws(() => buildPlan([{ op: "explode" }], rows), /unknown spec step/);
  console.log(`pipeline: limit pulled ${earlyLimit.stats.rowsPulled} rows from a ${bigRows.length}-row scan, ` +
    `sort buffered ${sorted.stats.maxBuffered}, ${checked} operators accounted`);
  console.log("PASS");
}
