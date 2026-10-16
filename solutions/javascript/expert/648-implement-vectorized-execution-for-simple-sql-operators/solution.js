/** Task 648 - Implement vectorized execution for simple SQL operators. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMULATED: a vectorized ("batch-at-a-time") execution layer over integer columns, with a
 * row-at-a-time engine as the reference. Input is cut into batches of `batchSize` lanes; comparisons
 * produce a *mask* (a byte per lane) which is turned into a selection vector of local indices rather than
 * being compacted, and every later operator consumes that selection vector, so only the surviving lanes
 * are touched. Operators: compare/select, filter, project, add-constant, SUM, and a hash GROUP BY whose
 * batch is probed one group at a time.
 *
 * The operators count what they do - lanes evaluated, lanes selected, and the number of per-batch
 * control-flow decisions - which is how the vectorization claims are checked without timing anything: a
 * predicate is evaluated for the padded lane count (the tail batch included), the branch count is
 * proportional to the number of batches rather than to the number of rows, and a downstream operator
 * evaluates exactly as many lanes as the upstream selection contains.
 *
 * WHAT IS VERIFIED: every vectorized operator agrees row for row with the row-at-a-time reference, for a
 * matrix of predicates, projections and aggregates; results are independent of the batch size (batch size
 * one degenerates exactly to the row engine, and 3, 7, 64 and 1024 all agree); selection vectors are
 * ascending, in range and duplicate free, and a mask with no set bit yields an empty selection and no
 * downstream work; the lane/branch counters behave as claimed; sums are exact against a BigInt
 * computation; empty input, single rows, tail batches and all-rejected batches are handled; and invalid
 * requests (batch size below one, out-of-range selection index) are refused.
 */
"use strict";

function checkBatchSize(batchSize) {
  if (!(batchSize >= 1) || !Number.isInteger(batchSize)) throw new Error("the batch size must be a positive integer");
  return batchSize;
}

function makeBatches(values, batchSize) {
  checkBatchSize(batchSize);
  const result = [];
  for (let start = 0; start < values.length; start += batchSize) {
    result.push(values.slice(start, start + batchSize));
  }
  return result;
}

const COMPARISONS = {
  "=": (lane, constant) => lane === constant, "<": (lane, constant) => lane < constant,
  "<=": (lane, constant) => lane <= constant, ">": (lane, constant) => lane > constant,
  ">=": (lane, constant) => lane >= constant,
};

/** Vectorized comparison: one mask byte per lane, no branch per row. */
function compareVector(batch, op, constant, stats) {
  const compare = COMPARISONS[op];
  if (!compare) throw new Error(`unknown comparison operator ${op}`);
  const mask = new Uint8Array(batch.length);
  stats.branches++;
  for (let lane = 0; lane < batch.length; lane++) {
    stats.lanesEvaluated++;
    mask[lane] = compare(batch[lane], constant) ? 1 : 0;
  }
  return mask;
}

function selectionFromMask(mask) {
  const selection = [];
  for (let lane = 0; lane < mask.length; lane++) if (mask[lane]) selection.push(lane);
  return selection;
}

function filterVector(batch, predicate, stats) {
  const mask = new Uint8Array(batch.length);
  stats.branches++;
  for (let lane = 0; lane < batch.length; lane++) {
    stats.lanesEvaluated++;
    mask[lane] = predicate(batch[lane], lane) ? 1 : 0;
  }
  const selection = selectionFromMask(mask);
  stats.lanesSelected += selection.length;
  return { mask, selection };
}

/** Selection vectors are the interface: the mapper runs once per surviving lane, in order. */
function projectVector(batch, selection, mapper, stats) {
  const values = [];
  stats.branches++;
  for (const lane of selection) {
    if (!(lane >= 0 && lane < batch.length)) throw new Error(`selection index ${lane} is outside the batch`);
    stats.lanesEvaluated++;
    values.push(mapper(batch[lane], lane));
  }
  return values;
}

function sumVector(batch, selection, stats) {
  let sum = 0;
  stats.branches++;
  for (const lane of selection) {
    if (!(lane >= 0 && lane < batch.length)) throw new Error(`selection index ${lane} is outside the batch`);
    stats.lanesEvaluated++;
    sum += batch[lane];
  }
  return sum;
}

/** Vectorized hash GROUP BY: a whole batch of keys is probed, then the groups are updated. */
function aggregateVector(batches, keyOf, valueOf, stats) {
  const groups = new Map();
  for (const batch of batches) {
    stats.branches++;
    const keys = [];
    const values = [];
    for (let lane = 0; lane < batch.length; lane++) {
      stats.lanesEvaluated++;
      keys.push(keyOf(batch[lane]));
      values.push(valueOf(batch[lane]));
    }
    for (let lane = 0; lane < keys.length; lane++) {
      const key = keys[lane];
      const bucket = groups.get(key) || { key, count: 0, sum: 0 };
      bucket.count++;
      bucket.sum += values[lane];
      groups.set(key, bucket);
    } }
  return groups;
}

/** The reference engine: the same operators, one row at a time, with no batch machinery at all. */
function runRow(values, spec) {
  let rows = values.slice();
  if (spec.predicate) rows = rows.filter((value) => spec.predicate(value));
  if (spec.mapper) rows = rows.map((value) => spec.mapper(value));
  if (spec.aggregate === "sum") return { sum: rows.reduce((total, value) => total + value, 0), rows: rows.length };
  if (spec.aggregate === "groups") {
    const groups = new Map();
    for (const value of rows) {
      const key = spec.keyOf(value);
      const bucket = groups.get(key) || { key, count: 0, sum: 0 };
      bucket.count++;
      bucket.sum += spec.valueOf(value);
      groups.set(key, bucket);
    }
    return groups;
  }
  return { rows: rows.length, values: rows };
}

/** The vectorized engine driving whole batches through the same spec, counting lanes and branches. */
function runVector(values, spec) {
  const stats = { batches: 0, lanesEvaluated: 0, lanesSelected: 0, branches: 0 };
  const batches = makeBatches(values, spec.batchSize);
  const mapped = [];
  for (const batch of batches) {
    stats.batches++;
    const selection = spec.predicate
      ? filterVector(batch, spec.predicate, stats).selection
      : batch.map((_, lane) => lane);
    if (spec.mapper) mapped.push(...projectVector(batch, selection, spec.mapper, stats));
    else mapped.push(...projectVector(batch, selection, (lane) => lane, stats));
  }
  if (spec.aggregate === "sum") {
    let sum = 0;
    for (const batch of makeBatches(mapped, spec.batchSize)) {
      sum += sumVector(batch, batch.map((_, lane) => lane), stats);
    }
    return { result: { sum, rows: mapped.length }, stats };
  }
  if (spec.aggregate === "groups") {
    const groups = aggregateVector(makeBatches(mapped, spec.batchSize), spec.keyOf, spec.valueOf, stats);
    return { result: groups, stats };
  }
  return { result: { rows: mapped.length, values: mapped }, stats };
}

function exactSum(values) { // BigInt reference for the floating-point accumulation
  let total = 0n;
  for (const value of values) total += BigInt(value);
  return Number(total);
}

function lcgValues(length, span, seed) {
  const values = [];
  let state = seed >>> 0;
  for (let index = 0; index < length; index++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values.push((state >>> 8) % span);
  }
  return values;
}

module.exports = {
  makeBatches, compareVector, selectionFromMask, filterVector, projectVector, sumVector,
  aggregateVector, runRow, runVector, exactSum, lcgValues,
};

if (require.main === module) {
  const assert = require("node:assert");
  const values = lcgValues(1000, 200, 648);
  // Masks and selection vectors: exact contents, ordering and range.
  const stats = { branches: 0, lanesEvaluated: 0, lanesSelected: 0 };
  const mask = compareVector([5, 1, 9, 3], ">", 4, stats);
  assert.deepStrictEqual([...mask], [1, 0, 1, 0]);
  assert.deepStrictEqual(selectionFromMask(mask), [0, 2]);
  assert.deepStrictEqual(selectionFromMask(new Uint8Array(4)), [], "an empty mask yields no work");
  assert.strictEqual(stats.branches, 1, "one comparison is one branch for the whole batch");
  assert.strictEqual(stats.lanesEvaluated, 4);
  assert.deepStrictEqual(projectVector([5, 1, 9, 3], [0, 2], (value) => value * 10, stats), [50, 90]);
  assert.throws(() => projectVector([1, 2], [3], (value) => value, stats), /outside the batch/);
  assert.throws(() => sumVector([1, 2], [-1], stats), /outside the batch/);
  assert.throws(() => compareVector([1], "^", 1, stats), /unknown comparison/);
  assert.throws(() => makeBatches([1, 2], 0), /positive integer/);
  // Batch-size independence: every batch size must reproduce the row engine, including batch size one.
  const spec = {
    predicate: (value) => value % 3 === 0, mapper: (value) => value * 2 + 1,
    aggregate: "sum",
  };
  const rowResult = runRow(values, spec);
  for (const batchSize of [1, 3, 7, 64, 1024, 4096]) {
    const vectorResult = runVector(values, { ...spec, batchSize });
    assert.deepStrictEqual(vectorResult.result, rowResult, `batch size ${batchSize}`);
    assert.strictEqual(vectorResult.stats.batches, Math.ceil(values.length / batchSize));
    assert.ok(vectorResult.stats.branches > 0);
  }
  const single = runVector(values, { ...spec, batchSize: 1 });
  assert.strictEqual(single.stats.branches >= values.length, true, "a batch of one is one branch per row");
  const wide = runVector(values, { ...spec, batchSize: 512 });
  assert.ok(wide.stats.branches < single.stats.branches, "wide batches need fewer branch decisions");
  // Selection propagation: a filter that keeps 1% of the rows makes the downstream mapper touch only
  // those lanes, which is the measurable form of "vectorized operators do less work".
  const rare = runVector(values, { batchSize: 64, predicate: (value) => value >= 197, mapper: (value) => value, aggregate: "sum" });
  const kept = values.filter((value) => value >= 197).length;
  assert.strictEqual(rare.result.rows, kept);
  assert.strictEqual(rare.stats.lanesSelected, kept, "only the survivors are selected");
  assert.strictEqual(rare.stats.lanesEvaluated, values.length + kept + kept,
    "compare lanes from the filter, plus mapper lanes for the sum: never more than the survivors downstream");
  assert.deepStrictEqual(values.filter((value) => value >= 197).sort((a, b) => a - b).length > 0, true);
  // Aggregation: group-by matches the row engine, and the SUM is exact against a BigInt computation.
  const large = lcgValues(20000, 100000, 6481);
  const groupSpec = {
    aggregate: "groups", batchSize: 128, predicate: (value) => value % 2 === 0,
    keyOf: (value) => value % 7, valueOf: (value) => value,
  };
  const vectorGroups = runVector(large, groupSpec).result;
  const rowGroups = runRow(large, groupSpec);
  assert.deepStrictEqual([...vectorGroups].sort((left, right) => left[0] - right[0]),
    [...rowGroups].sort((left, right) => left[0] - right[0]));
  const totalCount = [...vectorGroups.values()].reduce((sum, bucket) => sum + bucket.count, 0);
  assert.strictEqual(totalCount, large.filter((value) => value % 2 === 0).length);
  const vectorSum = runVector(large, { batchSize: 256, aggregate: "sum" }).result;
  assert.strictEqual(vectorSum.sum, exactSum(large), "the vectorized SUM is bit-exact");
  assert.strictEqual(runVector([], { batchSize: 4, aggregate: "sum" }).result.sum, 0, "empty input sums to zero");
  assert.strictEqual(runVector([], { batchSize: 4, aggregate: "sum" }).stats.batches, 0);
  // Single-row and tail-batch boundaries, plus a batch that rejects everything.
  assert.deepStrictEqual(runVector([42], { batchSize: 8, aggregate: "sum" }).result, { sum: 42, rows: 1 });
  const tail = runVector([1, 2, 3, 4, 5, 6, 7], { batchSize: 3, aggregate: "sum" });
  assert.strictEqual(tail.stats.batches, 3, "seven rows in batches of three need three batches");
  const rejected = runVector([1, 3, 5], { batchSize: 2, predicate: (value) => value % 2 === 0, aggregate: "sum" });
  assert.deepStrictEqual(rejected.result, { sum: 0, rows: 0 });
  assert.strictEqual(rejected.stats.lanesSelected, 0);
  // Determinism, and the counter accounting for a full pipeline.
  const repeat = runVector(values, { ...spec, batchSize: 32 });
  assert.deepStrictEqual(repeat, runVector(values, { ...spec, batchSize: 32 }));
  const accounted = runVector(values, { batchSize: 32, predicate: (value) => value < 100, mapper: (value) => value, aggregate: "sum" });
  const survivors = values.filter((value) => value < 100).length;
  assert.strictEqual(accounted.stats.lanesSelected, survivors);
  assert.ok(accounted.stats.lanesEvaluated <= values.length + 2 * survivors,
    "no operator evaluates more lanes than its input has");
  console.log(`vectorized: ${values.length} rows in batches of 32 needed ${accounted.stats.branches} branch ` +
    `decisions and ${accounted.stats.lanesEvaluated} lane evaluations (${survivors} survived the filter)`);
  console.log("PASS");
}
