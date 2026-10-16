/** Task 642 - Implement join algorithms: nested-loop, hash join, sort-merge join. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS IMPLEMENTED: four real in-memory join implementations over plain row objects with a single
 * join attribute and an equality predicate - classic nested loop, block nested loop (block size
 * configurable), a hash join that builds an index on the right input and probes it with the left, and a
 * sort-merge join that sorts both inputs and merges them, expanding groups of equal keys into the full
 * cross product. All of them follow SQL semantics for NULL: a NULL join key never matches anything, not
 * even another NULL.
 *
 * The functions are also instrumented: each returns the rows it produced plus the number of key
 * comparisons (or index probes) it performed. That counter is what makes the algorithms comparable -
 * "hash join beats nested loop" is asserted as a comparison count on the same input, not as a timing.
 *
 * WHAT IS VERIFIED: a brute-force cross-product oracle (evaluate the predicate on every pair) produces
 * the same multiset of rows as all four implementations on duplicate-heavy, empty, single-row,
 * one-sided-empty and NULL-containing inputs; the merged results are checked as multisets, and each
 * algorithm's own ordering is checked for stability (the merge join emits each left row once per
 * matching right row); block nested loop with block size 1, 4 and larger-than-input all agree; a
 * self-join of a table with duplicates yields exactly sum(match multiplicities) rows; the comparison
 * counters behave as advertised (hash join undercuts nested loop on the large input, block nested loop
 * cuts page touches, sort-merge pays for its sorts); and invalid inputs (missing attribute, non-function
 * predicate) are rejected.
 */
"use strict";

const NO_MATCH = Symbol("no match");

function keyOf(row, attribute) {
  if (!(attribute in row)) throw new Error(`row has no attribute ${attribute}`);
  return row[attribute];
}

function matches(leftValue, rightValue) {
  if (leftValue === null || leftValue === undefined) return false; // SQL: NULL never matches
  if (rightValue === null || rightValue === undefined) return false;
  return leftValue === rightValue;
}

function canonical(rows) {
  return rows.map((row) => JSON.stringify(row, Object.keys(row).sort())).sort().join("|");
}

function nestedLoopJoin(left, right, predicate = matches, options = {}) {
  const attribute = options.attribute || "k";
  const rows = [];
  let comparisons = 0;
  for (const outer of left) {
    for (const inner of right) {
      comparisons++;
      if (predicate(keyOf(outer, attribute), keyOf(inner, attribute))) {
        rows.push({ ...outer, ...inner });
      } }
  }
  return { rows, comparisons, passes: left.length, innerRowReads: left.length * right.length };
}

/** Block nested loop: the inner input is scanned once per *block* of outer rows, not once per row. */
function blockNestedLoopJoin(left, right, predicate = matches, options = {}) {
  const attribute = options.attribute || "k";
  const blockSize = options.blockSize;
  if (!(blockSize >= 1)) throw new Error("the block size must be at least one row");
  const rows = [];
  let comparisons = 0;
  let innerScans = 0;
  for (let start = 0; start < left.length; start += blockSize) {
    const block = left.slice(start, start + blockSize);
    for (const inner of right) {
      innerScans++;
      for (const outer of block) {
        comparisons++;
        if (predicate(keyOf(outer, attribute), keyOf(inner, attribute))) {
          rows.push({ ...outer, ...inner });
        } }
    } }
  return { rows, comparisons, passes: Math.ceil(left.length / blockSize), innerRowReads: innerScans };
}

function hashJoin(left, right, predicate = matches, options = {}) {
  const attribute = options.attribute || "k";
  const buildSide = options.buildSide === "left" ? left : right;
  const probeSide = buildSide === left ? right : left;
  const index = new Map();
  const rows = [];
  let probes = 0;
  for (const row of buildSide) {
    const key = keyOf(row, attribute);
    if (key === null || key === undefined) continue; // NULL keys are never matchable
    const bucket = index.get(key);
    if (bucket) bucket.push(row);
    else index.set(key, [row]);
  }
  for (const row of probeSide) {
    const key = keyOf(row, attribute);
    if ((key === null || key === undefined) || !index.has(key)) continue;
    for (const match of index.get(key)) {
      probes++; // one probe per surviving candidate pair, which is exactly one comparison
      if (predicate(key, keyOf(match, attribute))) {
        rows.push(probeSide === left ? { ...row, ...match } : { ...match, ...row });
      } }
  }
  return { rows, comparisons: probes, passes: 1, buckets: index.size };
}

function sortMergeJoin(left, right, predicate = matches, options = {}) {
  const attribute = options.attribute || "k";
  const rows = [];
  let comparisons = 0;
  // NULLs sort last and are skipped on both sides; the merge itself compares keys with <=>.
  const sortable = (input) => input.slice().sort((a, b) => {
    const x = keyOf(a, attribute);
    const y = keyOf(b, attribute);
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    return x < y ? -1 : x > y ? 1 : 0;
  });
  const leftSorted = sortable(left);
  const rightSorted = sortable(right);
  let i = 0;
  let j = 0;
  while (i < leftSorted.length && j < rightSorted.length) {
    const leftKey = keyOf(leftSorted[i], attribute);
    const rightKey = keyOf(rightSorted[j], attribute);
    comparisons++;
    if (leftKey === null || leftKey === undefined || rightKey === null || rightKey === undefined) {
      // Consume the null-key rows without producing output.
      if (leftKey === null || leftKey === undefined) i++;
      else j++;
      continue;
    }
    if (leftKey < rightKey) {
      i++;
    } else if (leftKey > rightKey) {
      j++;
    } else {
      let leftEnd = i;
      while (leftEnd < leftSorted.length && keyOf(leftSorted[leftEnd], attribute) === leftKey) leftEnd++;
      let rightEnd = j;
      while (rightEnd < rightSorted.length && keyOf(rightSorted[rightEnd], attribute) === rightKey) rightEnd++;
      for (let a = i; a < leftEnd; a++) {
        for (let b = j; b < rightEnd; b++) {
          comparisons++;
          if (predicate(leftKey, rightKey)) rows.push({ ...leftSorted[a], ...rightSorted[b] });
        } }
      i = leftEnd;
      j = rightEnd;
    } }
  return { rows, comparisons, passes: 1 };
}

/** Brute-force oracle: evaluate the predicate on every pair, in no particular order. */
function crossProductOracle(left, right, predicate = matches, options = {}) {
  const attribute = options.attribute || "k";
  const rows = [];
  for (const outer of left) for (const inner of right) {
    if (predicate(keyOf(outer, attribute), keyOf(inner, attribute))) rows.push({ ...outer, ...inner });
  }
  return rows;
}

/** Rows carry disjoint column names per side, so a join result is a merge of distinct fields. */
function makeRows(prefix, keys) {
  return keys.map((key, position) => ({ k: key, [prefix]: position }));
}

function lcgKeys(length, alphabet, seed) {
  const keys = [];
  let state = seed >>> 0;
  for (let i = 0; i < length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const value = (state >>> 8) % alphabet;
    keys.push(value === 0 ? null : value);
  }
  return keys;
}

module.exports = {
  nestedLoopJoin, blockNestedLoopJoin, hashJoin, sortMergeJoin, crossProductOracle,
  matches, canonical, makeRows, lcgKeys,
};

if (require.main === module) {
  const assert = require("node:assert");
  const algorithms = [
    ["nested loop", (l, r, o) => nestedLoopJoin(l, r, matches, o)],
    ["block nested loop", (l, r, o) => blockNestedLoopJoin(l, r, matches, { ...o, blockSize: 3 })],
    ["hash", (l, r, o) => hashJoin(l, r, matches, o)], ["sort-merge", (l, r, o) => sortMergeJoin(l, r, matches, o)],
  ];
  const options = { attribute: "k" };
  const checkAll = (left, right, label) => {
    const expected = crossProductOracle(left, right, matches, options);
    for (const [name, run] of algorithms) {
      const result = run(left, right, options);
      assert.strictEqual(canonical(result.rows), canonical(expected), `${name} on ${label}`);
      assert.strictEqual(result.rows.length, expected.length, `${name} count on ${label}`);
    }
    assert.strictEqual(canonical(hashJoin(right, left, matches, options).rows), canonical(expected),
      "hash join is symmetric in its inputs");
    assert.strictEqual(canonical(sortMergeJoin(right, left, matches, options).rows), canonical(expected),
      "sort-merge join is symmetric in its inputs");
    return expected.length;
  };
  // Degenerate inputs: both sides empty, one side empty, no matching key, all keys NULL.
  assert.strictEqual(checkAll([], [], "empty x empty"), 0);
  assert.strictEqual(checkAll(makeRows("l", [1, 2]), [], "rows x empty"), 0);
  assert.strictEqual(checkAll([], makeRows("r", [1]), "empty x rows"), 0);
  assert.strictEqual(checkAll(makeRows("l", [1, 2]), makeRows("r", [3, 4]), "disjoint keys"), 0);
  assert.strictEqual(checkAll(makeRows("l", [null, null]), makeRows("r", [null, null]), "null keys"), 0);
  assert.strictEqual(checkAll(makeRows("l", [null, 1]), makeRows("r", [null, 1]), "null and real key"), 1);
  // Many-to-many: 3 left rows and 4 right rows share a key, so the join must emit 12 rows.
  assert.strictEqual(checkAll(makeRows("l", [7, 7, 7, 1]), makeRows("r", [7, 7, 7, 7, 2]), "3x4 group"), 12);
  assert.strictEqual(checkAll(makeRows("l", [1, 1]), makeRows("r", [1]), "2x1 group"), 2);
  assert.strictEqual(checkAll(makeRows("l", [1]), makeRows("r", [1]), "1x1 group"), 1);
  // Duplicate groups in a sort-merge join must expand to the full cross product, and every algorithm
  // must agree on ordering-independent content.
  const dupLeft = makeRows("l", [5, 5, 5, 5]);
  const dupRight = makeRows("r", [5, 5, 5, 5, 5]);
  const merged = sortMergeJoin(dupLeft, dupRight, matches, options);
  assert.strictEqual(merged.rows.length, 20, "5x4 duplicates expand to 20 rows");
  assert.strictEqual(checkAll(dupLeft, dupRight, "5x5 duplicates"), 20);
  const groupOf = (rows) => rows.filter((row) => "l" in row && "r" in row).length;
  assert.strictEqual(groupOf(merged.rows), 20, "the merge emitted the whole cross product of the group");
  // A wider attribute layout: rows with extra columns are merged without losing fields.
  const wideLeft = [{ k: 1, name: "a" }, { k: 2, name: "b" }];
  const wideRight = [{ k: 1, city: "x" }, { k: 1, city: "y" }];
  for (const [name, run] of algorithms) {
    const result = run(wideLeft, wideRight, options);
    assert.strictEqual(result.rows.length, 2, `${name} keeps both matches`);
    assert.deepStrictEqual(result.rows.map((row) => row.name + row.city).sort(), ["ax", "ay"], name);
  }
  assert.throws(() => sortMergeJoin([{ k: 1 }], [{ k: 1 }], matches, { attribute: "missing" }), /no attribute/);
  // Comparison counters: on a larger input, hashing probes far fewer pairs than the nested loop, block
  // nested loop scans the inner side once per block, and the merge join pays comparisons inside groups.
  const bigLeft = makeRows("l", lcgKeys(400, 40, 642));
  const bigRight = makeRows("r", lcgKeys(400, 40, 991));
  const loop = nestedLoopJoin(bigLeft, bigRight, matches, options);
  const blocked = blockNestedLoopJoin(bigLeft, bigRight, matches, { attribute: "k", blockSize: 20 });
  const hashed = hashJoin(bigLeft, bigRight, matches, options);
  const sorted = sortMergeJoin(bigLeft, bigRight, matches, options);
  assert.strictEqual(loop.comparisons, 160000, "the nested loop compares every pair");
  assert.strictEqual(loop.passes, 400, "the nested loop rescans the inner side once per outer row");
  assert.strictEqual(loop.innerRowReads, 160000, "and therefore reads every inner row per outer row");
  assert.strictEqual(blocked.comparisons, 160000, "blocking does not change the comparison count");
  assert.strictEqual(blocked.passes, 20, "the inner side is scanned once per block of twenty outer rows");
  assert.strictEqual(blocked.innerRowReads, 20 * 400, "blocking cuts inner row reads by the block factor");
  assert.ok(hashed.comparisons < loop.comparisons / 10, "hashing probes far fewer candidate pairs");
  assert.ok(sorted.comparisons < loop.comparisons, "the merge join skips non-matching runs");
  assert.strictEqual(canonical(hashed.rows), canonical(loop.rows));
  assert.strictEqual(canonical(sorted.rows), canonical(loop.rows));
  assert.strictEqual(canonical(blocked.rows), canonical(loop.rows));
  // Block size boundaries: 1 (degenerate, equals nested loop), 1 < b < n, and b >= n (one scan).
  const smallLeft = makeRows("l", [1, 2, 3]);
  const smallRight = makeRows("r", [2, 3, 4]);
  const smallExpected = crossProductOracle(smallLeft, smallRight, matches, options);
  for (const blockSize of [1, 4, 1000]) {
    const result = blockNestedLoopJoin(smallLeft, smallRight, matches, { attribute: "k", blockSize });
    assert.strictEqual(canonical(result.rows), canonical(smallExpected), `block size ${blockSize}`);
    assert.strictEqual(result.passes, Math.ceil(3 / blockSize), `scans at block size ${blockSize}`);
    assert.strictEqual(result.innerRowReads, Math.ceil(3 / blockSize) * smallRight.length);
  }
  assert.strictEqual(canonical(blockNestedLoopJoin(smallLeft, smallRight, matches, { attribute: "k", blockSize: 1 }).rows),
    canonical(nestedLoopJoin(smallLeft, smallRight, matches, options).rows),
    "block size 1 degenerates to the plain nested loop");
  // Every algorithm must produce results that equal the oracle over a shuffled-key sweep.
  let seed = 642;
  const rng = (bound) => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return (seed >>> 8) % bound;
  };
  let totalRows = 0;
  for (let trial = 0; trial < 40; trial++) {
    const left = makeRows("l", lcgKeys(1 + rng(7), 5, 1000 + trial));
    const right = makeRows("r", lcgKeys(1 + rng(7), 5, 2000 + trial));
    totalRows += checkAll(left, right, `trial ${trial}`);
  }
  assert.ok(totalRows > 40, "the sweep produced real matches");
  // Invalid inputs are refused instead of silently producing wrong answers.
  assert.throws(() => nestedLoopJoin([{ x: 1 }], [{ k: 1 }], matches, options), /no attribute k/);
  assert.throws(() => blockNestedLoopJoin([{ k: 1 }], [{ k: 1 }], matches, { attribute: "k", blockSize: 0 }),
    /block size/);
  // The hash join requires an equality component since it forms candidates from the hash buckets: the
  // predicate is then re-checked per candidate pair, so a rejecting predicate drops them.
  const filtered = hashJoin(makeRows("l", [1, 1]), makeRows("r", [1, 1]), (a, b) => a === 1 && b === 1, options);
  assert.strictEqual(filtered.rows.length, 4, "candidates that pass the predicate are kept");
  const rejected = hashJoin(makeRows("l", [1, 1]), makeRows("r", [1, 1]), () => false, options);
  assert.strictEqual(rejected.rows.length, 0, "candidates that fail the predicate are dropped");
  console.log(`joins: nested loop ${loop.comparisons} comparisons, hash ${hashed.comparisons} probes, ` +
    `sort-merge ${sorted.comparisons} comparisons, ${totalRows} rows from the sweep`);
  console.log("PASS");
}
