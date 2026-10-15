/**
 * Task 352 - Implement fork-join style parallel sum (threads). (Tier: Intermediate, CTCI Ch 15)
 *
 * The split is pure arithmetic, so JavaScript follows the documented task-195 simulation
 * pattern: `parallelSum` forks the left half as a pending async task while the right half is
 * computed inline, then awaits the fork (the join). Node's single-threaded event loop runs
 * those tasks sequentially, so the numbers must - and do - match the sequential sum, while
 * the fork tree, leaf count and spawned-task count are accounted exactly as a real fork-join
 * pool would report them. THRESHOLD = 256: leaf ranges hold at most 256 elements, and a
 * range splits in half until every piece is <= 256, so for power-of-two sizes leaves == n/256.
 */
"use strict";

const THRESHOLD = 256;

function sequentialSum(values) {
  return values.reduce((a, b) => a + b, 0);
}

/** Independent (iterative) oracle for the number of leaves the split policy creates. */
function countLeaves(size) {
  let leaves = 0;
  const pending = [[0, size]];
  while (pending.length > 0) {
    const [lo, hi] = pending.pop();
    if (hi - lo <= THRESHOLD) {
      leaves += 1;
    } else {
      const mid = Math.floor((lo + hi) / 2);
      pending.push([lo, mid]);
      pending.push([mid, hi]);
    }
  }
  return leaves;
}

async function parallelSum(values) {
  const stats = { leaves: 0, spawned: 0 };

  async function sumRange(lo, hi) {
    if (hi - lo <= THRESHOLD) {                   // leaf task: sum without splitting
      stats.leaves += 1;
      let sum = 0;
      for (let i = lo; i < hi; i++) sum += values[i];
      return sum;
    }
    const mid = Math.floor((lo + hi) / 2);
    stats.spawned += 1;
    const forked = sumRange(lo, mid);             // fork: pending async task
    const right = await sumRange(mid, hi);        // compute one half inline
    return (await forked) + right;                // join the forked half
  }

  const total = await sumRange(0, values.length);
  return { total, stats };
}

function lcgArray(size, seed = 24680) {
  const values = [];
  for (let i = 0; i < size; i++) {
    seed = (seed * 75 + 74) % 65537;
    values.push(seed);
  }
  return values;
}

async function main() {
  const assert = require("node:assert");

  const empty = await parallelSum([]);
  assert.strictEqual(empty.total, 0, "empty array sums to 0");
  assert.strictEqual(empty.stats.leaves, 1, "empty range is a single leaf");
  assert.strictEqual(empty.stats.spawned, 0, "empty range spawns nothing");

  const sizes = [1, 7, THRESHOLD, THRESHOLD + 1, 2 * THRESHOLD, 3 * THRESHOLD,
    1024, 2048, 4096, 8192];
  for (const size of sizes) {
    const values = lcgArray(size);
    const { total, stats } = await parallelSum(values);
    assert.strictEqual(total, sequentialSum(values), `parallel sum mismatch at size ${size}`);
    assert.strictEqual(stats.leaves, countLeaves(size), `leaf count at size ${size}`);
    assert.strictEqual(stats.spawned, stats.leaves - 1, `binary fork tree at size ${size}`);
  }

  // Documented threshold accounting for power-of-two sizes: leaves == n / THRESHOLD.
  for (const size of [512, 1024, 2048, 4096, 8192]) {
    const { total, stats } = await parallelSum(lcgArray(size));
    assert.strictEqual(total, sequentialSum(lcgArray(size)), `power-of-two sum at ${size}`);
    assert.strictEqual(stats.leaves, size / THRESHOLD, `power-of-two leaves at ${size}`);
    assert.strictEqual(stats.spawned, size / THRESHOLD - 1, `power-of-two forks at ${size}`);
  }

  // Splitting really happens above the threshold, and 8192 splits into exactly 31 forks.
  assert.strictEqual((await parallelSum(lcgArray(THRESHOLD))).stats.spawned, 0, "at threshold: no fork");
  assert.strictEqual((await parallelSum(lcgArray(THRESHOLD + 1))).stats.spawned, 1, "just above: one fork");
  assert.strictEqual((await parallelSum(lcgArray(8192))).stats.spawned, 31, "8192 -> 32 leaves, 31 forks");

  // LCG-driven sizes across the whole range, each cross-checked against the sequential sum.
  let seed = 9753;
  for (let trial = 0; trial < 20; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const size = seed % 3000;
    const values = lcgArray(size);
    const { total, stats } = await parallelSum(values);
    assert.strictEqual(total, sequentialSum(values), `LCG size ${size}`);
    assert.strictEqual(stats.leaves, countLeaves(size), `LCG leaves at ${size}`);
    assert.strictEqual(stats.spawned, stats.leaves - 1, `LCG fork tree at ${size}`);
  }
}

module.exports = { parallelSum, sequentialSum, countLeaves, THRESHOLD };

if (require.main === module) {
  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
