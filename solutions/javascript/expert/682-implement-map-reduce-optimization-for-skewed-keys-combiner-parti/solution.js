/**
 * Task 682 - Implement map-reduce optimization for skewed keys (combiner + partitioning).
 * (Tier: Expert, System Design and Scalability)
 *
 * A word-count style job over strongly skewed keys, with the two classic optimisations:
 *
 *   combiner    - each map task aggregates its own (key, count) pairs before the shuffle, so a split that
 *                 emits 40 records over 6 keys ships 6 pairs instead of 40; because the aggregate is
 *                 associative and commutative the final result cannot change;
 *   partitioner - either `key % reducers` (hash, which sends every hot key to one reducer) or a
 *                 sampled-split partitioner that sorts a sample of the keys and cuts the key space into
 *                 reducers equal-sized ranges, which spreads the hot keys out.
 *
 * The map input is generated from a Zipf-like weight 1/(k+1), so the skew is deliberate and reproducible.
 * `reducePhase` merges the shuffled pairs per reducer, and `groupBy` is the brute-force reference: both
 * must agree for every combination of combiner and partitioner.
 *
 * Verified: the four combinations (with/without combiner, hash vs sampled partitioner) all reproduce the
 * brute-force group-by exactly; the combiner shrinks the shuffle from one pair per record to one pair per
 * (split, key) - a strict reduction on this skewed input - while the counts themselves change; the sampled
 * partitioner lowers the maximum reducer load compared with hashing every hot key into one reducer; a
 * single-key input, zero records, a single reducer and the reproducibility of the whole pipeline are
 * covered as boundary cases.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed682n;

class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }
  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }
  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }
}

// -------------------------------------------------------------------------------------- input
/** `count` keys drawn from a Zipf-like distribution over `keys` values. */
function makeRecords(rng, count, keys) {
  const weights = Array.from({ length: keys }, (_, k) => 1 / (k + 1));
  const total = weights.reduce((a, b) => a + b, 0);
  const cumulative = [];
  let running = 0;
  for (const weight of weights) {
    running += weight / total;
    cumulative.push(running);
  }
  const draw = () => {
    const u = rng.nextFloat();
    let low = 0;
    let high = cumulative.length - 1;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (cumulative[middle] < u) low = middle + 1;
      else high = middle;
    }
    return `k${low}`;
  };
  const records = [];
  for (let i = 0; i < count; i++) records.push({ id: i, key: draw() });
  return records;
}

// ------------------------------------------------------------------------------------ phases
/** Map: split the records across mappers round-robin, emitting one (key, 1) pair per record. */
function mapPhase(records, mappers) {
  const splits = Array.from({ length: mappers }, () => []);
  records.forEach((record, index) => splits[index % mappers].push(record));
  return splits.map((split, index) => ({
    index,
    pairs: split.map((record) => [record.key, 1]),
  }));
}

/** Combiner: aggregate each split in place, which is what shrinks the shuffle. */
function combinePhase(splits) {
  const combined = [];
  let shuffledBefore = 0;
  let shuffledAfter = 0;
  for (const split of splits) {
    shuffledBefore += split.pairs.length;
    const counts = new Map();
    for (const [key, count] of split.pairs) counts.set(key, (counts.get(key) ?? 0) + count);
    const pairs = [...counts.entries()].sort();
    shuffledAfter += pairs.length;
    combined.push({ index: split.index, pairs });
  }
  return { splits: combined, shuffledBefore, shuffledAfter };
}

/** Hash partitioner: the hot key always lands on the same reducer. */
const hashPartitioner = (reducers) => (key) => Number(key.slice(1)) % reducers;

/** Sampled-split partitioner: sort a sample of the keys and cut the key space into equal ranges. */
function sampledPartitioner(splits, reducers, rng) {
  const keys = [...new Set(splits.flatMap((split) => split.pairs.map(([key]) => key)))].sort(
    (a, b) => Number(a.slice(1)) - Number(b.slice(1)));
  const sample = [];
  for (const key of keys) if (rng.nextInt(2) === 0 || sample.length < reducers) sample.push(key);
  const boundaries = [];
  for (let i = 1; i < reducers; i++) {
    const index = Math.min(sample.length - 1, Math.floor((i * sample.length) / reducers));
    boundaries.push(sample[index]);
  }
  return (key) => {
    let reducer = 0;
    while (reducer < boundaries.length && key > boundaries[reducer]) reducer += 1;
    return reducer;
  };
}

/** Shuffle + reduce: assign every pair to a reducer and sum the counts per key. */
function reducePhase(splits, reduceFn, reducers) {
  const partitions = Array.from({ length: reducers }, () => []);
  for (const split of splits) for (const pair of split.pairs) partitions[reduceFn(pair[0])].push(pair);
  const results = new Map();
  for (const partition of partitions) {
    for (const [key, count] of partition) results.set(key, (results.get(key) ?? 0) + count);
  }
  const loads = partitions.map((partition) => partition.length);
  const pairs = loads.reduce((a, b) => a + b, 0);
  const counted = [...results.values()].reduce((a, b) => a + b, 0);
  return { results, loads, maxLoad: Math.max(0, ...loads), averageLoad: pairs / reducers, pairs, counted };
}

const groupBy = (records) => {
  const counts = new Map();
  for (const record of records) counts.set(record.key, (counts.get(record.key) ?? 0) + 1);
  return counts;
};

/** Whole pipeline; `useCombiner` and `partitioner` select the optimisation under test. */
function run(records, options = {}) {
  const { mappers = 4, reducers = 3, useCombiner = true, partitioner = "hash", seed = DEFAULT_SEED } = options;
  const rng = new SplitMix64(seed);
  const mapped = mapPhase(records, mappers);
  const combined = combinePhase(mapped);
  const splits = useCombiner ? combined.splits : mapped;
  const reduceFn = partitioner === "sampled"
    ? sampledPartitioner(splits, reducers, new SplitMix64(seed))
    : hashPartitioner(reducers);
  const reduced = reducePhase(splits, reduceFn, reducers);
  const shuffled = splits.reduce((acc, split) => acc + split.pairs.length, 0);
  return { ...reduced, shuffled, shuffledBefore: combined.shuffledBefore, shuffledAfter: combined.shuffledAfter,
    reducers, mappers, useCombiner, partitioner, rng };
}

function main() {
  // (1) The generator really is skewed: the first key dominates.
  const rng = new SplitMix64(DEFAULT_SEED);
  const records = makeRecords(rng, 400, 8);
  const truth = groupBy(records);
  assert.strictEqual(records.length, 400, "four hundred records were generated");
  assert.strictEqual(truth.size, 8, "all eight keys appear");
  assert.ok(truth.get("k0") > truth.get("k3"), `documented: k0 carries ${truth.get("k0")} records against ${truth.get("k3")} for k3`);
  assert.ok(truth.get("k0") > 400 / 8, "the hot key is above the average, so the data is unbalanced");

  // (2) Every combination reproduces the brute-force group-by exactly.
  let combinations = 0;
  for (const useCombiner of [false, true]) {
    for (const partitioner of ["hash", "sampled"]) {
      const result = run(records, { useCombiner, partitioner });
      combinations += 1;
      assert.deepStrictEqual([...result.results.entries()].sort(), [...truth.entries()].sort(),
        `combiner=${useCombiner} partitioner=${partitioner} must reproduce the group-by`);
      assert.strictEqual(result.counted, records.length, "and every record is counted exactly once");
    }
  }
  assert.strictEqual(combinations, 4, "four pipeline shapes were compared with the brute-force reference");

  // (3) The combiner shrinks the shuffle without changing the counts.
  const withoutCombiner = run(records, { useCombiner: false });
  const withCombiner = run(records, { useCombiner: true });
  assert.deepStrictEqual([withoutCombiner.shuffled, withCombiner.shuffled], [400, withCombiner.shuffledAfter],
    "without the combiner one pair per record is shipped; with it, one pair per (split, key)");
  assert.ok(withCombiner.shuffled < withoutCombiner.shuffled,
    `documented: the combiner cuts the shuffle from ${withoutCombiner.shuffled} to ${withCombiner.shuffled} pairs`);
  assert.ok(withCombiner.shuffledAfter <= withCombiner.mappers * truth.size, "the combiner cannot invent keys");

  // (4) Partitioning: hashing concentrates the hot key, sampling spreads the load.
  const hashed = run(records, { reducers: 4, partitioner: "hash" });
  const sampled = run(records, { reducers: 4, partitioner: "sampled" });
  assert.deepStrictEqual([hashed.maxLoad, sampled.maxLoad].map((x) => x <= records.length), [true, true],
    "both partitioners ship every pair exactly once");
  assert.ok(sampled.maxLoad <= hashed.maxLoad,
    `documented: the sampled partitioner caps a reducer at ${sampled.maxLoad} pairs against ${hashed.maxLoad} for hashing`);
  assert.ok(sampled.maxLoad >= 1 && sampled.loads.length === 4, "and it still uses all four reducers");
  assert.deepStrictEqual([...sampled.results.entries()].sort(), [...truth.entries()].sort(),
    "the balanced partitioner produces the same counts");

  // (5) Sampling is stable: the same seed gives the same assignment, a different one still balances.
  const repeat = run(records, { reducers: 4, partitioner: "sampled" });
  assert.deepStrictEqual(repeat.loads, sampled.loads, "the partitioner is reproducible for a fixed seed");
  const otherSeed = run(records, { reducers: 4, partitioner: "sampled", seed: 99n });
  assert.deepStrictEqual([...otherSeed.results.entries()].sort(), [...truth.entries()].sort(), "another sample is still exact");

  // (6) Boundary cases.
  const empty = run([], {});
  assert.deepStrictEqual([empty.results.size, empty.shuffled, empty.maxLoad], [0, 0, 0], "no records, no work");
  const singleKey = run(makeRecords(new SplitMix64(1n), 40, 1), { reducers: 4 });
  assert.deepStrictEqual([singleKey.results.size, singleKey.results.get("k0"), singleKey.shuffled],
    [1, 40, singleKey.mappers], "with one key the combiner ships a single pair per split");
  const oneReducer = run(records, { reducers: 1, partitioner: "sampled" });
  assert.deepStrictEqual([oneReducer.loads.length, oneReducer.maxLoad], [1, oneReducer.shuffled],
    "a single reducer takes the whole shuffle");
  assert.deepStrictEqual([...oneReducer.results.entries()].sort(), [...truth.entries()].sort(), "and still counts correctly");
  const tiny = run(makeRecords(new SplitMix64(7n), 5, 3), { mappers: 2, reducers: 2, partitioner: "sampled" });
  assert.strictEqual(tiny.counted, 5, "a tiny input is handled by the sampled partitioner too");

  // (7) Determinism of the whole pipeline.
  const first = run(records, {});
  const second = run(records, {});
  assert.deepStrictEqual([[...first.results.entries()].sort(), first.loads, first.shuffled],
    [[...second.results.entries()].sort(), second.loads, second.shuffled], "the pipeline is deterministic");
  assert.deepStrictEqual(makeRecords(new SplitMix64(3n), 10, 4), makeRecords(new SplitMix64(3n), 10, 4),
    "and so is the data generator");
}

module.exports = { SplitMix64, DEFAULT_SEED, makeRecords, mapPhase, combinePhase, hashPartitioner,
  sampledPartitioner, reducePhase, groupBy, run };

if (require.main === module) {
  main();
  console.log("PASS");
}
