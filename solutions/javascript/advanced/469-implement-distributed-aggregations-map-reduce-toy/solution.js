/** Task 469 - Implement distributed aggregations map-reduce toy. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Deterministic word-count MapReduce with explicit phases:
 *   split   - three input splits of LCG-generated lines over a 40-word vocabulary;
 *   map     - each split emits (word, 1) pairs;
 *   partition - partition_of(word) = polynomial hash (h = (h*31 + ord(ch)) mod
 *             1000003) mod R with R = 4 reducers (documented rule);
 *   shuffle - pairs grouped by (partition, word), partitions and words in sorted order;
 *   reduce  - each (partition, word) group summed to one (word, count) pair.
 *
 * Checks: reduced counts equal a plain count of all words; every word's pairs land in
 * exactly one partition equal to its hash; per-partition pair totals are conserved;
 * every word is reduced exactly once; the pipeline is deterministic; and the
 * hand-computed partition of "w00" is 3 (h("w00") = 115895, mod 4 = 3).
 */
"use strict";

const REDUCERS = 4;

function makeLcg(state) {
  let seed = state;
  return () => (seed = (seed * 75 + 74) % 65537);
}

function makeSplits(startSeed) {
  const lcg = makeLcg(startSeed);
  const splits = [];
  for (let s = 0; s < 3; s++) {
    const words = [];
    for (let line = 0; line < 20; line++) {
      const length = (lcg() % 8) + 1;
      for (let k = 0; k < length; k++) words.push(`w${String(lcg() % 40).padStart(2, "0")}`);
    }
    splits.push(words);
  }
  return splits;
}

function partitionOf(word) {
  let h = 0;
  for (let i = 0; i < word.length; i++) h = (h * 31 + word.charCodeAt(i)) % 1000003;
  return h % REDUCERS;
}

function mapPhase(splits) {
  const pairs = [];
  for (let splitId = 0; splitId < splits.length; splitId++) {
    for (const word of splits[splitId]) pairs.push([splitId, word]);
  }
  return pairs;
}

function shufflePhase(pairs) {
  const buckets = new Map();
  for (const [, word] of pairs) {
    const part = partitionOf(word);
    if (!buckets.has(part)) buckets.set(part, new Map());
    const words = buckets.get(part);
    if (!words.has(word)) words.set(word, []);
    words.get(word).push(1);
  }
  return buckets;
}

function reducePhase(buckets) {
  const reduced = new Map();
  for (const part of [...buckets.keys()].sort((a, b) => a - b)) {
    const words = buckets.get(part);
    const out = new Map();
    for (const word of [...words.keys()].sort()) {
      out.set(
        word,
        words.get(word).reduce((a, b) => a + b, 0)
      );
    }
    reduced.set(part, out);
  }
  return reduced;
}

function runPipeline(startSeed) {
  return reducePhase(shufflePhase(mapPhase(makeSplits(startSeed))));
}

module.exports = { partitionOf, makeSplits, mapPhase, shufflePhase, reducePhase, runPipeline };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(partitionOf("w00"), 3); // hand-checked: h("w00") = 115895 -> 3
  assert.strictEqual(partitionOf("w"), 3); // hand-checked: h("w") = 119 -> 3

  const splits = makeSplits(20260116);
  const pairs = mapPhase(splits);
  const reduced = runPipeline(20260116);

  // Plain count oracle over all words.
  const counts = new Map();
  for (const split of splits) {
    for (const word of split) counts.set(word, (counts.get(word) || 0) + 1);
  }
  const flat = new Map();
  for (const words of reduced.values()) {
    for (const [word, total] of words) {
      assert.ok(!flat.has(word), `word reduced once: ${word}`);
      flat.set(word, total);
    }
  }
  assert.strictEqual(flat.size, counts.size);
  for (const [word, total] of counts) assert.strictEqual(flat.get(word), total);

  // Routing: every word sits in the partition of its hash.
  for (const [part, words] of reduced) {
    for (const word of words.keys()) assert.strictEqual(part, partitionOf(word));
  }

  // Conservation of pair totals per partition.
  const perPartitionIn = new Array(REDUCERS).fill(0);
  for (const [, word] of pairs) perPartitionIn[partitionOf(word)]++;
  const perPartitionOut = new Array(REDUCERS).fill(0);
  for (const [part, words] of reduced) {
    let total = 0;
    for (const count of words.values()) total += count;
    perPartitionOut[part] = total;
  }
  assert.deepStrictEqual(perPartitionIn, perPartitionOut);
  assert.strictEqual(
    perPartitionIn.reduce((a, b) => a + b, 0),
    pairs.length
  );

  // Determinism.
  const again = runPipeline(20260116);
  assert.strictEqual(again.size, reduced.size);
  for (const [part, words] of reduced) {
    const other = again.get(part);
    assert.deepStrictEqual([...words.entries()], [...other.entries()]);
  }
  console.log("PASS");
}
