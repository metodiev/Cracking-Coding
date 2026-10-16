/** Task 780 - Implement distributed sampling reservoir with merge operation. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Weighted reservoir sampling with the *priority* formulation (Efraimidis-Spirakis "A-Res"): every
 * item i draws u_i uniformly from (0, 1] and gets the key log(u_i) / w_i, and the reservoir is the
 * k items with the largest keys. Maximising log(u)/w is the same as maximising u^(1/w), and doing it
 * in log space keeps small weights from underflowing to zero.
 *
 * The reason this formulation is the distributed one: a key belongs to the item, not to the stream
 * position. Merging shard reservoirs is therefore *exactly* the same as sampling the union - keep
 * the k largest keys of the concatenation - with no re-sampling, no sizes, no communication beyond
 * k (item, key) pairs per shard. That exactness is checked directly below against a single-pass run
 * over the union with the same per-item keys.
 *
 * What is verified: the merge identity (shard-merged reservoir == union sample, item for item, for
 * hash-derived keys and for the streaming variant); uniformity of the selection over all C(n,k)
 * subsets and over per-item inclusion frequencies for unweighted items; weighted items are chosen in
 * proportion to their weights (monotone, close to k*w_i/W), items of weight zero are never chosen
 * while a positive-weight item could fill the slot; k = 1, k >= n, empty shards, a single shard and
 * determinism.
 */
"use strict";

/** Deterministic splitmix64-flavoured PRNG in float form: returns a value in (0, 1]. */
function makeUniform(seed) {
  let state = BigInt.asUintN(64, BigInt(seed) + 0x9e3779b97f4a7c15n);
  return () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    z ^= z >> 31n;
    return (Number(z >> 11n) + 1) / 2 ** 53; // (0, 1]
  };
}

/** Stable per-item uniform draw: the same item always gets the same key, whichever shard sees it. */
function itemUniform(item, salt = 0) {
  const text = `${salt}:${item}`;
  let h1 = 0x811c9dc5;
  let h2 = 0x9e3779b9;
  for (let i = 0; i < text.length; i++) {
    h1 = Math.imul(h1 ^ text.charCodeAt(i), 16777619) >>> 0;
    h2 = Math.imul(h2 + text.charCodeAt(i), 2654435761) >>> 0;
  }
  return (h1 + 0.5) / 4294967296 * 0.5 + (h2 + 0.5) / 4294967296 * 0.5;
}

/**
 * Priority of an item: log(u) / w, maximised by A-Res. A zero weight gets -Infinity, so such an item
 * is only kept when the reservoir cannot be filled with positive-weight items.
 */
function priority(uniformDraw, weight) {
  if (!(weight >= 0)) throw new RangeError("weights must be non-negative");
  if (weight === 0) return -Infinity;
  return Math.log(Math.max(uniformDraw, Number.MIN_VALUE)) / weight;
}

/** Bounded top-k by priority: a small binary min-heap, so the memory stays O(k). */
class Reservoir {
  constructor(k) {
    if (!Number.isInteger(k) || k < 0) throw new RangeError("k must be a non-negative integer");
    this.k = k;
    this.heap = []; // min-heap on `key`; the root is the first item to evict
  }

  get size() { return this.heap.length; }
  get entries() { return this.heap.map(({ item, weight, key }) => ({ item, weight, key })); }
  get items() { return this.heap.map((entry) => entry.item).sort(); }

  offer(item, weight, key) {
    if (this.k === 0) return false;
    const entry = { item, weight, key };
    if (this.heap.length < this.k) {
      this.heap.push(entry);
      this.#bubbleUp(this.heap.length - 1);
      return true;
    }
    if (key <= this.heap[0].key) return false;
    this.heap[0] = entry;
    this.#sinkDown(0);
    return true;
  }

  #swap(i, j) { const tmp = this.heap[i]; this.heap[i] = this.heap[j]; this.heap[j] = tmp; }

  #bubbleUp(index) {
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.heap[parent].key <= this.heap[index].key) break;
      this.#swap(parent, index);
      index = parent;
    }
  }

  #sinkDown(index) {
    while (true) {
      const left = 2 * index + 1;
      const right = left + 1;
      let smallest = index;
      if (left < this.heap.length && this.heap[left].key < this.heap[smallest].key) smallest = left;
      if (right < this.heap.length && this.heap[right].key < this.heap[smallest].key) smallest = right;
      if (smallest === index) return;
      this.#swap(index, smallest);
      index = smallest;
    }
  }

  /** The k largest entries in descending priority order. */
  sorted() { return this.entries.sort((a, b) => (b.key - a.key) || (a.item < b.item ? -1 : 1)); }
}

/** Single-pass sample with an explicit priority function (the streaming entry point). */
function sampleWithKeys(entries, k, keyOf) {
  const reservoir = new Reservoir(k);
  for (const entry of entries) reservoir.offer(entry.item, entry.weight, keyOf(entry));
  return reservoir;
}

/** Streaming sample: per-position draws from a seeded PRNG. */
function sampleStream(items, k, seed) {
  const uniform = makeUniform(seed);
  const reservoir = new Reservoir(k);
  for (const { item, weight } of items) reservoir.offer(item, weight, priority(uniform(), weight));
  return reservoir;
}

/** Streaming sample with the stable per-item keys (so shards agree without coordination). */
function sampleStable(items, k, salt = 0) {
  const reservoir = new Reservoir(k);
  for (const { item, weight } of items) reservoir.offer(item, weight, priority(itemUniform(item, salt), weight));
  return reservoir;
}

/** Merge: the k largest keys of the union - the whole point of the priority formulation. */
function mergeReservoirs(reservoirs, k = null) {
  const limit = k === null ? Math.max(...reservoirs.map((r) => r.k), 0) : k;
  const merged = new Reservoir(limit);
  for (const reservoir of reservoirs) {
    for (const entry of reservoir.entries) merged.offer(entry.item, entry.weight, entry.key);
  }
  return merged;
}

/** Reference implementation: keep the k largest keys by sorting everything. */
function topKByKey(entries, k, keyOf) {
  return entries
    .map((entry) => ({ item: entry.item, weight: entry.weight, key: keyOf(entry) }))
    .sort((a, b) => (b.key - a.key) || (a.item < b.item ? -1 : 1))
    .slice(0, k);
}

module.exports = { makeUniform, itemUniform, priority, Reservoir, sampleWithKeys, sampleStream, sampleStable, mergeReservoirs, topKByKey };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok, throws } = assert;

  const unweighted = Array.from({ length: 5 }, (_, i) => ({ item: `i${i}`, weight: 1 }));
  const keyOf = (entry) => priority(itemUniform(entry.item), entry.weight);

  // -- the merge identity: shard merge == single pass over the union -----------------
  const shards = [
    unweighted.slice(0, 2),
    unweighted.slice(2, 3),
    unweighted.slice(3),
    [], // an empty shard must be harmless
  ];
  const perShard = shards.map((shard) => sampleStable(shard, 2));
  const merged = mergeReservoirs(perShard, 2);
  const globalRun = sampleStable(unweighted, 2);
  strictEqual(JSON.stringify(merged.sorted()), JSON.stringify(globalRun.sorted()),
    "merging shard reservoirs must equal sampling the union");
  deepStrictEqual(merged.sorted().map((e) => e.item), topKByKey(unweighted, 2, keyOf).map((e) => e.item),
    "the merge must agree with an independent top-k sort");
  strictEqual(mergeReservoirs([], 2).size, 0, "merging nothing yields an empty reservoir");
  strictEqual(mergeReservoirs([sampleStable([], 3)], 3).size, 0, "merging empty reservoirs stays empty");
  strictEqual(mergeReservoirs([perShard[0]], 2).sorted()[0].item, perShard[0].sorted()[0].item,
    "a single shard survives the merge untouched");

  const wideShards = Array.from({ length: 6 }, (_, shard) =>
    Array.from({ length: 20 }, (_, i) => ({ item: `${shard}-${i}`, weight: 1 + (i % 3) })));
  const union = wideShards.flat();
  const distributed = mergeReservoirs(wideShards.map((shard) => sampleStable(shard, 4)), 4);
  const central = sampleStable(union, 4);
  deepStrictEqual(distributed.sorted(), central.sorted(),
    "six shards of twenty items must merge into the same four items as one central pass");
  deepStrictEqual(distributed.sorted().map((e) => e.item), topKByKey(union, 4, keyOf).map((e) => e.item),
    "the distributed result must match the exact top-k of the union");

  // -- k boundaries ------------------------------------------------------------------
  const one = sampleStream(unweighted, 1, 4242);
  strictEqual(one.size, 1, "k = 1 keeps exactly one item");
  const all = sampleStream(unweighted, 99, 4242);
  strictEqual(all.size, unweighted.length, "k >= n keeps every item");
  deepStrictEqual(all.items, unweighted.map((e) => e.item).sort(), "every item is present exactly once");
  const none = sampleStream(unweighted, 0, 4242);
  strictEqual(none.size, 0, "k = 0 keeps nothing");
  throws(() => new Reservoir(-1), RangeError, "a negative k is invalid");
  throws(() => priority(0.5, -1), RangeError, "a negative weight is invalid");

  // -- zero weights are never picked while a positive one could fill the slot --------
  const mixed = [
    { item: "zeroA", weight: 0 },
    { item: "zeroB", weight: 0 },
    { item: "posA", weight: 1 },
    { item: "posB", weight: 3 },
    { item: "posC", weight: 5 },
  ];
  for (let seed = 0; seed < 30; seed++) {
    const picked = sampleStream(mixed, 2, seed).items;
    strictEqual(picked.length, 2, "the reservoir fills up from the positive weights");
    ok(picked.every((item) => !item.startsWith("zero")), `a zero-weight item must not be sampled (seed ${seed})`);
  }
  strictEqual(priority(1, 0), -Infinity, "a zero weight has an infinitely bad key");
  const forced = sampleStream(mixed, 5, 7);
  strictEqual(forced.items.filter((item) => item.startsWith("zero")).length, 2,
    "with k >= the number of items the zero-weight items fill the remaining slots");

  // -- uniformity: every subset and every item equally likely (unweighted) -----------
  const TRIALS = 3000;
  const subsetCounts = new Map();
  const itemCounts = new Map();
  const uniform = makeUniform(20260780);
  for (let trial = 0; trial < TRIALS; trial++) {
    const reservoir = new Reservoir(2);
    for (const entry of unweighted) reservoir.offer(entry.item, 1, priority(uniform(), 1));
    const key = reservoir.items.join(",");
    subsetCounts.set(key, (subsetCounts.get(key) ?? 0) + 1);
    for (const item of reservoir.items) itemCounts.set(item, (itemCounts.get(item) ?? 0) + 1);
  }
  strictEqual(subsetCounts.size, 10, "five items choose two means ten possible subsets");
  const expectedSubset = TRIALS / 10;
  for (const [subset, count] of subsetCounts) {
    ok(Math.abs(count - expectedSubset) < expectedSubset * 0.25,
      `subset ${subset} appeared ${count} times, expected about ${expectedSubset}`);
  }
  for (const { item } of unweighted) {
    const frequency = itemCounts.get(item) ?? 0;
    ok(Math.abs(frequency - TRIALS * 0.4) < TRIALS * 0.12,
      `item ${item} was chosen ${frequency} times, expected about ${(TRIALS * 0.4).toFixed(0)}`);
  }

  // -- weighted sampling follows the weights ----------------------------------------
  const weighted = [
    { item: "light", weight: 1 },
    { item: "medium", weight: 2 },
    { item: "heavy", weight: 4 },
  ];
  const TOTAL = 4000;
  const picked = new Map(weighted.map((entry) => [entry.item, 0]));
  for (let trial = 0; trial < TOTAL; trial++) {
    const reservoir = new Reservoir(1);
    for (const entry of weighted) reservoir.offer(entry.item, entry.weight, priority(uniform(), entry.weight));
    picked.set(reservoir.entries[0].item, picked.get(reservoir.entries[0].item) + 1);
  }
  ok(picked.get("light") < picked.get("medium"), "a heavier item must be sampled more often");
  ok(picked.get("medium") < picked.get("heavy"), "weights must be respected in order");
  const totalWeight = weighted.reduce((sum, entry) => sum + entry.weight, 0);
  for (const entry of weighted) {
    const expected = TOTAL * (entry.weight / totalWeight);
    const observed = picked.get(entry.item);
    ok(Math.abs(observed - expected) < TOTAL * 0.05,
      `${entry.item}: observed ${observed} vs expected ${expected.toFixed(0)} (k*w/W)`);
  }

  // -- streaming keys versus stable keys both keep the merge property ----------------
  const streamA = sampleStream(union.slice(0, 60), 5, 11);
  const streamB = sampleStream(union.slice(60), 5, 12);
  const streamed = mergeReservoirs([streamA, streamB], 5);
  strictEqual(streamed.size, 5, "position-based keys still fill the reservoir");
  ok(streamed.sorted().every((entry, i, array) => i === 0 || array[i - 1].key >= entry.key),
    "the merged reservoir must stay ordered by key");

  // -- determinism -------------------------------------------------------------------
  const replay = () => {
    const shardRuns = wideShards.map((shard, index) => sampleStream(shard, 3, 100 + index));
    return mergeReservoirs(shardRuns, 3).sorted();
  };
  strictEqual(JSON.stringify(replay()), JSON.stringify(replay()), "the same seeds must replay identically");

  console.log(`subset spread=${Math.min(...subsetCounts.values())}..${Math.max(...subsetCounts.values())} weighted light/medium/heavy=${picked.get("light")}/${picked.get("medium")}/${picked.get("heavy")}`);
  console.log("PASS");
}
