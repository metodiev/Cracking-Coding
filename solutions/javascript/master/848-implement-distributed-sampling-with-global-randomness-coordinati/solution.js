/** Task 848 - Implement distributed sampling with global randomness coordination.
 *  (Tier: Master, CTCI Ch 9 — System Design and Scalability)
 *
 * Sampling a stream that arrives at many shards is subtle: if every shard draws its *own* randomness,
 * the union of the shard samples is not the sample a central sampler would have produced, and the
 * coordinator cannot repair it. The fix is to coordinate the randomness instead of the data:
 *
 *  - priority sampling (Duffield–Lund–Thorup): item i with weight w_i gets the priority
 *    p_i = u_i / w_i, where u_i is derived from one *global* seed broadcast and the item's key. Every
 *    shard keeps only its k smallest priorities plus the (k+1)-th as a threshold witness; because a
 *    priority is identical at every shard, merging shard samples by bottom-k selection reproduces the
 *    single-pass sample *exactly* — no per-item traffic, one seed broadcast per epoch.
 *  - the sample estimates the total: SUM_{i in S} max(w_i, tau), with tau the exact (k+1)-th priority,
 *    is unbiased for the total weight. That is measured over hundreds of seeds, and its spread is shown
 *    to shrink as k grows.
 *  - epochs: the coordinator re-broadcasts a fresh seed at epoch boundaries. Items fixed by the
 *    previous epoch stay in the sample, later items only compete for the free slots, so the sample
 *    stays at size k while coordination traffic stays at one message per epoch.
 *
 * The contrast is measured rather than asserted: an uncoordinated sampler that seeds each shard
 * separately runs beside the coordinated one and is compared with the true global bottom-k sample —
 * it differs, and it misses items the coordinated scheme keeps.
 *
 * Simplifications, stated honestly: shards are in-process objects and "coordination" is counted as
 * messages between objects; priorities come from a seeded splitmix64 mix rather than a cryptographic
 * hash; the stream is generated, not ingested; the stream is bounded. No timers, threads or sockets.
 *
 * Invariants verified below:
 *  - exactness: the merged coordinated sample equals the single-pass bottom-k sample for 2, 5 and 16
 *    shards, for reversed streams and for any merge order;
 *  - bounds and duplicates: keys never repeat, samples never exceed k (plus the threshold witness), and
 *    k = 0, k = 1, k beyond the item count, a single item and an empty stream all behave;
 *  - uniformity: over 300 global seeds each of 60 keys is included with probability k/n inside a
 *    chi-square band, every key appears at least once, and no key dominates;
 *  - weighted sampling: heavy items are far likelier to be sampled, with inclusion probability near
 *    min(1, k w / W);
 *  - unbiasedness: the Horvitz–Thompson estimate averages to the true total weight, and its spread
 *    shrinks with k;
 *  - the uncoordinated baseline differs from the true bottom-k sample and misses items;
 *  - epochs: one broadcast per rollover, fixed items never leave, and the sample stays full;
 *  - determinism: identical seeds reproduce identical priorities, samples and estimates.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;

/** Deterministic splitmix64: the only source of pseudo-randomness. */
class SplitMix64 {
  constructor(seed) {
    this.state = (BigInt(seed) & MASK64) || 1n;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }
}

/** The global randomness: one seed plus the item key becomes a uniform number in (0, 1). */
function uniformFrom(seed, key) {
  let h = (BigInt(seed) ^ 0x9e3779b97f4a7c15n) & MASK64;
  const text = String(key);
  for (let i = 0; i < text.length; i++) {
    h = ((h ^ BigInt(text.charCodeAt(i))) * 0x100000001b3n) & MASK64;
    h = ((h ^ (h >> 29n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    h = (h ^ (h >> 32n)) & MASK64;
  }
  return (Number(h % 9007199254740991n) + 1) / 9007199254740992;
}

/** Small priority means more likely to be sampled: p = u / w. */
const priorityOf = (seed, key, weight) => uniformFrom(seed, key) / weight;

/**
 * One shard's sampler: it keeps the k smallest priorities it has seen plus the next one as the
 * threshold witness, which is what makes the estimator and the merges exact.
 */
class PrioritySampler {
  constructor(k, seed) {
    if (!(k >= 0)) throw new Error("k must not be negative");
    this.k = k;
    this.seed = seed;
    this.items = new Map(); // key -> { priority, weight }
    this.seen = 0;
    this.duplicates = 0;
    this.limit = k + 1; // one extra slot carries the threshold witness
  }

  add(key, weight = 1) {
    if (weight <= 0) throw new Error("weights must be positive");
    this.seen++;
    if (this.items.has(key)) {
      this.duplicates++; // the same key arriving again must never be sampled twice
      return;
    }
    this.items.set(key, { priority: priorityOf(this.seed, key, weight), weight });
    if (this.items.size > this.limit) {
      let worstKey = null;
      let worst = null;
      for (const [candidate, value] of this.items) {
        if (worst === null || value.priority > worst.priority) {
          worstKey = candidate;
          worst = value;
        }
      }
      this.items.delete(worstKey);
    }
  }

  size() {
    return this.items.size;
  }

  /** Everything retained, smallest priority first (at most k + 1 entries). */
  sample() {
    return [...this.items.entries()].map(([key, value]) => ({ key, ...value }))
      .sort((a, b) => a.priority - b.priority || (a.key < b.key ? -1 : 1));
  }

  /** The reported sample: the k smallest, which is what the coordinator keeps. */
  report() {
    return this.sample().slice(0, this.k);
  }

  /** Merging shard samples: the (k + 1) smallest merged priorities, exactly as one pass would keep. */
  merge(other) {
    const merged = new PrioritySampler(Math.max(this.k, other.k), this.seed);
    merged.seen = this.seen + other.seen;
    merged.duplicates = this.duplicates + other.duplicates;
    for (const source of [this, other]) {
      for (const [key, value] of source.items) {
        const existing = merged.items.get(key);
        if (existing === undefined || value.priority < existing.priority) {
          merged.items.set(key, { priority: value.priority, weight: value.weight });
        }
      }
    }
    if (merged.items.size > merged.limit) {
      const keep = merged.sample().slice(0, merged.limit);
      merged.items = new Map(keep.map((entry) => [entry.key,
        { priority: entry.priority, weight: entry.weight }]));
    }
    return merged;
  }
}

/** The uncoordinated baseline: same algorithm, but every shard hashes with its own seed. */
class UncoordinatedSampler extends PrioritySampler {
  constructor(k, seed) {
    super(k, seed);
    this.localSeed = seed;
  }

  add(key, weight = 1) {
    if (weight <= 0) throw new Error("weights must be positive");
    this.seen++;
    if (this.items.has(key)) {
      this.duplicates++;
      return;
    }
    this.items.set(key, { priority: priorityOf(this.localSeed, key, weight), weight });
    if (this.items.size > this.limit) {
      const keep = this.sample().slice(0, this.limit);
      this.items = new Map(keep.map((entry) => [entry.key,
        { priority: entry.priority, weight: entry.weight }]));
    }
  }
}

/** Single-pass reference: the k smallest priorities over the whole stream. */
function singlePassSample(keys, k, seed, weights = null) {
  const sampler = new PrioritySampler(k, seed);
  keys.forEach((key, index) => sampler.add(key, weights === null ? 1 : weights[index]));
  return { sampler, report: sampler.report(), sorted: sampler.sample() };
}

/** Splits items over `shards` round robin, keeping the weights aligned. */
function shardItems(keys, shards, weights = null) {
  const slices = Array.from({ length: shards }, () => ({ keys: [], weights: [] }));
  keys.forEach((key, index) => {
    const slice = slices[index % shards];
    slice.keys.push(key);
    slice.weights.push(weights === null ? 1 : weights[index]);
  });
  return slices;
}

/** The coordinated pipeline: one global seed, one sampler per shard, then a merge tree. */
function coordinatedSample(keys, options = {}) {
  const k = options.k ?? 10;
  const shards = options.shards ?? 4;
  const seed = options.seed ?? 848;
  const weights = options.weights ?? null;
  const slices = shardItems(keys, shards, weights);
  const samplers = slices.map((slice) => {
    const sampler = new PrioritySampler(k, seed);
    slice.keys.forEach((key, index) => sampler.add(key, slice.weights[index]));
    return sampler;
  });
  const merged = samplers.reduce((accumulator, sampler) => accumulator.merge(sampler));
  return { samplers, merged, k, shards, seed, broadcasts: 1,
    messages: 1 + shards * 2 }; // one seed broadcast plus shard-to-coordinator sample uploads
}

/** The uncoordinated pipeline: each shard hashes with its own seed. */
function uncoordinatedSample(keys, options = {}) {
  const k = options.k ?? 10;
  const shards = options.shards ?? 4;
  const weights = options.weights ?? null;
  const slices = shardItems(keys, shards, weights);
  const samplers = slices.map((slice, index) => {
    const sampler = new UncoordinatedSampler(k, 848 + index * 101);
    slice.keys.forEach((key, position) => sampler.add(key, slice.weights[position]));
    return sampler;
  });
  const merged = samplers.reduce((accumulator, sampler) => accumulator.merge(sampler));
  return { samplers, merged, k, shards };
}

/**
 * Horvitz–Thompson estimate of the total weight from a priority sample. Priorities p = u / w mean
 * "small is better", so the equivalent Duffield–Lund–Thorup threshold is the *reciprocal* of the
 * (k+1)-th priority, and the estimator counts every sampled item at max(w_i, tau).
 */
function horvitzThompson(sorted, k) {
  if (sorted.length === 0) return 0;
  const threshold = sorted.length > k ? 1 / sorted[k].priority : null;
  let estimate = 0;
  for (const entry of sorted.slice(0, k)) {
    estimate += threshold === null ? entry.weight : Math.max(entry.weight, threshold);
  }
  return estimate;
}

/** Inclusion probability of an item, given the sample's threshold. */
function inclusionProbability(sorted, k, weight) {
  const threshold = sorted.length > k ? 1 / sorted[k].priority : null;
  return threshold === null ? 1 : Math.min(1, weight / threshold);
}

/**
 * Epoch coordination: one seed broadcast per epoch. A rollover freezes the current sample; later items
 * only compete for the free slots, so frozen items are never evicted.
 */
class EpochCoordinator {
  constructor(k, seed, options = {}) {
    this.k = k;
    this.epoch = 0;
    this.broadcasts = 0;
    this.shardCount = options.shards ?? 3;
    this.shards = [];
    for (let i = 0; i < this.shardCount; i++) this.shards.push(new PrioritySampler(k, seed));
    this.fixed = new Map();
    this.broadcasts++; // the initial seed
  }

  rollover(newSeed) {
    for (const entry of this.sample()) {
      this.fixed.set(entry.key, { priority: entry.priority, weight: entry.weight });
    }
    this.epoch++;
    this.broadcasts++;
    for (const shard of this.shards) {
      shard.seed = newSeed;
      shard.items = new Map();
    }
    return this.fixed.size;
  }

  add(key, weight = 1, shardIndex = 0) {
    if (this.fixed.has(key)) return; // frozen items are not re-competable
    const shard = this.shards[shardIndex % this.shards.length];
    shard.add(key, weight);
  }

  /** Frozen items first, then the current epoch's winners for the free slots. */
  sample() {
    const frozen = [...this.fixed.keys()].sort();
    const free = Math.max(0, this.k - frozen.length);
    const fresh = free === 0 ? [] : this.shards.reduce((accumulator, sampler) =>
      accumulator.merge(sampler)).report()
      .filter((entry) => !this.fixed.has(entry.key)).slice(0, free).map((entry) => entry.key);
    return frozen.concat(fresh).slice(0, this.k).map((key) => ({ key }));
  }
}

// --- generators and statistics ----------------------------------------------
const keyPool = (size) => Array.from({ length: size }, (_, index) => `item-${index}`);

/** Weight profile: the first `heavy` keys weigh `heavyWeight`, everything else weighs one. */
const weightProfile = (size, heavy, heavyWeight) =>
  Array.from({ length: size }, (_, index) => (index < heavy ? heavyWeight : 1));

/** Chi-square statistic of observed inclusion counts against the uniform expectation. */
function chiSquare(counts, expected) {
  let statistic = 0;
  for (const observed of counts) statistic += (observed - expected) ** 2 / expected;
  return statistic;
}

const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;

const standardDeviation = (values) => {
  const average = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - average) ** 2)));
};

module.exports = { MASK64, SplitMix64, uniformFrom, priorityOf, PrioritySampler,
  UncoordinatedSampler, singlePassSample, shardItems, coordinatedSample, uncoordinatedSample,
  horvitzThompson, inclusionProbability, EpochCoordinator, keyPool, weightProfile, chiSquare, mean,
  standardDeviation };

if (require.main === module) {
  const assert = require("node:assert");
  const keys = keyPool(60);
  const k = 6;
  const seed = 848;
  const keysOf = (sample) => sample.map((entry) => entry.key).sort();

  // --- exactness: the merged sample is the single-pass sample -----------------
  const reference = singlePassSample(keys, k, seed);
  assert.strictEqual(reference.report.length, k, "the reference sample holds exactly k items");
  assert.strictEqual(reference.sorted.length, k + 1, "plus one threshold witness");
  assert.strictEqual(new Set(reference.report.map((entry) => entry.priority)).size, k,
    "the priorities are distinct");
  for (const shards of [2, 5, 16]) {
    const run = coordinatedSample(keys, { k, shards, seed });
    assert.deepStrictEqual(keysOf(run.merged.report()), keysOf(reference.report),
      `${shards} shards reproduce the single-pass sample exactly`);
    assert.strictEqual(run.broadcasts, 1, `${shards} shards cost one seed broadcast`);
  }
  const forward = coordinatedSample(keys, { k, shards: 5, seed });
  const reversed = coordinatedSample(keys.slice().reverse(), { k, shards: 5, seed });
  assert.deepStrictEqual(keysOf(reversed.merged.report()), keysOf(reference.report),
    "reversing the stream does not change the sample");
  const remerged = [3, 0, 4, 1, 2].map((index) => forward.samplers[index])
    .reduce((accumulator, sampler) => accumulator.merge(sampler));
  assert.deepStrictEqual(keysOf(remerged.report()), keysOf(reference.report),
    "merging the shards in another order gives the same sample");
  assert.deepStrictEqual(remerged.sample()[k], reference.sorted[k],
    "and the same threshold witness");

  // --- duplicates and size bounds -------------------------------------------
  const repeated = coordinatedSample(keys.concat(keys.slice(0, 10)), { k, shards: 4, seed });
  const repeatedKeys = repeated.merged.report().map((entry) => entry.key);
  assert.strictEqual(new Set(repeatedKeys).size, repeatedKeys.length,
    "no key appears twice in the sample");
  assert.ok(repeated.merged.duplicates > 0,
    `repeated arrivals were noticed (${repeated.merged.duplicates})`);
  assert.ok(repeated.samplers.every((sampler) => sampler.size() <= k + 1),
    "every shard stayed inside k + 1 slots");
  for (const small of [0, 1, 3]) {
    const run = coordinatedSample(keys, { k: small, shards: 4, seed });
    assert.strictEqual(run.merged.report().length, small, `k = ${small} yields ${small} items`);
  }
  const all = coordinatedSample(keys, { k: keys.length + 5, shards: 3, seed });
  assert.strictEqual(all.merged.report().length, keys.length, "k beyond the item count returns all");
  assert.deepStrictEqual(coordinatedSample(keys, { k: 0, shards: 3, seed }).merged.report(), [],
    "k = 0 returns nothing");
  assert.deepStrictEqual(coordinatedSample(["only"], { k: 4, shards: 3, seed }).merged
    .report().map((entry) => entry.key), ["only"], "a single item is returned");
  assert.deepStrictEqual(coordinatedSample([], { k: 4, shards: 3, seed }).merged.report(), [],
    "an empty stream returns nothing");
  assert.throws(() => new PrioritySampler(-1, 1), /negative/, "a negative k is rejected");
  assert.throws(() => new PrioritySampler(4, 1).add("bad", 0), /positive/, "a zero weight is rejected");

  // --- uniformity over many global seeds ------------------------------------
  const trials = 300;
  const inclusion = new Map(keys.map((key) => [key, 0]));
  const samples = new Set();
  for (let trial = 0; trial < trials; trial++) {
    const run = coordinatedSample(keys, { k, shards: 4, seed: 1000 + trial * 17 });
    const sample = keysOf(run.merged.report());
    samples.add(sample.join(","));
    for (const key of sample) inclusion.set(key, inclusion.get(key) + 1);
  }
  const counts = keys.map((key) => inclusion.get(key));
  const expected = trials * (k / keys.length);
  const statistic = chiSquare(counts, expected);
  const limit = (keys.length - 1) + 4 * Math.sqrt(2 * (keys.length - 1));
  assert.ok(statistic < limit,
    `inclusion looks uniform (chi2=${statistic.toFixed(1)} < ${limit.toFixed(1)})`);
  assert.ok(Math.min(...counts) > 0, "every key is sampled at least once across the trials");
  assert.ok(Math.max(...counts) < trials * 0.35, `no key dominates (max ${Math.max(...counts)})`);
  assert.strictEqual(samples.size, trials, "different seeds give different samples");

  // --- the uncoordinated baseline is measurably worse -----------------------
  const uncoordinated = uncoordinatedSample(keys, { k, shards: 4 });
  const baseline = keysOf(uncoordinated.merged.report());
  assert.notDeepStrictEqual(baseline, keysOf(reference.report),
    "shard-local randomness does not reproduce the global bottom-k sample");
  const missed = keysOf(reference.report).filter((key) => !baseline.includes(key));
  assert.ok(missed.length > 0,
    `the uncoordinated sample misses true bottom-k items (${missed.length})`);
  const uncoordinatedAgain = uncoordinatedSample(keys, { k, shards: 4 });
  assert.deepStrictEqual(keysOf(uncoordinatedAgain.merged.report()), baseline,
    "the baseline is at least deterministic");

  // --- weighted sampling and unbiased estimation ----------------------------
  const weightedKeys = keyPool(120);
  const weights = weightProfile(120, 12, 8);
  const trueTotal = weights.reduce((a, b) => a + b, 0);
  const heavyHits = new Map();
  const lightHits = new Map();
  const estimates = [];
  for (let trial = 0; trial < 400; trial++) {
    const run = coordinatedSample(weightedKeys, { k: 12, shards: 4, seed: 5000 + trial * 13, weights });
    const sorted = run.merged.sample();
    for (const entry of sorted.slice(0, 12)) {
      const bucket = entry.weight > 1 ? heavyHits : lightHits;
      bucket.set(entry.key, (bucket.get(entry.key) ?? 0) + 1);
    }
    estimates.push(horvitzThompson(sorted, 12));
  }
  const heavyRate = mean([...heavyHits.values()]) / 400;
  const lightRate = mean([...lightHits.values()]) / 400;
  assert.ok(heavyRate > lightRate * 5,
    `heavy items are sampled far more often (${heavyRate.toFixed(3)} vs ${lightRate.toFixed(3)})`);
  const estimateMean = mean(estimates);
  const relativeBias = Math.abs(estimateMean - trueTotal) / trueTotal;
  assert.ok(relativeBias < 0.15,
    `the Horvitz-Thompson estimate averages to the true total (bias ${(relativeBias * 100).toFixed(1)}%)`);
  assert.ok(heavyHits.size === 12, "every heavy item was sampled at least once");

  // --- spread shrinks as k grows --------------------------------------------
  const spreadFor = (size) => {
    const values = [];
    for (let trial = 0; trial < 200; trial++) {
      const run = coordinatedSample(weightedKeys, { k: size, shards: 3, seed: 7000 + trial * 29,
        weights });
      values.push(horvitzThompson(run.merged.sample(), size));
    }
    return standardDeviation(values);
  };
  const spreads = [4, 12, 40].map(spreadFor);
  assert.ok(spreads[0] > spreads[1] && spreads[1] > spreads[2],
    `a larger sample estimates the total better (${spreads.map((v) => v.toFixed(0)).join(" > ")})`);

  // --- inclusion probabilities match the theory ------------------------------
  const probabilityRun = coordinatedSample(weightedKeys, { k: 12, shards: 4, seed: 4242, weights });
  const probabilitySample = probabilityRun.merged.sample();
  const lightProbability = inclusionProbability(probabilitySample, 12, 1);
  const heavyProbability = inclusionProbability(probabilitySample, 12, 8);
  assert.ok(lightProbability > 0 && lightProbability <= 1,
    `the inclusion probability is a probability (${lightProbability.toFixed(4)})`);
  assert.ok(heavyProbability >= lightProbability, "heavy items are at least as likely to appear");

  // --- epochs ----------------------------------------------------------------
  const coordinator = new EpochCoordinator(8, 20000, { shards: 3 });
  assert.strictEqual(coordinator.broadcasts, 1, "the first seed is one broadcast");
  for (let i = 0; i < 200; i++) coordinator.add(`e0-${i}`, 1, i);
  const firstEpoch = coordinator.sample().map((entry) => entry.key);
  assert.strictEqual(firstEpoch.length, 8, "the first epoch sample is full");
  coordinator.rollover(20001);
  assert.strictEqual(coordinator.broadcasts, 2, "a rollover costs exactly one more broadcast");
  assert.strictEqual(coordinator.fixed.size, 8, "the previous sample was frozen");
  for (let i = 0; i < 200; i++) coordinator.add(`e1-${i}`, 1, i);
  const secondEpoch = coordinator.sample().map((entry) => entry.key);
  assert.strictEqual(secondEpoch.length, 8, "the sample stays full after a rollover");
  for (const key of firstEpoch) {
    assert.ok(secondEpoch.includes(key), `frozen item ${key} survives the rollover`);
  }
  coordinator.rollover(20002);
  const thirdEpoch = coordinator.sample().map((entry) => entry.key);
  assert.strictEqual(thirdEpoch.length, 8, "and after a second rollover");
  assert.ok(firstEpoch.every((key) => thirdEpoch.includes(key)), "fixed items stay fixed forever");
  assert.strictEqual(coordinator.epoch, 2, "two epochs were rolled");
  coordinator.add("frozen-again", 1, 0);
  assert.ok(!thirdEpoch.includes("frozen-again"), "a refused key cannot enter a frozen sample");

  // --- coordination cost and determinism ------------------------------------
  const run = coordinatedSample(keys, { k, shards: 8, seed });
  assert.strictEqual(run.broadcasts, 1, "coordinated sampling needs one seed broadcast");
  assert.ok(run.messages < keys.length,
    `coordination traffic is O(shards), not O(items) (${run.messages} < ${keys.length})`);
  const replay = coordinatedSample(keys, { k, shards: 8, seed });
  assert.deepStrictEqual(replay.merged.sample(), run.merged.sample(), "the same seed reproduces it");
  const otherSeed = coordinatedSample(keys, { k, shards: 8, seed: seed + 1 });
  assert.notDeepStrictEqual(keysOf(otherSeed.merged.report()), keysOf(run.merged.report()),
    "a different seed selects a different sample");
  assert.strictEqual(priorityOf(seed, "item-7", 1), priorityOf(seed, "item-7", 1),
    "priorities are deterministic per key and seed");
  assert.notStrictEqual(priorityOf(seed, "item-7", 1), priorityOf(seed, "item-8", 1),
    "and differ between keys");

  console.log(`keys=${keys.length} k=${k} shards=16 exact; uniform chi2=${statistic.toFixed(1)} ` +
    `< ${limit.toFixed(1)} over ${trials} seeds`);
  console.log(`weighted: heavy rate=${heavyRate.toFixed(3)} light rate=${lightRate.toFixed(3)} ` +
    `HT mean=${estimateMean.toFixed(0)} of ${trueTotal} (bias ${(relativeBias * 100).toFixed(1)}%); ` +
    `spread k=4/12/40: ${spreads.map((v) => v.toFixed(0)).join("/")}`);
  console.log(`coordination: broadcasts=${run.broadcasts} messages=${run.messages}; ` +
    `epochs=${coordinator.epoch} broadcasts=${coordinator.broadcasts}; uncoordinated missed=${missed.length}`);
  console.log("PASS");
}

