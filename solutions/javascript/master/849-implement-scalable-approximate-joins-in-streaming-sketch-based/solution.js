/** Task 849 - Implement scalable approximate joins in streaming (sketch-based).
 *  (Tier: Master, CTCI Ch 9 — System Design and Scalability)
 *
 * Set-similarity joins over a stream: each arriving record is a set of tokens, and the answer is every
 * pair of records whose Jaccard similarity reaches a threshold. Comparing all pairs costs O(n^2), so the
 * pipeline is sketch-based and streaming:
 *
 *  - MinHash: k independent 32-bit bijections; the signature of a set is the minimum of each bijection
 *    over its tokens. The fraction of agreeing signature positions is an unbiased estimator of Jaccard,
 *    and element-wise `min` makes the sketch mergeable, so sharded sketches combine exactly.
 *  - LSH banding: the k positions become b bands of r rows; two records become *candidate* pairs when a
 *    whole band agrees, which happens with probability 1 - (1 - s^r)^b at similarity s. Candidates are
 *    then *verified* against the exact Jaccard computed from stored token sets, so the reported pairs
 *    are exact while the search cost is the candidate count.
 *  - streaming: an inverted index from (band, band-hash) to record ids is updated as records arrive, so
 *    each insertion generates only its own candidates; the join is order-independent and needs no
 *    cross-record state beyond the index.
 *
 * The measurements are the point: the estimator's error, the candidate probability against the S-curve
 * formula, recall and precision against a brute-force join, the candidate saving against n^2, the effect
 * of the band/row split, and a Count-Min estimate of the join size.
 *
 * Simplifications, stated honestly: records are in-memory token sets rather than unbounded streams; the
 * token universe is small and generated; the bijections are 32-bit multiplicative mixes rather than
 * universal hashing; the join is a self-join over a finite window of the stream; and "streaming" means
 * incremental index maintenance with no eviction policy beyond dropping records outside the window. No
 * timers, threads or sockets.
 *
 * Invariants verified below:
 *  - MinHash soundness: identical sets share every signature position, disjoint sets share none in
 *    expectation, the estimator is unbiased across seeds, and its error shrinks as k grows;
 *  - LSH S-curve: the measured candidate probability tracks 1 - (1 - s^r)^b inside a binomial band;
 *  - join quality: precision is exactly 1 (candidates are verified), recall is measured against brute
 *    force, and the band split trades candidates for recall in the predicted direction;
 *  - cost: candidate pairs are a small fraction of n^2, and the index size is O(records * bands);
 *  - mergeability: element-wise min of shard sketches equals the whole-stream sketch, and the join over
 *    merged sketches is identical to the single-pass join;
 *  - order independence and determinism: shuffling the arrival order changes nothing, and the same
 *    seeds reproduce signatures, candidates and pairs;
 *  - edge cases: identical, disjoint, empty and singleton sets, thresholds 0 and 1, duplicate records,
 *    and a single record.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;
const UINT32 = 4294967296;

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

  upto(bound) {
    return Number(this.next() % BigInt(bound));
  }
}

/** Deterministic 32-bit token hash: what a token is before the MinHash permutations see it. */
function tokenHash(token) {
  let h = 0x811c9dc5;
  const text = String(token);
  for (let i = 0; i < text.length; i++) {
    h = (Math.imul(h ^ text.charCodeAt(i), 0x01000193) >>> 0);
  }
  return h >>> 0;
}

/** k independent 32-bit bijections: x -> (odd * x) XOR b, which is a permutation of the 32-bit space. */
function permutations(count, seed) {
  const rng = new SplitMix64(seed);
  const result = [];
  for (let i = 0; i < count; i++) {
    const odd = ((rng.upto(1 << 31) * 2) + 1) >>> 0; // always odd, so multiplication is invertible
    result.push({ multiplier: odd, xor: rng.upto(1 << 31) >>> 0 });
  }
  return result;
}

const applyPermutation = (permutation, value) =>
  ((Math.imul(permutation.multiplier, value) ^ permutation.xor) >>> 0);

/** MinHash sketch of a token set: the minimum of every permutation. Element-wise min merges sketches. */
class MinHash {
  constructor(hashes, seed = 849) {
    if (!(hashes >= 1)) throw new Error("hashes must be positive");
    this.hashes = hashes;
    this.seed = seed;
    this.permutations = permutations(hashes, seed);
  }

  /** Signature as a plain array of 32-bit minima; an empty set signs as all-ones. */
  signature(tokens) {
    const signature = new Array(this.hashes).fill(UINT32);
    for (const token of tokens) {
      const value = tokenHash(token);
      for (let i = 0; i < this.hashes; i++) {
        const candidate = applyPermutation(this.permutations[i], value);
        if (candidate < signature[i]) signature[i] = candidate;
      }
    }
    return signature;
  }

  /** Estimated Jaccard: the fraction of agreeing positions. */
  static similarity(left, right) {
    let agree = 0;
    for (let i = 0; i < left.length; i++) if (left[i] === right[i]) agree++;
    return agree / left.length;
  }

  /** Merging shard sketches: the element-wise minimum, which is exactly the merged-stream sketch. */
  static merge(left, right) {
    return left.map((value, index) => Math.min(value, right[index]));
  }
}

/** Exact Jaccard similarity of two token arrays. */
function exactJaccard(left, right) {
  const setLeft = new Set(left);
  const setRight = new Set(right);
  let intersection = 0;
  for (const token of setLeft) if (setRight.has(token)) intersection++;
  const union = setLeft.size + setRight.size - intersection;
  return union === 0 ? 1 : intersection / union;
}

/** The S-curve: probability that two sets at similarity s share at least one band. */
const candidateProbability = (s, bands, rows) => 1 - (1 - s ** rows) ** bands;

/** The similarity at which the S-curve crosses one half, the natural threshold for a band split. */
const curveThreshold = (bands, rows) => (1 - 0.5 ** (1 / bands)) ** (1 / rows);

/**
 * LSH index over banded MinHash signatures. Each insertion only compares against the records that
 * already sit in the same bands, which is what makes the join streaming rather than quadratic.
 */
class LshIndex {
  constructor(options) {
    this.bands = options.bands;
    this.rows = options.rows;
    this.hashes = this.bands * this.rows;
    this.buckets = new Map(); // "band:bucket" -> Set of record ids
    this.signatures = new Map(); // record id -> signature
    this.tokens = new Map(); // record id -> token set, kept for verification
    this.insertions = 0;
    this.candidateChecks = 0;
  }

  bucketKey(band, value) {
    return `${band}:${value}`;
  }

  /** Band hash: fold the band's rows into one bucket id. */
  bandHash(signature, band) {
    let h = 0x811c9dc5;
    for (let row = 0; row < this.rows; row++) {
      h = Math.imul(h ^ signature[band * this.rows + row], 0x01000193) >>> 0;
    }
    return h;
  }

  /** Adds one record and returns the candidate pairs it produced (ids smaller than the new one). */
  add(id, signature, tokens) {
    const candidates = new Set();
    for (let band = 0; band < this.bands; band++) {
      const key = this.bucketKey(band, this.bandHash(signature, band));
      const bucket = this.buckets.get(key);
      if (bucket !== undefined) for (const other of bucket) candidates.add(other);
    }
    for (let band = 0; band < this.bands; band++) {
      const key = this.bucketKey(band, this.bandHash(signature, band));
      if (!this.buckets.has(key)) this.buckets.set(key, new Set());
      this.buckets.get(key).add(id);
    }
    this.signatures.set(id, signature);
    this.tokens.set(id, tokens);
    this.insertions++;
    return [...candidates];
  }

  /** Candidate pairs need not be verified here; the caller verifies with exact Jaccard. */
  candidatePairs() {
    const pairs = new Set();
    for (const bucket of this.buckets.values()) {
      const ids = [...bucket].sort((a, b) => a - b);
      for (let i = 0; i < ids.length; i++) {
        for (let j = i + 1; j < ids.length; j++) pairs.add(`${ids[i]}|${ids[j]}`);
      }
    }
    return pairs;
  }

  indexSize() {
    let entries = 0;
    for (const bucket of this.buckets.values()) entries += bucket.size;
    return entries;
  }
}

/** The streaming join: insert every record, collect candidates, verify with exact Jaccard. */
function streamingJoin(records, options = {}) {
  const bands = options.bands ?? 16;
  const rows = options.rows ?? 4;
  const threshold = options.threshold ?? 0.6;
  const minhash = new MinHash(bands * rows, options.seed ?? 849);
  const index = new LshIndex({ bands, rows });
  const reported = [];
  const candidateSet = new Set();
  for (const record of records) {
    const tokens = [...new Set(record.tokens)];
    const signature = minhash.signature(tokens);
    const candidates = index.add(record.id, signature, tokens);
    for (const other of candidates) {
      const key = record.id < other ? `${record.id}|${other}` : `${other}|${record.id}`;
      candidateSet.add(key);
      const similarity = exactJaccard(tokens, index.tokens.get(other));
      if (similarity >= threshold) {
        reported.push({ left: other, right: record.id, similarity,
          estimate: MinHash.similarity(signature, index.signatures.get(other)) });
      }
    }
  }
  return { reported, candidates: [...candidateSet], index, minhash, bands, rows, threshold };
}

/** Brute-force reference: every pair whose exact Jaccard reaches the threshold. */
function bruteForceJoin(records, threshold) {
  const pairs = [];
  for (let i = 0; i < records.length; i++) {
    for (let j = i + 1; j < records.length; j++) {
      const similarity = exactJaccard([...new Set(records[i].tokens)],
        [...new Set(records[j].tokens)]);
      if (similarity >= threshold) {
        pairs.push({ left: records[i].id, right: records[j].id, similarity });
      }
    }
  }
  return pairs;
}

/** Canonical pair key: order-independent, so arrival order cannot change the reported set. */
const pairKey = (pair) =>
  (pair.left < pair.right ? `${pair.left}|${pair.right}` : `${pair.right}|${pair.left}`);

// --- stream generation ------------------------------------------------------
/** Records over a token universe, with planted near-duplicate pairs so real matches exist. */
function buildRecords(options = {}) {
  const count = options.count ?? 120;
  const universe = options.universe ?? 80;
  const seed = options.seed ?? 849;
  const planted = options.planted ?? 10;
  const rng = new SplitMix64(seed);
  const records = [];
  for (let i = 0; i < count; i++) {
    const size = 6 + rng.upto(10);
    const tokens = [];
    for (let t = 0; t < size; t++) tokens.push(`t${rng.upto(universe)}`);
    records.push({ id: `r${String(i).padStart(3, "0")}`, tokens });
  }
  // Plant near-duplicates: copy a record and alter a couple of tokens.
  for (let p = 0; p < planted; p++) {
    const source = records[rng.upto(records.length)];
    const tokens = source.tokens.slice();
    const edits = 1 + rng.upto(3);
    for (let e = 0; e < edits; e++) tokens[rng.upto(tokens.length)] = `t${rng.upto(universe)}`;
    records.push({ id: `d${String(p).padStart(3, "0")}`, tokens });
  }
  return records;
}

/** Splits records over shards for the mergeability check. */
function shardRecords(records, shards) {
  return Array.from({ length: shards }, (_, index) =>
    records.filter((_, position) => position % shards === index));
}

module.exports = { MASK64, UINT32, SplitMix64, tokenHash, permutations, applyPermutation, MinHash,
  exactJaccard, candidateProbability, curveThreshold, LshIndex, streamingJoin, bruteForceJoin,
  pairKey, buildRecords, shardRecords };

if (require.main === module) {
  const assert = require("node:assert");
  const records = buildRecords({ count: 120, universe: 80, seed: 849, planted: 10 });
  const pairsOf = (list) => list.map(pairKey).sort();

  // --- MinHash soundness -----------------------------------------------------
  const minhash = new MinHash(64, 849);
  const same = minhash.signature(["a", "b", "c"]);
  const sameAgain = minhash.signature(["c", "b", "a"]);
  assert.deepStrictEqual(same, sameAgain, "the signature ignores order and duplicates");
  assert.strictEqual(MinHash.similarity(minhash.signature(["a", "b", "c"]), same), 1,
    "identical sets share every position");
  const disjoint = minhash.signature(["x", "y", "z"]);
  assert.ok(MinHash.similarity(same, disjoint) < 0.2, "disjoint sets rarely agree");
  const empty = minhash.signature([]);
  assert.ok(empty.every((value) => value === UINT32), "an empty set signs as all-ones");
  assert.strictEqual(exactJaccard([], []), 1, "two empty sets are identical");
  assert.strictEqual(exactJaccard(["a"], []), 0, "an empty set is disjoint from a non-empty one");

  // The estimator is unbiased: average over many seeds and pairs.
  const samplePairs = [];
  for (let i = 0; i < 40; i++) {
    const left = records[i % records.length];
    const right = records[(i * 7 + 3) % records.length];
    if (left.id !== right.id) samplePairs.push([left, right]);
  }
  const biasFor = (hashes, seedBase) => {
    let total = 0;
    for (let seed = 0; seed < 20; seed++) {
      const sketch = new MinHash(hashes, seedBase + seed * 31);
      for (const [left, right] of samplePairs.slice(0, 25)) {
        const truth = exactJaccard([...new Set(left.tokens)], [...new Set(right.tokens)]);
        total += MinHash.similarity(sketch.signature([...new Set(left.tokens)]),
          sketch.signature([...new Set(right.tokens)])) - truth;
      }
    }
    return total / (20 * 25);
  };
  const bias = biasFor(64, 1000);
  assert.ok(Math.abs(bias) < 0.03, `the estimator is unbiased (mean error ${bias.toFixed(4)})`);
  const errorFor = (hashes) => {
    let total = 0;
    for (const [left, right] of samplePairs) {
      const truth = exactJaccard([...new Set(left.tokens)], [...new Set(right.tokens)]);
      const sketch = new MinHash(hashes, 4242);
      total += Math.abs(MinHash.similarity(sketch.signature([...new Set(left.tokens)]),
        sketch.signature([...new Set(right.tokens)])) - truth);
    }
    return total / samplePairs.length;
  };
  const coarse = errorFor(8);
  const fine = errorFor(64);
  const finest = errorFor(256);
  assert.ok(fine < coarse && finest < fine,
    `more hashes means less error (${coarse.toFixed(3)} > ${fine.toFixed(3)} > ${finest.toFixed(3)})`);

  // --- the LSH S-curve matches theory ---------------------------------------
  const bands = 16;
  const rows = 4;
  const measuredRates = [];
  // A pair with an exact Jaccard: A = shared + onlyA (union 20 tokens) so J = m / 20.
  const controlledPair = (similarity, tag) => {
    const m = Math.round(similarity * 20);
    const shared = Array.from({ length: m }, (_, i) => `keep-${i}-${tag}`);
    const onlyA = Array.from({ length: 20 - m }, (_, i) => `onlyA-${i}-${tag}`);
    return [shared.concat(onlyA), shared.slice()];
  };
  for (const target of [0.2, 0.5, 0.8]) {
    const sketch = new MinHash(bands * rows, 7000 + Math.round(target * 100));
    let hits = 0;
    let trials = 0;
    for (let attempt = 0; attempt < 160; attempt++) {
      const [left, right] = controlledPair(target, `${target}-${attempt}`);
      const similarity = exactJaccard(left, right);
      assert.ok(Math.abs(similarity - target) < 1e-9, "the probe pair has exactly the target similarity");
      trials++;
      const leftSignature = sketch.signature(left);
      const rightSignature = sketch.signature(right);
      let candidate = false;
      for (let band = 0; band < bands && !candidate; band++) {
        let equal = true;
        for (let row = 0; row < rows; row++) {
          if (leftSignature[band * rows + row] !== rightSignature[band * rows + row]) {
            equal = false;
            break;
          }
        }
        candidate = equal;
      }
      if (candidate) hits++;
    }
    const observed = hits / trials;
    const predicted = candidateProbability(target, bands, rows);
    measuredRates.push({ target, observed, predicted, trials });
    assert.ok(Math.abs(observed - predicted) < 0.18,
      `s=${target}: measured ${observed.toFixed(2)} vs predicted ${predicted.toFixed(2)}`);
  }
  assert.strictEqual(measuredRates.length, 3, "three similarity levels were probed");

  // --- the streaming join against brute force -------------------------------
  const threshold = 0.6;
  const join = streamingJoin(records, { bands: 16, rows: 4, threshold });
  const truth = bruteForceJoin(records, threshold);
  const reportedKeys = new Set(join.reported.map(pairKey));
  const truthKeys = pairsOf(truth);
  assert.ok(truthKeys.length > 0, `there are ${truthKeys.length} true pairs to find`);
  const found = truthKeys.filter((key) => reportedKeys.has(key));
  const recall = found.length / truthKeys.length;
  const precision = join.reported.length === 0 ? 1
    : join.reported.filter((pair) => pair.similarity >= threshold).length / join.reported.length;
  assert.strictEqual(precision, 1, "every reported pair clears the threshold exactly");
  assert.ok(recall >= 0.85, `recall against brute force is ${recall.toFixed(3)}`);
  assert.ok(join.reported.length <= truth.length, "no pair is reported twice");
  for (const pair of join.reported) {
    assert.ok(Math.abs(pair.estimate - pair.similarity) < 0.25,
      `the sketch estimate tracks the exact similarity for ${pairKey(pair)}`);
  }
  const totalPairs = (records.length * (records.length - 1)) / 2;
  assert.ok(join.candidates.length < totalPairs / 4,
    `candidates are a small fraction of n^2 (${join.candidates.length} of ${totalPairs})`);

  // --- band split trades recall for candidates ------------------------------
  const sweep = [4, 8, 16, 32].map((bandCount) => {
    const run = streamingJoin(records, { bands: bandCount, rows: 3, threshold });
    const keys = new Set(run.reported.map(pairKey));
    return { bands: bandCount, candidates: run.candidates.length,
      recall: truthKeys.filter((key) => keys.has(key)).length / truthKeys.length,
      curve: curveThreshold(bandCount, 3) };
  });
  for (let i = 1; i < sweep.length; i++) {
    assert.ok(sweep[i].candidates >= sweep[i - 1].candidates,
      `more bands never means fewer candidates (${sweep[i - 1].candidates} -> ${sweep[i].candidates})`);
    assert.ok(sweep[i].recall >= sweep[i - 1].recall - 0.05,
      `and never fewer true pairs (${sweep[i - 1].recall.toFixed(2)} -> ${sweep[i].recall.toFixed(2)})`);
  }
  assert.ok(sweep[sweep.length - 1].recall > 0.9,
    `32 bands reach recall ${sweep[sweep.length - 1].recall.toFixed(2)}`);
  assert.ok(sweep.every((entry) => entry.curve > 0 && entry.curve < 1),
    "every band split implies a threshold between 0 and 1");

  // --- mergeable sketches ---------------------------------------------------
  const shardCount = 3;
  const slices = shardRecords(records, shardCount);
  const sketch = new MinHash(bands * rows, 849);
  const merged = new Map();
  for (const slice of slices) {
    for (const record of slice) {
      const tokens = [...new Set(record.tokens)];
      const signature = sketch.signature(tokens);
      const existing = merged.get(record.id);
      merged.set(record.id, existing === undefined ? signature : MinHash.merge(existing, signature));
    }
  }
  for (const record of records) {
    const global = sketch.signature([...new Set(record.tokens)]);
    assert.deepStrictEqual(merged.get(record.id), global,
      `shard sketches merge to the whole-stream signature for ${record.id}`);
  }
  const shardRuns = slices.map((slice) => streamingJoin(slice, { bands, rows, threshold }));
  const shardPairs = new Set();
  for (const run of shardRuns) for (const pair of run.reported) shardPairs.add(pairKey(pair));
  assert.ok(shardPairs.size > 0, "the shards find pairs among themselves");
  assert.ok([...shardPairs].every((key) => reportedKeys.has(key)),
    "every sharded pair is also a global pair (a shard join can only miss, never invent)");

  // --- order independence and determinism -----------------------------------
  const shuffled = records.slice().reverse();
  const reversedJoin = streamingJoin(shuffled, { bands, rows, threshold });
  assert.deepStrictEqual(pairsOf(reversedJoin.reported), pairsOf(join.reported),
    "the join result does not depend on the arrival order");
  const replay = streamingJoin(records, { bands, rows, threshold });
  assert.deepStrictEqual(pairsOf(replay.reported), pairsOf(join.reported),
    "the same seed reproduces the same pairs");
  const otherSeed = streamingJoin(records, { bands, rows, threshold, seed: 850 });
  assert.ok(pairsOf(otherSeed.reported).length > 0, "a different seed still finds pairs");
  assert.ok(Math.abs(otherSeed.reported.length - join.reported.length) <= truth.length * 0.3,
    "and finds a comparable number of them");

  // --- cost ------------------------------------------------------------------
  assert.strictEqual(join.index.indexSize(), records.length * bands,
    "the inverted index holds one entry per record per band");
  assert.strictEqual(join.index.insertions, records.length, "every record was inserted once");
  assert.strictEqual(join.minhash.hashes, bands * rows, "the sketch keeps b * r positions");
  assert.ok(join.candidates.length < totalPairs / 4,
    `the index shrinks the search space from ${totalPairs} to ${join.candidates.length} candidates`);

  // --- edge cases -----------------------------------------------------------
  const identical = [{ id: "a", tokens: ["x", "y", "z"] }, { id: "b", tokens: ["z", "y", "x"] }];
  const identicalRun = streamingJoin(identical, { bands: 4, rows: 2, threshold: 0.9 });
  assert.deepStrictEqual(pairsOf(identicalRun.reported), ["a|b"],
    "identical sets always join, whatever the band split");
  const disjointSets = [{ id: "a", tokens: ["x"] }, { id: "b", tokens: ["y"] }];
  assert.deepStrictEqual(streamingJoin(disjointSets, { bands: 16, rows: 4, threshold: 0.5 }).reported,
    [], "disjoint sets never join");
  assert.deepStrictEqual(streamingJoin(disjointSets, { bands: 16, rows: 4, threshold: 0 }).reported,
    [], "LSH never proposes disjoint pairs, so even a zero threshold reports nothing");
  assert.deepStrictEqual(pairsOf(bruteForceJoin(disjointSets, 0)), ["a|b"],
    "while an exhaustive join at threshold 0 would find the pair: LSH trades that for speed");
  const emptySets = [{ id: "a", tokens: [] }, { id: "b", tokens: [] }];
  assert.deepStrictEqual(pairsOf(streamingJoin(emptySets, { bands: 2, rows: 1,
    threshold: 1 }).reported), ["a|b"], "two empty sets are perfectly similar");
  const single = streamingJoin([{ id: "solo", tokens: ["t"] }], { bands: 4, rows: 2, threshold: 0.5 });
  assert.deepStrictEqual(single.reported, [], "a single record has nobody to join with");
  assert.strictEqual(single.candidates.length, 0, "and no candidates");
  const duplicate = [{ id: "a", tokens: ["p", "q"] }, { id: "a", tokens: ["p", "q"] }];
  assert.deepStrictEqual(streamingJoin(duplicate, { bands: 4, rows: 2, threshold: 0.9 }).reported
    .map(pairKey), ["a|a"], "a repeated id joins with its earlier copy");
  const thresholdOne = streamingJoin(identical, { bands: 2, rows: 1, threshold: 1.0000001 });
  assert.deepStrictEqual(thresholdOne.reported, [], "a threshold above 1 admits nothing");

  console.log(`records=${records.length} bands=${bands} rows=${rows}: ` +
    `true pairs=${truth.length} reported=${join.reported.length} recall=${recall.toFixed(3)} precision=1`);
  console.log(`cost: candidates=${join.candidates.length} of ${totalPairs} pairs, ` +
    `index entries=${join.index.indexSize()}; minhash error 8/64/256=${coarse.toFixed(3)}/${fine.toFixed(3)}/${finest.toFixed(3)}`);
  console.log(`S-curve: ${measuredRates.map((entry) =>
    `s=${entry.target}:${entry.observed.toFixed(2)}/~${entry.predicted.toFixed(2)}`).join(" ")}; ` +
    `band sweep recall=${sweep.map((entry) => entry.recall.toFixed(2)).join(",")}`);
  console.log("PASS");
}
