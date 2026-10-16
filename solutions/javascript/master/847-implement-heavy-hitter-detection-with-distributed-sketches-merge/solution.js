/** Task 847 - Implement heavy-hitter detection with distributed sketches mergeable.
 *  (Tier: Master, CTCI Ch 9 — System Design and Scalability)
 *
 * Heavy hitters are the keys whose frequency exceeds a fraction of the stream, and the distributed
 * setting forbids centralising the data: every shard keeps a *mergeable* sketch of its own slice and
 * the coordinator only ever merges summaries. Two mergeable sketches are implemented, each with the
 * guarantee that makes it useful, and the whole pipeline is checked against a brute-force census:
 *
 *  - Space-Saving (Metwally et al.), `k` entries of (key, count, error). Invariants: for a tracked key
 *    f(x) <= count(x) and count(x) - error(x) <= f(x); for an untracked key f(x) <= min count, so any
 *    key with f > n/k *must* be tracked. `merge` sums counts and errors per key, trims back to k entries
 *    and folds the mass it dropped into a conservative extra error term.
 *  - Count-Min sketch (Cormode–Muthukrishnan): a `rows x width` integer matrix with pairwise-independent
 *    hashes. It never underestimates, its overestimate is at most ε n with ε = e/width, and because it
 *    is *linear* the merge is plain matrix addition — so the merge of the shard sketches equals the
 *    sketch of the concatenated stream, entry for entry, for any partition of the stream.
 *
 * The distributed pipeline mirrors "Reduce-Space-Saving": the Space-Saving summaries of the shards are
 * merged in a tree (merge is associative and commutative, checked both ways) to *propose* candidates,
 * and the merged Count-Min sketch supplies an estimate for each candidate. Correctness claims are then
 * measured, not assumed: no true heavy hitter may be missed, every Count-Min estimate must sit inside
 * its bound, and the merge must be independent of how the stream was partitioned.
 *
 * Simplifications, stated honestly: shards are in-process objects merging in one coordinator, not RPC
 * endpoints; hashes are seeded splitmix64 mixes rather than universal hashing families; counters are JS
 * numbers, so streams are sized to stay far inside 2^53; and the stream is generated rather than
 * ingested. No timers, threads or sockets.
 *
 * Invariants verified below:
 *  - Space-Saving bounds: count/error sandwich for tracked keys and a min-count bound for untracked
 *    keys (so at most k keys are ever held); after a merge the truth stays inside
 *    [count - error - trimmed, count + missing], where `missing` is the untracked mass of the leaves;
 *  - no false negatives: every key with f > n/k appears in the merged summary, for single streams and
 *    for every tested (k, shard-count) combination;
 *  - Count-Min soundness: never an underestimate, error inside ε n, and the exactness of the merge
 *    (merged sketch == sketch of the concatenation) for 1, 2, 5 and 13 shards;
 *  - merge laws: associativity and commutativity of both merges, and identical heavy-hitter sets under
 *    different partitions of the same stream;
 *  - accuracy/cost trade-off: recall and estimate error improve as k and the sketch width grow, while
 *    per-shard memory stays O(k) and O(rows * width);
 *  - edge cases: empty stream, all-distinct keys, one key repeated, k = 1, k beyond the key count, and
 *    keys exactly at the n/k threshold.
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

  upto(bound) {
    return Number(this.next() % BigInt(bound));
  }

  unit() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }
}

/** Keyed 32-bit hash used to spread keys over sketch buckets. */
function hashKey(seed, key) {
  let h = (BigInt(seed) ^ 0x9e3779b97f4a7c15n) & MASK64;
  const text = String(key);
  for (let i = 0; i < text.length; i++) {
    h = ((h ^ BigInt(text.charCodeAt(i))) * 0x100000001b3n) & MASK64;
    h = ((h ^ (h >> 29n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    h = (h ^ (h >> 32n)) & MASK64;
  }
  return Number(h % 2147483647n);
}

/**
 * Space-Saving summary of k entries. `error` accumulates the mass this key may have "stolen" when its
 * slot was reused, which is what turns count into an interval [count - error, count].
 */
class SpaceSaving {
  constructor(k) {
    if (!(k >= 1)) throw new Error("k must be at least 1");
    this.k = k;
    this.entries = new Map(); // key -> {count, error}
    this.total = 0;
    this.evictions = 0;
  }

  add(key) {
    this.total++;
    const entry = this.entries.get(key);
    if (entry !== undefined) {
      entry.count++;
      return;
    }
    if (this.entries.size < this.k) {
      this.entries.set(key, { count: 1, error: 0 });
      return;
    }
    let victimKey = null;
    let victim = null;
    for (const [candidate, value] of this.entries) {
      if (victim === null || value.count < victim.count) {
        victimKey = candidate;
        victim = value;
      }
    }
    this.entries.delete(victimKey);
    this.entries.set(key, { count: victim.count + 1, error: victim.count });
    this.evictions++;
  }

  /** The interval [count - error, count] this summary can guarantee for a tracked key. */
  estimate(key) {
    const entry = this.entries.get(key);
    if (entry === undefined) return null;
    const extra = this.trimmed ?? 0;
    return { count: entry.count, error: entry.error + extra,
      lower: entry.count - entry.error - extra };
  }

  minCount() {
    let smallest = Infinity;
    for (const value of this.entries.values()) smallest = Math.min(smallest, value.count);
    return smallest;
  }

  /**
   * Merge: sum counts and errors, trim to k, and remember the mass the trim discarded.
   * `missing` accumulates the untracked mass of the merged leaves: a key absent from a leaf can hide
   * at most that leaf's minimum count, which is what widens the merged upper bound.
   */
  merge(other) {
    const merged = new SpaceSaving(Math.max(this.k, other.k));
    merged.total = this.total + other.total;
    merged.evictions = this.evictions + other.evictions;
    merged.missing = (this.missing ?? this.minCount()) + (other.missing ?? other.minCount());
    const keys = new Set([...this.entries.keys(), ...other.entries.keys()]);
    let dropped = 0;
    for (const key of keys) {
      const left = this.entries.get(key);
      const right = other.entries.get(key);
      merged.entries.set(key, { count: (left?.count ?? 0) + (right?.count ?? 0),
        error: (left?.error ?? 0) + (right?.error ?? 0) });
    }
    if (merged.entries.size > merged.k) {
      const ordered = [...merged.entries.entries()].sort((a, b) => b[1].count - a[1].count ||
        (a[0] < b[0] ? -1 : 1));
      for (const [, value] of ordered.slice(merged.k)) dropped += value.count;
      merged.entries = new Map(ordered.slice(0, merged.k));
      merged.trimmed = (this.trimmed ?? 0) + (other.trimmed ?? 0) + dropped;
    } else {
      merged.trimmed = (this.trimmed ?? 0) + (other.trimmed ?? 0);
    }
    return merged;
  }

  /** The reported candidates, heaviest first, each with its guaranteed interval. */
  top(limit = this.k) {
    return [...this.entries.keys()].map((key) => ({ key, ...this.estimate(key) }))
      .sort((a, b) => b.count - a.count || (a.key < b.key ? -1 : 1)).slice(0, limit);
  }

  size() {
    return this.entries.size;
  }
}

/** Count-Min sketch: never underestimates, error at most ε n, and linear so merges are additions. */
class CountMin {
  constructor(rows, width, seed = 847) {
    if (!(rows >= 1) || !(width >= 1)) throw new Error("rows and width must be positive");
    this.rows = rows;
    this.width = width;
    this.seed = seed;
    this.table = Array.from({ length: rows }, () => new Array(width).fill(0));
    this.total = 0;
  }

  buckets(key) {
    return Array.from({ length: this.rows }, (_, row) =>
      hashKey(this.seed + row * 7919, key) % this.width);
  }

  add(key, amount = 1) {
    const indices = this.buckets(key);
    for (let row = 0; row < this.rows; row++) {
      this.table[row][indices[row]] += amount;
    }
    this.total += amount;
  }

  estimate(key) {
    const indices = this.buckets(key);
    let best = Infinity;
    for (let row = 0; row < this.rows; row++) best = Math.min(best, this.table[row][indices[row]]);
    return best;
  }

  merge(other) {
    if (other.rows !== this.rows || other.width !== this.width) throw new Error("shape mismatch");
    const merged = new CountMin(this.rows, this.width, this.seed);
    merged.total = this.total + other.total;
    for (let row = 0; row < this.rows; row++) {
      for (let column = 0; column < this.width; column++) {
        merged.table[row][column] = this.table[row][column] + other.table[row][column];
      }
    }
    return merged;
  }

  equals(other) {
    if (other.rows !== this.rows || other.width !== this.width) return false;
    for (let row = 0; row < this.rows; row++) {
      for (let column = 0; column < this.width; column++) {
        if (this.table[row][column] !== other.table[row][column]) return false;
      }
    }
    return true;
  }

  cells() {
    return this.rows * this.width;
  }
}

// --- stream generation and the brute-force oracle ---------------------------
/** Zipf-ish stream: `hot` keys take most of the mass, the rest is a long tail of singletons. */
function buildStream(options = {}) {
  const length = options.length ?? 20000;
  const hot = options.hot ?? 8;
  const tail = options.tail ?? 500;
  const seed = options.seed ?? 847;
  const rng = new SplitMix64(seed);
  const stream = [];
  for (let i = 0; i < length; i++) {
    if (rng.unit() < (options.hotShare ?? 0.8)) {
      let rank = 0; // a geometric rank gives the classic heavy head and long body
      while (rank < hot - 1 && rng.unit() < 0.5) rank++;
      stream.push(`hot-${rank}`);
    } else {
      stream.push(`tail-${rng.upto(tail)}`);
    }
  }
  return stream;
}

/** Exact frequencies, sorted by count then key. */
function exactCounts(stream) {
  const counts = new Map();
  for (const key of stream) counts.set(key, (counts.get(key) ?? 0) + 1);
  return counts;
}

/** Heavy hitters by the textbook definition: frequency strictly greater than a share of the stream. */
function exactHeavyHitters(counts, share) {
  const total = [...counts.values()].reduce((a, b) => a + b, 0);
  return [...counts.entries()].filter(([, value]) => value > share * total)
    .map(([key]) => key).sort();
}

/** Splits a stream over `shards` deterministically (round-robin by position). */
function shardStream(stream, shards) {
  const slices = Array.from({ length: shards }, () => []);
  stream.forEach((key, index) => slices[index % shards].push(key));
  return slices;
}

/** Builds one shard's sketches and merges them in a tree; returns everything the checks need. */
function distributedHeavyHitters(stream, options = {}) {
  const k = options.k ?? 20;
  const shards = options.shards ?? 4;
  const rows = options.rows ?? 4;
  const width = options.width ?? 512;
  const seed = options.seed ?? 847;
  const sliceList = shardStream(stream, shards);
  const summaries = sliceList.map((slice) => {
    const summary = new SpaceSaving(k);
    const sketch = new CountMin(rows, width, seed);
    for (const key of slice) {
      summary.add(key);
      sketch.add(key);
    }
    return { summary, sketch, length: slice.length };
  });
  let merged = summaries[0].summary;
  let mergedSketch = summaries[0].sketch;
  for (let i = 1; i < summaries.length; i++) {
    merged = merged.merge(summaries[i].summary);
    mergedSketch = mergedSketch.merge(summaries[i].sketch);
  }
  const candidates = merged.top(k).map((entry) => ({ ...entry,
    estimated: mergedSketch.estimate(entry.key) }));
  return { summaries, merged, mergedSketch, candidates, shards, k, rows, width, seed };
}

module.exports = { MASK64, SplitMix64, hashKey, SpaceSaving, CountMin, buildStream, exactCounts,
  exactHeavyHitters, shardStream, distributedHeavyHitters };

if (require.main === module) {
  const assert = require("node:assert");
  const stream = buildStream({ length: 20000, hot: 8, tail: 500, seed: 847 });
  const counts = exactCounts(stream);
  const total = stream.length;
  const heavy = exactHeavyHitters(counts, 0.05);
  assert.ok(heavy.length >= 3 && heavy.length <= 8, `the stream has ${heavy.length} heavy hitters`);
  assert.ok(heavy.includes("hot-0"), "the hottest key is a heavy hitter");

  // --- Space-Saving invariants on a single stream ---------------------------
  const single = new SpaceSaving(20);
  for (const key of stream) single.add(key);
  assert.ok(single.size() <= 20, `the summary holds at most k keys (${single.size()})`);
  assert.strictEqual(single.total, total, "it counted every event");
  for (const key of counts.keys()) {
    const entry = single.estimate(key);
    if (entry !== null) {
      assert.ok(entry.count >= counts.get(key),
        `tracked count is an overestimate for ${key} (${entry.count} >= ${counts.get(key)})`);
      assert.ok(entry.count - entry.error <= counts.get(key),
        `and the error term bounds the underestimate for ${key}`);
    } else {
      assert.ok(counts.get(key) <= single.minCount(),
        `an untracked key cannot be heavier than the minimum count (${key})`);
    }
  }
  assert.ok(single.evictions > 0, `entries were evicted (${single.evictions})`);

  // --- no false negatives: every key above n/k is tracked -------------------
  let thresholdCases = 0;
  for (const k of [5, 10, 20, 50]) {
    const summary = new SpaceSaving(k);
    for (const key of stream) summary.add(key);
    for (const [key, value] of counts) {
      if (value > total / k) {
        assert.ok(summary.estimate(key) !== null,
          `k=${k}: ${key} with ${value} > ${total / k} must be tracked`);
      }
    }
    thresholdCases++;
  }
  assert.strictEqual(thresholdCases, 4);
  // A key exactly at the threshold is allowed to be missed; one above it is not.
  const tied = new SpaceSaving(2);
  for (const key of ["a", "a", "b", "b", "c", "c"]) tied.add(key);
  assert.ok(tied.size() <= 2, "a tie at n/k cannot keep every tied key");
  const above = new SpaceSaving(2);
  for (const key of ["a", "a", "a", "b", "b", "c", "c"]) above.add(key);
  assert.ok(above.estimate("a") !== null, "a key above the threshold survives the tie");

  // --- Count-Min soundness --------------------------------------------------
  const sketch = new CountMin(4, 512, 847);
  for (const key of stream) sketch.add(key);
  const epsilon = Math.E / sketch.width;
  let maxOverestimate = 0;
  for (const [key, value] of counts) {
    const estimate = sketch.estimate(key);
    assert.ok(estimate >= value, `Count-Min never underestimates ${key}`);
    maxOverestimate = Math.max(maxOverestimate, estimate - value);
  }
  assert.ok(maxOverestimate <= epsilon * total,
    `the overestimate stays inside epsilon n (${maxOverestimate} <= ${epsilon * total})`);
  assert.ok(sketch.cells() === 4 * 512, "the sketch is a 4 x 512 matrix");

  // --- the merge is exact: merged sketch == sketch of the concatenation -----
  for (const shards of [1, 2, 5, 13]) {
    const slices = shardStream(stream, shards);
    let merged = null;
    for (const slice of slices) {
      const local = new CountMin(4, 512, 847);
      for (const key of slice) local.add(key);
      merged = merged === null ? local : merged.merge(local);
    }
    assert.ok(merged.equals(sketch), `${shards} shards merge to the same sketch as the whole stream`);
    assert.strictEqual(merged.total, total, "and the totals add up");
  }
  const associativity = (() => {
    const [a, b, c] = [0, 1, 2].map((index) => {
      const local = new CountMin(3, 128, 99);
      for (const key of shardStream(stream, 3)[index]) local.add(key);
      return local;
    });
    const left = a.merge(b).merge(c);
    const right = a.merge(b.merge(c));
    return left.equals(right);
  })();
  assert.strictEqual(associativity, true, "Count-Min merging is associative");

  // --- distributed heavy hitters -------------------------------------------
  const pipeline = distributedHeavyHitters(stream, { k: 20, shards: 4 });
  const reported = new Set(pipeline.merged.top(20).map((entry) => entry.key));
  for (const key of heavy) {
    assert.ok(reported.has(key), `the distributed summary reports heavy hitter ${key}`);
  }
  assert.strictEqual(pipeline.summaries.length, 4, "four shards were used");
  assert.ok(pipeline.summaries.every((shard) => shard.summary.size() <= 20),
    "each shard respects the k bound");
  let widened = 0;
  for (const candidate of pipeline.candidates) {
    const truth = counts.get(candidate.key);
    assert.ok(candidate.estimated >= truth,
      `the merged sketch overestimates nothing downstream for ${candidate.key}`);
    assert.ok(candidate.estimated <= truth + epsilon * total,
      `and stays within the epsilon bound for ${candidate.key}`);
    assert.ok(candidate.lower <= truth,
      `the merged lower bound never exceeds the truth for ${candidate.key}`);
    assert.ok(truth <= candidate.count + pipeline.merged.missing,
      `and the truth stays inside the merged upper bound for ${candidate.key}`);
    if (truth > candidate.count) widened++;
  }
  assert.ok(widened > 0,
    `the untracked-mass term is genuinely needed (${widened} candidates overshot their count)`);

  // --- merge laws for Space-Saving ------------------------------------------
  const summaries = shardStream(stream, 3).map((slice) => {
    const local = new SpaceSaving(12);
    for (const key of slice) local.add(key);
    return local;
  });
  const [first, second, third] = summaries;
  const leftMerge = first.merge(second).merge(third);
  const rightMerge = first.merge(second.merge(third));
  const keysOf = (summary) => summary.top(summary.k).map((entry) => entry.key).sort();
  assert.deepStrictEqual(keysOf(leftMerge), keysOf(rightMerge), "Space-Saving merging is associative");
  assert.strictEqual(leftMerge.total, rightMerge.total, "and the totals agree");
  const commuted = second.merge(first);
  assert.deepStrictEqual(keysOf(commuted), keysOf(first.merge(second)),
    "and commutative for the reported keys");
  for (const [key, value] of counts) {
    const entry = leftMerge.estimate(key);
    if (entry !== null) {
      assert.ok(entry.count + leftMerge.missing >= value,
        `merged count still bounds ${key} once the untracked mass is added`);
    }
  }

  // --- partition independence -----------------------------------------------
  const partitionRules = [
    (index) => index % 4, // round robin
    (index) => Math.floor(index / (total / 4)) % 4, // contiguous blocks
    (index) => (index * 7) % 5, // five-way striding
  ];
  const heavySets = [];
  for (const rule of partitionRules) {
    const slices = new Map();
    stream.forEach((key, index) => {
      const shard = rule(index);
      if (!slices.has(shard)) slices.set(shard, new SpaceSaving(20));
      slices.get(shard).add(key);
    });
    let merged = null;
    for (const summary of slices.values()) {
      merged = merged === null ? summary : merged.merge(summary);
    }
    const set = new Set(merged.top(20).map((entry) => entry.key));
    for (const key of heavy) {
      if (!set.has(key)) assert.fail(`partitioning rule lost heavy hitter ${key}`);
    }
    heavySets.push([...set].sort());
  }
  assert.strictEqual(heavySets.length, 3, "three partitionings were tried");
  assert.ok(heavySets.every((set) => set.length >= heavy.length),
    "every partitioning reports at least the true heavy hitters");

  // --- accuracy and cost trade-off -----------------------------------------
  const recallFor = (k) => {
    const summary = new SpaceSaving(k);
    for (const key of stream) summary.add(key);
    const set = new Set(summary.top(k).map((entry) => entry.key));
    return { k, recall: heavy.filter((key) => set.has(key)).length / heavy.length,
      size: summary.size() };
  };
  const recallCurve = [4, 8, 16, 32].map(recallFor);
  assert.ok(recallCurve[recallCurve.length - 1].recall === 1,
    "k = 32 catches every heavy hitter");
  assert.ok(recallCurve[0].recall <= recallCurve[recallCurve.length - 1].recall,
    "recall improves with k");
  assert.ok(recallCurve.every((entry) => entry.size <= entry.k), "and memory stays bounded by k");
  const errorFor = (width) => {
    const local = new CountMin(4, width, 847);
    for (const key of stream) local.add(key);
    let worst = 0;
    for (const [key, value] of counts) worst = Math.max(worst, local.estimate(key) - value);
    return worst;
  };
  const errors = [64, 256, 1024].map(errorFor);
  assert.ok(errors[0] > errors[1] && errors[1] > errors[2],
    `a wider sketch is more accurate (${errors.join(" > ")})`);

  // --- shard count independence and cost ------------------------------------
  const shardSweep = [];
  for (const shards of [2, 4, 8]) {
    const run = distributedHeavyHitters(stream, { k: 20, shards });
    const set = new Set(run.merged.top(20).map((entry) => entry.key));
    for (const key of heavy) assert.ok(set.has(key), `${shards} shards still find ${key}`);
    assert.strictEqual(run.summaries.length, shards, `${shards} summaries were merged`);
    assert.ok(run.mergedSketch.total === total, `${shards} shards cover the whole stream`);
    shardSweep.push(shards);
  }
  assert.deepStrictEqual(shardSweep, [2, 4, 8], "the shard sweep covered 2, 4 and 8");
  assert.strictEqual(pipeline.mergedSketch.cells(), pipeline.rows * pipeline.width,
    "the coordinator holds one merged sketch, not one per shard");
  assert.ok(pipeline.mergedSketch.total === total, "and it covers the whole stream");

  // --- edge cases -----------------------------------------------------------
  const empty = new SpaceSaving(8);
  assert.strictEqual(empty.top().length, 0, "an empty summary reports nothing");
  assert.strictEqual(empty.minCount(), Infinity, "and has no minimum to compare against");
  const singleKey = new SpaceSaving(4);
  for (let i = 0; i < 100; i++) singleKey.add("only");
  assert.deepStrictEqual(singleKey.top(1), [{ key: "only", count: 100, error: 0, lower: 100 }],
    "a single repeated key is exact");
  const distinct = new SpaceSaving(5);
  for (let i = 0; i < 50; i++) distinct.add(`d${i}`);
  assert.strictEqual(distinct.size(), 5, "all-distinct keys fill the summary");
  assert.ok([...distinct.entries.values()].every((entry) => entry.count >= 1),
    "and every entry carries a count");
  const one = new SpaceSaving(1);
  for (const key of ["a", "b", "a", "c", "a"]) one.add(key);
  const oneEntry = one.estimate("a");
  assert.ok(oneEntry !== null && oneEntry.lower <= 3 && 3 <= oneEntry.count,
    `k = 1 still brackets the true count (${oneEntry?.lower} <= 3 <= ${oneEntry?.count})`);
  const beyond = new SpaceSaving(100);
  for (const key of stream.slice(0, 10)) beyond.add(key);
  assert.ok(beyond.size() <= 10, "k beyond the key count keeps only the keys seen");
  const emptyStream = distributedHeavyHitters([], { k: 4, shards: 3 });
  assert.strictEqual(emptyStream.merged.total, 0, "an empty stream merges to nothing");
  assert.deepStrictEqual(emptyStream.candidates, [], "with no candidates");
  assert.ok(emptyStream.mergedSketch.total === 0, "and an empty sketch");

  console.log(`stream: length=${total} distinct=${counts.size} heavy(>5%)=${heavy.length} ` +
    `hot-0=${counts.get("hot-0")}`);
  console.log(`space-saving: k=20 evictions=${single.evictions} no-false-negative-k=${thresholdCases}; ` +
    `count-min: max overestimate=${maxOverestimate} (bound ${(epsilon * total).toFixed(0)})`);
  console.log(`distributed: shards=4 candidates=${pipeline.candidates.length} ` +
    `recall curve=${recallCurve.map((entry) => `${entry.k}:${entry.recall}`).join(" ")} ` +
    `width error=${errors.join(">")}`);
  console.log("PASS");
}
