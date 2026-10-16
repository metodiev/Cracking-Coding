/** Task 911 - Implement probabilistic data structure for heavy hitters with error guarantees and mergeability.
 *  (Tier: Master, CTCI Beyond — Streaming Algorithms & Sketches)
 *
 * Two summaries of a stream, both with provable guarantees and both mergeable across shards:
 *
 *   Count-Min  width w = ceil(e/eps), depth d = ceil(ln(1/delta)); h_i(x) = (a_i*h(x) + b_i) mod p mod w.
 *              est(x) = min_i C[i][x_i] satisfies true(x) <= est(x) <= true(x) + eps*N. Because a standard
 *              update is *linear* in the counters, merging two sketches is counter-wise addition, so
 *              merge(A,B) equals a single pass over the concatenated stream exactly.
 *   Misra-Gries  k counters; every item with true count > N/(k+1) survives in the summary and every
 *              reported count is within N/(k+1) of the truth. Merging two summaries adds the common keys
 *              and subtracts min(m_A, m_B) from the rest, which keeps the same error bound for the union.
 *
 * Checks: the Count-Min error bound against exact counts, merge = single pass (and commutativity and
 * associativity of the merge), conservative update never exceeding the standard one, Misra-Gries recall
 * and error bounds, the merged summary still finding the union's heavy hitters, spot guarantees
 * (empty stream, eps = 1/2, k = 1, all-distinct stream), fixed memory regardless of stream length, and
 * reproducibility. Counts are exact integers; the only randomness is which items the seeded stream draws.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  int(bound) { return this.next() % bound; }
  float() { return this.next() / 4294967296; }
}
const PRIME = 2147483647;
/** FNV-1a over the string form of an item: the shared hash the sketch rows are derived from. */
function hashItem(item) {
  const text = String(item);
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
}

class CountMinSketch {
  /** mode: "standard" (linear, exactly mergeable) or "conservative" (tighter, not linear). */
  constructor(epsilon, delta, seed, mode = "standard") {
    assert.ok(epsilon > 0 && epsilon < 1, `epsilon must lie in (0,1), got ${epsilon}`);
    assert.ok(delta > 0 && delta < 1, `delta must lie in (0,1), got ${delta}`);
    assert.ok(mode === "standard" || mode === "conservative", `unknown mode ${mode}`);
    this.width = Math.ceil(Math.E / epsilon);
    this.depth = Math.ceil(Math.log(1 / delta));
    this.mode = mode;
    const prng = new Prng(seed);
    this.rows = Array.from({ length: this.depth }, () => ({ a: prng.int(PRIME - 1) + 1, b: prng.int(PRIME - 1) + 1 }));
    this.counters = Array.from({ length: this.depth }, () => new Array(this.width).fill(0));
    this.seen = 0;
  }
  columns(item) {
    const hash = hashItem(item);
    return this.rows.map(({ a, b }) => ((a * hash + b) % PRIME) % this.width);
  }
  update(item, amount = 1) {
    const columns = this.columns(item);
    if (this.mode === "conservative") {
      const estimate = this.query(item);
      const target = estimate + amount;
      this.counters.forEach((row, i) => { row[columns[i]] = Math.max(row[columns[i]], target); });
    } else {
      this.counters.forEach((row, i) => { row[columns[i]] += amount; });
    }
    this.seen += amount;
    return this;
  }
  query(item) {
    const columns = this.columns(item);
    return this.counters.reduce((worst, row, i) => Math.min(worst, row[columns[i]]), Infinity);
  }
  /** Counter-wise sum — exact for the linear update, so merge(A,B) == one pass over A ++ B. */
  merge(other) {
    assert.strictEqual(other.width, this.width, "sketches must share a width");
    assert.strictEqual(other.depth, this.depth, "sketches must share a depth");
    assert.strictEqual(this.mode, other.mode, "only sketches with the same update rule merge linearly");
    assert.deepStrictEqual(other.rows, this.rows, "sketches must share the hash rows");
    const merged = Object.create(CountMinSketch.prototype);
    merged.width = this.width;
    merged.depth = this.depth;
    merged.mode = this.mode;
    merged.rows = this.rows;
    merged.counters = this.counters.map((row, i) => row.map((value, j) => value + other.counters[i][j]));
    merged.seen = this.seen + other.seen;
    return merged;
  }
}
/** Misra-Gries summary of at most k counters over a stream. */
class MisraGries {
  constructor(capacity) {
    assert.ok(Number.isInteger(capacity) && capacity > 0, "capacity must be a positive integer");
    this.capacity = capacity;
    this.counters = new Map();
    this.seen = 0;
  }
  update(item, amount = 1) {
    this.seen += amount;
    const current = this.counters.get(item) ?? 0;
    if (current > 0) {
      this.counters.set(item, current + amount);
      return this;
    }
    if (this.counters.size < this.capacity) {
      this.counters.set(item, amount);
      return this;
    }
    for (const [key, value] of this.counters) {
      const next = value - amount;
      if (next <= 0) this.counters.delete(key); else this.counters.set(key, next);
    }
    return this;
  }
  get error() { return Math.floor(this.seen / (this.capacity + 1)); }
  estimate(item) { return this.counters.get(item) ?? 0; }
  /** Merge two summaries of disjoint shards: add common keys, subtract min(m_A, m_B) from the rest. */
  merge(other) {
    const merged = new MisraGries(this.capacity);
    const floor = Math.min(minCounter(this), minCounter(other));
    const combined = new Map();
    for (const [key, value] of this.counters) combined.set(key, value);
    for (const [key, value] of other.counters) combined.set(key, (combined.get(key) ?? 0) + value);
    for (const [key, value] of combined) {
      const adjusted = value - floor;
      if (adjusted > 0) merged.counters.set(key, adjusted);
    }
    merged.seen = this.seen + other.seen;
    const trimmed = [...merged.counters].sort((a, b) => b[1] - a[1]).slice(0, this.capacity);
    merged.counters = new Map(trimmed);
    return merged;
  }
}
const minCounter = (summary) => (summary.counters.size === 0 ? 0 : Math.min(...summary.counters.values()));
/** Exact counts of a stream (the oracle). */
function exactCounts(stream) {
  const counts = new Map();
  for (const item of stream) counts.set(item, (counts.get(item) ?? 0) + 1);
  return counts;
}
const heavyHitters = (counts, threshold) =>
  [...counts].filter(([, count]) => count > threshold).map(([item]) => item).sort();
const allItems = (stream) => [...new Set(stream)].sort();

function main() {
  const epsilon = 0.02;
  const delta = 0.01;
  const seed = 0x5eed0911;
  const stream = (() => {
    const prng = new Prng(seed);
    const items = [];
    for (let i = 0; i < 20000; i++) items.push(`key-${Math.floor(Math.pow(prng.float(), 3) * 120)}`);
    return items;
  })();
  const counts = exactCounts(stream);
  const sketch = new CountMinSketch(epsilon, delta, 7);
  for (const item of stream) sketch.update(item);
  // 1. the Count-Min guarantee — never an underestimate, and overestimated by at most eps*N.
  {
    let worstOver = 0;
    for (const [item, truth] of counts) {
      const estimate = sketch.query(item);
      assert.ok(estimate >= truth, `est ${estimate} below true ${truth} for ${item}`);
      worstOver = Math.max(worstOver, estimate - truth);
    }
    assert.ok(worstOver <= epsilon * stream.length, `worst overestimate ${worstOver} > eps*N`);
    assert.strictEqual(sketch.seen, stream.length, "the sketch counts the whole stream");
    assert.ok(sketch.width >= Math.ceil(Math.E / epsilon) && sketch.depth >= 1, "sized from eps and delta");
  }
  // 2. mergeability — merging two sketches equals one pass over the concatenation, exactly.
  {
    const left = new CountMinSketch(epsilon, delta, 7);
    const right = new CountMinSketch(epsilon, delta, 7);
    for (let i = 0; i < stream.length; i++) (i % 3 === 0 ? left : right).update(stream[i]);
    const merged = left.merge(right);
    assert.deepStrictEqual(merged.counters, sketch.counters, "counter-wise merge matches a single pass");
    assert.strictEqual(merged.seen, sketch.seen);
    for (const item of allItems(stream)) assert.strictEqual(merged.query(item), sketch.query(item));
    assert.deepStrictEqual(right.merge(left).counters, merged.counters, "the merge is commutative");
    const third = new CountMinSketch(epsilon, delta, 7);
    for (let i = 0; i < stream.length; i += 7) third.update(stream[i]);
    const assoc = left.merge(right).merge(third);
    const otherOrder = left.merge(right.merge(third));
    assert.deepStrictEqual(assoc.counters, otherOrder.counters, "the merge is associative");
    assert.ok(assoc.query(allItems(stream)[0]) >= third.query(allItems(stream)[0]), "merging only adds evidence");
  }
  // 3. conservative update — tighter, still never an underestimate, never above the linear estimate.
  {
    const conservative = new CountMinSketch(epsilon, delta, 7, "conservative");
    for (const item of stream) conservative.update(item);
    let totalStandard = 0;
    let totalConservative = 0;
    let tighter = 0;
    for (const [item, truth] of counts) {
      const estimate = conservative.query(item);
      assert.ok(estimate >= truth, "conservative update still never underestimates");
      assert.ok(estimate <= sketch.query(item), "conservative update never exceeds the standard update");
      totalStandard += sketch.query(item) - truth;
      totalConservative += estimate - truth;
      if (estimate < sketch.query(item)) tighter += 1;
    }
    assert.ok(totalConservative < totalStandard / 3, `conservative total overestimate ${totalConservative} vs ${totalStandard}`);
    assert.ok(tighter >= 10, `conservative update tightened only ${tighter} of ${counts.size} items`);
    assert.throws(() => sketch.merge(conservative), /same update rule/);
    assert.throws(() => sketch.merge(new CountMinSketch(epsilon, delta, 8)), /share the hash rows/);
  }
  // 4. Misra-Gries — recalls every heavy hitter and bounds the error by N/(k+1).
  {
    const capacity = 40;
    const summary = new MisraGries(capacity);
    for (const item of stream) summary.update(item);
    assert.strictEqual(summary.error, Math.floor(stream.length / (capacity + 1)));
    const trueHeavy = heavyHitters(counts, summary.error);
    for (const item of trueHeavy) {
      assert.ok(summary.counters.has(item), `${item} with count ${counts.get(item)} must be retained`);
      assert.ok(summary.estimate(item) <= counts.get(item), "the estimate never exceeds the truth");
      assert.ok(counts.get(item) - summary.estimate(item) <= summary.error, `error bound for ${item}`);
    }
    assert.ok(summary.counters.size <= capacity, "the summary respects its capacity");
    assert.ok(trueHeavy.length > 3, "the stream really has heavy hitters");
  }
  // 5. merging summaries of two shards keeps the guarantee for the union.
  {
    const capacity = 40;
    const shardA = new MisraGries(capacity);
    const shardB = new MisraGries(capacity);
    const split = Math.floor(stream.length / 2);
    for (let i = 0; i < split; i++) shardA.update(stream[i]);
    for (let i = split; i < stream.length; i++) shardB.update(stream[i]);
    const merged = shardA.merge(shardB);
    const singlePass = new MisraGries(capacity);
    for (const item of stream) singlePass.update(item);
    const threshold = Math.floor(stream.length / (capacity + 1));
    const trueHeavy = heavyHitters(counts, threshold);
    for (const item of trueHeavy) {
      assert.ok(merged.counters.has(item), `merged summary lost heavy hitter ${item}`);
      assert.ok(counts.get(item) - merged.estimate(item) <= 2 * threshold, `merged error for ${item}: ${counts.get(item) - merged.estimate(item)} > ${2 * threshold}`);
    }
    const singleHeavy = [...singlePass.counters.keys()].sort();
    assert.ok(trueHeavy.every((item) => singleHeavy.includes(item)), "the single-pass summary agrees");
    assert.ok(merged.counters.size <= capacity, "the merged summary respects capacity");
  }
  // 6. boundary — an empty stream, eps = 1/2, k = 1, and a stream with no heavy hitters.
  {
    const empty = new CountMinSketch(0.5, 0.5, 3);
    assert.strictEqual(empty.query("anything"), 0, "an untouched sketch answers zero");
    assert.strictEqual(empty.seen, 0);
    assert.strictEqual(empty.merge(new CountMinSketch(0.5, 0.5, 3)).query("x"), 0, "merging two empty sketches");
    assert.strictEqual(new MisraGries(1).update("solo").estimate("solo"), 1, "capacity one keeps one item");
    const distinct = Array.from({ length: 500 }, (_, i) => `unique-${i}`);
    const sparse = new MisraGries(5);
    for (const item of distinct) sparse.update(item);
    assert.deepStrictEqual(heavyHitters(exactCounts(distinct), sparse.error), [], "no heavy hitters exist");
    assert.ok(sparse.counters.size <= 5, "capacity respected on a flat stream");
    assert.throws(() => new CountMinSketch(0, 0.5, 1), /epsilon must lie/);
    assert.throws(() => new MisraGries(0), /positive integer/);
    assert.throws(() => new CountMinSketch(0.1, 0.1, 1, "wishful"), /unknown mode/);
  }
  // 7. fixed memory — the sketch size does not grow with the stream.
  {
    const short = new CountMinSketch(epsilon, delta, 11);
    const long = new CountMinSketch(epsilon, delta, 11);
    for (let i = 0; i < 100; i++) short.update(`item-${i}`);
    for (let i = 0; i < 100000; i++) long.update(`item-${i % 1000}`);
    assert.strictEqual(short.counters.length, long.counters.length, "depth is independent of the stream");
    assert.strictEqual(short.counters[0].length, long.counters[0].length, "width is independent too");
    assert.ok(long.seen > 100 * short.seen, "the long stream really is longer");
    assert.ok(short.counters.flat().reduce((a, b) => a + b, 0) < long.counters.flat().reduce((a, b) => a + b, 0), "the merged evidence grows with the stream");
  }
  // 8. determinism — identical seeds give identical structure and answers.
  {
    const replay = new CountMinSketch(epsilon, delta, 7);
    for (const item of stream) replay.update(item);
    assert.deepStrictEqual(replay.counters, sketch.counters);
    assert.deepStrictEqual(replay.rows, sketch.rows);
    const other = new CountMinSketch(epsilon, delta, 8);
    assert.notDeepStrictEqual(other.rows, sketch.rows, "a different seed hashes differently");
    assert.deepStrictEqual(exactCounts(stream), counts, "the stream is reproducible");
  }
  const worst = Math.max(...[...counts].map(([item, truth]) => sketch.query(item) - truth));
  console.log(`sketches: ${stream.length} events over ${counts.size} keys, Count-Min ${sketch.width}x${sketch.depth}` + ` (eps=${epsilon}), worst overestimate ${worst} <= ${(epsilon * stream.length).toFixed(1)}; ` +
    `Misra-Gries recalls all ${heavyHitters(counts, Math.floor(stream.length / 41)).length} heavy hitters ` + `and merges across two shards`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  Prng, hashItem, CountMinSketch, MisraGries, exactCounts, heavyHitters, minCounter, };
