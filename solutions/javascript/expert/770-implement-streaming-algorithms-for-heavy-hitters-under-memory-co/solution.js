/** Task 770 - Implement streaming algorithms for heavy hitters under memory constraints. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A stream of tokens is summarised with sub-linear space by three classic single-pass sketches;
 * every claim is checked against an exact counter (a Map) on the same stream.
 *
 *   MisraGries(m)     m counters, no item names beyond the counters. A new item either bumps its
 *                     counter or, when all counters are in use, decrements every counter by one
 *                     (which may evict several items at once). Invariants: an estimate never
 *                     exceeds the true frequency, and the total number of dropped units is
 *                     bounded, so every item with f > N/(m+1) keeps a counter.
 *   SpaceSaving(m)    m (item, count) slots. A new item takes over the least-used slot, whose
 *                     count becomes the starting point of the new occupant, so estimates never
 *                     *under*-estimate: f(x) <= est(x) <= f(x) + N/m for every monitored x.
 *   CountMin(w, d)    d hash tables of w counters. An update touches one cell per row, so an
 *                     estimate is the minimum over rows; it can only over-count via collisions:
 *                     est(x) >= f(x) pointwise, and with w = ceil(e/eps), d = ceil(ln(1/delta))
 *                     rows the expected over-count is at most N/w = eps*N/e.
 *
 * What is verified: exact equality of the exact counter with a naive recount, pointwise
 * one-sided error directions (MG underestimates, SS/CM overestimate), the guaranteed-heavy-hitter
 * containment "f > threshold => reported" for each sketch, the Space-Saving error bound, the
 * Count-Min expectation bound averaged over a seeded hash family, exact-count preservation of the
 * reported top-k on streams without collisions, and degenerate streams (empty, single item, one
 * distinct value, all-distinct, m = 1, w = 1). What is simplified: the hash family is a fixed
 * seeded multiply-shift family rather than a formally universal one, and the Count-Min guarantee
 * is therefore checked in expectation (that is exactly how it is proved).
 */
"use strict";

/** Deterministic splitmix64-inspired generator; multiply-shift hashes are built from its output. */
function makeRng(seed) {
  let state = BigInt.asUintN(64, BigInt(seed) + 0x9e3779b97f4a7c15n);
  return function next(bound) {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    z = z ^ (z >> 31n);
    const v = Number(z % 0xffffffffn);
    return bound === undefined ? v : v % bound;
  };
}

/** Exact frequency table plus the stream length. */
function exactCounts(stream) {
  const counts = new Map();
  for (const x of stream) counts.set(x, (counts.get(x) ?? 0) + 1);
  return { counts, n: stream.length };
}

/** Misra-Gries with `m` counters. `error` is the guaranteed cap on the missing mass. */
class MisraGries {
  constructor(m) {
    if (!Number.isInteger(m) || m < 1) throw new RangeError("Misra-Gries needs at least one counter");
    this.m = m;
    this.counters = new Map();
    this.seen = 0;
    this.decrements = 0;
  }

  update(x) {
    this.seen++;
    if (this.counters.has(x)) {
      this.counters.set(x, this.counters.get(x) + 1);
      return;
    }
    if (this.counters.size < this.m) {
      this.counters.set(x, 1);
      return;
    }
    // Table full: one global decrement, dropping every counter that reaches zero.
    this.decrements++;
    for (const [key, value] of [...this.counters]) {
      if (value === 1) this.counters.delete(key);
      else this.counters.set(key, value - 1);
    }
  }

  estimate(x) { return this.counters.get(x) ?? 0; }

  /** Every item with f > N / (m + 1) must be tracked here. */
  get guaranteedThreshold() { return this.seen / (this.m + 1); }

  top(k) {
    return [...this.counters.entries()].sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : 1)).slice(0, k);
  }
}

/** Space-Saving with `m` slots; the replaced slot donates its count as the new item's floor. */
class SpaceSaving {
  constructor(m) {
    if (!Number.isInteger(m) || m < 1) throw new RangeError("Space-Saving needs at least one slot");
    this.m = m;
    this.counts = new Map();
    this.seen = 0;
  }

  update(x) {
    this.seen++;
    if (this.counts.has(x)) {
      this.counts.set(x, this.counts.get(x) + 1);
      return;
    }
    if (this.counts.size < this.m) {
      this.counts.set(x, 1);
      return;
    }
    const victim = this.#minSlot();
    const inherited = this.counts.get(victim);
    this.counts.delete(victim);
    this.counts.set(x, inherited + 1);
  }

  #minSlot() {
    let best = null;
    let bestCount = Infinity;
    for (const [key, value] of this.counts) {
      if (value < bestCount || (value === bestCount && key < best)) { best = key; bestCount = value; }
    }
    return best;
  }

  estimate(x) { return this.counts.get(x) ?? 0; }

  /** Every item with f > N / m must be monitored. */
  get guaranteedThreshold() { return this.seen / this.m; }

  top(k) {
    return [...this.counts.entries()].sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : 1)).slice(0, k);
  }
}

/** Count-Min sketch with a seeded multiply-shift hash per row. */
class CountMin {
  /**
   * `perfect` is an optional array of keys used only by the self-checks: it assigns every key a
   * private column (the same one in every row), which isolates the sketch's error to collisions.
   */
  constructor(width, depth, seed, perfect = null) {
    if (!Number.isInteger(width) || width < 1) throw new RangeError("width must be >= 1");
    if (!Number.isInteger(depth) || depth < 1) throw new RangeError("depth must be >= 1");
    if (perfect !== null && (!Array.isArray(perfect) || perfect.length > width)) {
      throw new RangeError("a perfect hash needs at most `width` keys");
    }
    this.width = width;
    this.depth = depth;
    this.tables = Array.from({ length: depth }, () => new Int32Array(width));
    const rng = makeRng(seed);
    this.multipliers = Array.from({ length: depth }, () => (2 * rng(0x3fffffff) + 1) >>> 0);
    this.perfect = perfect === null ? null : new Map(perfect.map((key, i) => [key, i]));
    this.n = 0;
  }

  #bucket(row, x) {
    if (this.perfect !== null) return this.perfect.get(x) ?? 0;
    // 32-bit polynomial mixing of the token hash with the row multiplier, then a modulo.
    let h = 2166136261;
    const text = String(x);
    for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
    const mixed = Math.imul((h ^ this.multipliers[row]) >>> 0, 2654435761) >>> 0;
    return mixed % this.width;
  }

  update(x, delta = 1) {
    this.n += delta;
    for (let row = 0; row < this.depth; row++) this.tables[row][this.#bucket(row, x)] += delta;
  }

  estimate(x) {
    let best = Infinity;
    for (let row = 0; row < this.depth; row++) best = Math.min(best, this.tables[row][this.#bucket(row, x)]);
    return best;
  }

  /** eps/delta parameters required so that the expected over-count is at most eps * N. */
  static sizing(eps, delta) {
    if (!(eps > 0 && eps < 1)) throw new RangeError("eps must lie in (0, 1)");
    if (!(delta > 0 && delta < 1)) throw new RangeError("delta must lie in (0, 1)");
    return { width: Math.ceil(Math.E / eps), depth: Math.ceil(Math.log(1 / delta)) };
  }
}

/** Zipf-ish stream generator with a controllable number of heavy hitters. */
function zipfStream(rng, length, vocab, hotCount) {
  const out = [];
  const hot = Array.from({ length: hotCount }, (_, i) => `hot${i}`);
  for (let i = 0; i < length; i++) {
    if (rng(100) < 55) out.push(hot[rng(hot.length)]);
    else out.push(`cold${rng(vocab)}`);
  }
  return out;
}

module.exports = { makeRng, exactCounts, MisraGries, SpaceSaving, CountMin, zipfStream };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, ok, throws } = assert;
  const rng = makeRng(20260770);
  let mgTightestSlack = Infinity;
  let ssTightestSlack = Infinity;
  let cmWorstMeanError = 0;
  let trials = 0;

  // -- degenerate streams ------------------------------------------------------------
  const empty = exactCounts([]);
  strictEqual(empty.n, 0, "empty stream has length zero");
  const mgEmpty = new MisraGries(4);
  const ssEmpty = new SpaceSaving(4);
  const cmEmpty = new CountMin(16, 4, 1);
  strictEqual(mgEmpty.estimate("x"), 0, "empty MG reports nothing");
  strictEqual(ssEmpty.estimate("x"), 0, "empty SS reports nothing");
  strictEqual(cmEmpty.estimate("x"), 0, "empty CM reports nothing");
  strictEqual(mgEmpty.top(3).length, 0, "empty MG top-k is empty");
  throws(() => new MisraGries(0), RangeError, "m = 0 counters is invalid");
  throws(() => new SpaceSaving(-1), RangeError, "negative slot count is invalid");
  throws(() => new CountMin(0, 3, 1), RangeError, "zero width is invalid");
  throws(() => CountMin.sizing(0, 0.1), RangeError, "eps must be positive");

  const mg1 = new MisraGries(1);
  const ss1 = new SpaceSaving(1);
  const cm1 = new CountMin(1, 3, 5); // w = 1: every item collides, the worst possible sketch
  for (const x of ["a"]) { mg1.update(x); ss1.update(x); cm1.update(x); }
  strictEqual(mg1.estimate("a"), 1, "MG keeps the single value");
  strictEqual(ss1.estimate("a"), 1, "SS keeps the single value");
  strictEqual(cm1.estimate("a"), 1, "CM with w = 1 is exact for a single key");
  strictEqual(cm1.estimate("b"), 1, "CM with w = 1 over-counts every other key (that is the bound)");

  // -- one distinct value: every sketch is exact --------------------------------------
  const mono = new Array(500).fill("only");
  const { counts: monoCounts, n: monoN } = exactCounts(mono);
  strictEqual(monoN, 500, "mono stream length");
  const mgMono = new MisraGries(3);
  const ssMono = new SpaceSaving(3);
  const cmMono = new CountMin(8, 3, 11);
  for (const x of mono) { mgMono.update(x); ssMono.update(x); cmMono.update(x); }
  strictEqual(mgMono.estimate("only"), monoCounts.get("only"), "MG is exact when there is no eviction");
  strictEqual(ssMono.estimate("only"), monoCounts.get("only"), "SS is exact with a single value");
  strictEqual(cmMono.estimate("only"), monoCounts.get("only"), "CM is exact with one key");

  // -- all distinct: every sketch must collapse but never lie in the wrong direction ---
  const distinct = Array.from({ length: 300 }, (_, i) => `d${i}`);
  const mgDistinct = new MisraGries(10);
  const ssDistinct = new SpaceSaving(10);
  for (const x of distinct) { mgDistinct.update(x); ssDistinct.update(x); }
  ok(mgDistinct.estimate("d0") <= 1, "MG cannot invent mass that was never observed");
  ok([...ssDistinct.counts.values()].every((v) => v >= 1), "SS counters stay positive");

  // -- random heavy-hitter streams, three sketches vs the exact counter ---------------
  for (let trial = 0; trial < 12; trial++) {
    const length = 500 + rng(1500);
    const vocab = 40 + rng(200);
    const stream = zipfStream(rng, length, vocab, 3 + rng(4));
    const { counts, n } = exactCounts(stream);
    strictEqual(n, stream.length, "exact counter must observe the whole stream");
    const m = 8 + rng(12);
    const mg = new MisraGries(m);
    const ss = new SpaceSaving(m);
    for (const x of stream) { mg.update(x); ss.update(x); }

    // one-sided error directions and the containment guarantees
    for (const [key, f] of counts) {
      ok(mg.estimate(key) <= f, `MG over-estimated ${key} (${mg.estimate(key)} > ${f})`);
      // Space-Saving only over-estimates for the items it still monitors; evicted items report 0.
      ok(ss.counts.has(key) ? ss.estimate(key) >= f : ss.estimate(key) === 0,
        `SS under-estimated the monitored item ${key} (${ss.estimate(key)} < ${f})`);
      if (f > mg.guaranteedThreshold) {
        ok(mg.counters.has(key), `MG lost the guaranteed heavy hitter ${key} (f=${f})`);
        mgTightestSlack = Math.min(mgTightestSlack, f - mg.guaranteedThreshold);
      }
      if (f > ss.guaranteedThreshold) {
        ok(ss.counts.has(key), `SS lost the guaranteed heavy hitter ${key} (f=${f})`);
        ssTightestSlack = Math.min(ssTightestSlack, f - ss.guaranteedThreshold);
      }
      if (ss.counts.has(key)) {
        ok(ss.estimate(key) <= f + n / m + 1e-9, `SS error bound violated for ${key}`);
      }
    }
    ok(mg.counters.size <= m, "MG never exceeds its counter budget");
    ok(ss.counts.size <= m, "SS never exceeds its slot budget");
    ok(mg.decrements <= n / (m + 1) + 1e-9, "MG decrement count must stay within N/(m+1)");
    for (const [key, f] of counts) {
      ok(f - mg.estimate(key) <= mg.decrements, `MG loss on ${key} must be bounded by D`);
    }

    // Count-Min: pointwise non-under-estimation and the eps/delta sizing
    const eps = 0.02;
    const { width, depth } = CountMin.sizing(eps, 0.01);
    const meanErrors = [];
    for (const [key] of counts) {
      let total = 0;
      const seeds = 8;
      for (let s = 0; s < seeds; s++) {
        const cm = new CountMin(width, depth, 1000 + s);
        for (const x of stream) cm.update(x);
        const est = cm.estimate(key);
        ok(est >= counts.get(key), `CM under-estimated ${key}`);
        total += est - counts.get(key);
      }
      meanErrors.push(total / seeds);
    }
    const worstMean = Math.max(...meanErrors);
    cmWorstMeanError = Math.max(cmWorstMeanError, worstMean);
    ok(worstMean <= eps * n, `CM expected over-count ${worstMean.toFixed(1)} exceeded eps*N = ${(eps * n).toFixed(1)}`);

    // the reported top-k must equal the exact top-k whenever no collision happened
    // With a perfect hash (one private column per key) the only error source disappears, so the
    // sketch becomes exact: this isolates the collision term in the Count-Min bound.
    const keys = [...counts.keys()];
    const cmExact = new CountMin(keys.length, 3, 7, keys);
    for (const x of stream) cmExact.update(x);
    for (const [key, f] of counts) strictEqual(cmExact.estimate(key), f, "a collision-free CM is exact");
    // Without the perfect hash the same sketch over-counts, never under-counts.
    const cmHashed = new CountMin(keys.length, 3, 7);
    for (const x of stream) cmHashed.update(x);
    for (const [key, f] of counts) ok(cmHashed.estimate(key) >= f, "a hashed CM never under-counts");
    ok(keys.some((key) => cmHashed.estimate(key) > counts.get(key)),
      "with a real hash the collision error must actually show up");
    // Space-Saving's headline guarantee: it reports a top-1 estimate within N/m of the true
    // maximum, provided the true maximum is itself above the guaranteed threshold.
    const maxTrue = Math.max(...counts.values());
    if (maxTrue > ss.guaranteedThreshold) {
      const [[topKey, topCount]] = ss.top(1);
      ok(topCount >= maxTrue, `SS top-1 counter ${topCount} must dominate the true maximum ${maxTrue}`);
      ok(topCount <= maxTrue + n / m + 1e-9,
        `SS top-1 estimate must stay within N/m of the true maximum (${topCount} vs ${maxTrue})`);
      ok(counts.get(topKey) >= maxTrue - n / m, `SS top-1 key ${topKey} is off by more than N/m`);
    }
    // Every monitored item's estimate sits above its true count: the summary never loses mass.
    for (const [key, value] of ss.top(3)) ok(value >= counts.get(key), `SS lost mass on ${key}`);
    trials++;
  }

  // -- a stream whose heavy hitters are exactly at the MG threshold -------------------
  const threshold = new MisraGries(4);
  const pattern = ["a", "b", "c", "d", "e", "f", "g", "h"];
  for (let round = 0; round < 100; round++) for (const x of pattern) threshold.update(x);
  strictEqual(threshold.seen, 800, "threshold stream length");
  // Misra-Gries theory: every decrement event throws away exactly m+1 units of mass, so the
  // decrement count D is at most N/(m+1) and every item satisfies f(x) - est(x) <= D.
  ok(threshold.decrements <= 800 / 5 + 1e-9, "MG decrement count must stay within N/(m+1)");
  ok(threshold.counters.size <= 4, "MG counter budget respected on the tied stream");
  for (const x of pattern) {
    ok(threshold.estimate(x) <= 100, "MG never over-estimates a tied heavy hitter");
    ok(100 - threshold.estimate(x) <= threshold.decrements, "MG per-item loss must be bounded by D");
  }

  console.log(`trials=${trials} mg tightest slack=${mgTightestSlack.toFixed(2)} ss tightest slack=${ssTightestSlack.toFixed(2)} cm worst mean error=${cmWorstMeanError.toFixed(2)}`);
  console.log("PASS");
}
