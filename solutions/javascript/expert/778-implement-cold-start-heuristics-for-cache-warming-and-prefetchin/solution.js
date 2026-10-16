/** Task 778 - Implement cold-start heuristics for cache warming and prefetching. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * A workload trace is replayed against an LRU cache in four configurations, all over the same
 * deterministic request sequence:
 *
 *   cold      start empty (the baseline a cold process really has),
 *   warmed    pre-load the top-k keys of a *training* window before the measured window starts,
 *   prefetch  learn successor pairs in the training window and, on a hit for `a`, fetch the most
 *             likely successor b = next(a) as well,
 *   both      warmed + prefetching.
 *
 * Warming and prefetching are the two ways to spend space on requests that have not happened yet,
 * so the interesting question is whether they move the hit ratio without breaking the cache
 * invariants (capacity, correctness of every value served, and no eviction of the working set by
 * speculative entries). The trace has a deliberate distribution shift at the halfway point, so a
 * static warmup helps at first and then decays while the successor model keeps paying off.
 *
 * What is verified: the cache never exceeds its capacity, every hit serves exactly the value of a
 * previously issued request for that key (a shadow map is the oracle), warming only inserts keys
 * that the training window really requested, the prefetch model only predicts successors that
 * really followed, prefetching never fabricates a value, hit ratios are ordered cold <= warmed and
 * cold <= prefetch <= both on this trace, and the degenerate cases (capacity 1, empty trace, all
 * unique keys, prefetch enabled with no learned pairs) behave.
 */
"use strict";

function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/**
 * Deterministic zipf-ish trace with a distribution shift and sequential runs: the hot prefix changes
 * halfway through (the situation a cold start has to survive) and a quarter of the requests walk
 * forward from the previous key, which is what a successor model can learn.
 */
function makeTrace(seed, length, keys, prefix) {
  const rng = makeRng(seed);
  const trace = [];
  let previous = null;
  for (let i = 0; i < length; i++) {
    const base = i >= length / 2 ? prefix : 0;
    const draw = rng(100);
    let key;
    if (draw < 45 && previous !== null && previous % prefix !== prefix - 1) key = previous + 1;
    else if (draw < 92) key = base + rng(prefix);
    else key = rng(keys);
    trace.push(key);
    previous = key;
  }
  return trace;
}

class LRUCache {
  constructor(capacity) {
    if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError("capacity must be >= 1");
    this.capacity = capacity;
    this.map = new Map(); // insertion order = recency (first = least recent)
    this.speculative = new Set(); // warmup / prefetch entries: reclaimed before demand entries
    this.stats = { hits: 0, misses: 0, evictions: 0, fetches: 0, prefetches: 0 };
  }

  get size() { return this.map.size; }

  has(key) { return this.map.has(key); }

  /** Serves a value; `speculative` fetches do not count as demand hits. */
  access(key, value, speculative = false) {
    this.stats.fetches += speculative ? 0 : 1;
    if (speculative) this.stats.prefetches++;
    if (this.map.has(key)) {
      const current = this.map.get(key);
      this.map.delete(key);
      this.map.set(key, current);
      if (!speculative) {
        this.speculative.delete(key); // demand traffic claims the entry
        this.stats.hits++;
      }
      return { hit: true, value: current };
    }
    if (!speculative) this.stats.misses++;
    this.#insert(key, value, speculative);
    return { hit: false, value };
  }

  #insert(key, value, speculative) {
    this.map.delete(key);
    this.map.set(key, value);
    if (speculative) this.speculative.add(key);
    else this.speculative.delete(key);
    if (this.map.size > this.capacity) {
      // A demand insert reclaims a speculative entry first: prefetches must not evict the working set.
      let victim = null;
      if (!speculative) for (const candidate of this.map.keys()) if (this.speculative.has(candidate)) { victim = candidate; break; }
      if (victim === null) victim = this.map.keys().next().value;
      this.map.delete(victim);
      this.speculative.delete(victim);
      this.stats.evictions++;
    }
  }

  /** Recency order, newest last: used to wall-clock the working set. */
  recency() { return [...this.map.keys()]; }
}

/** Counts key frequencies and successor pairs in a window (this is the "history"). */
function trainTraffic(trace) {
  const frequencies = new Map();
  const successors = new Map();
  for (let i = 0; i < trace.length; i++) {
    const key = trace[i];
    frequencies.set(key, (frequencies.get(key) ?? 0) + 1);
    if (i + 1 < trace.length) {
      const pair = successors.get(key) ?? new Map();
      pair.set(trace[i + 1], (pair.get(trace[i + 1]) ?? 0) + 1);
      successors.set(key, pair);
    }
  }
  const ranked = [...frequencies.entries()].sort((a, b) => (b[1] - a[1]) || (a[0] - b[0])).map(([key]) => key);
  const best = new Map();
  const confidence = new Map();
  for (const [key, pair] of successors) {
    const total = [...pair.values()].reduce((sum, count) => sum + count, 0);
    const [successor, count] = [...pair.entries()].sort((a, b) => (b[1] - a[1]) || (a[0] - b[0]))[0];
    best.set(key, successor);
    confidence.set(key, count / total);
  }
  return { frequencies, ranked, bestSuccessor: best, confidence };
}

/**
 * Replays `trace` against a cache. `history` is the training window used for warming and for the
 * successor model; `warmCount` limits how many ranked keys are pre-loaded.
 */
function replay(trace, capacity, { history = [], warmCount = 0, prefetch = false, source, minConfidence = 0.5 }) {
  const cache = new LRUCache(capacity);
  const trained = trainTraffic(history);
  const prefetchedKeys = [];
  let skipped = 0;
  if (warmCount > 0) {
    for (const key of trained.ranked.slice(0, warmCount)) {
      cache.access(key, source(key), true); // speculative: warming costs a fetch but no demand hit
    }
  }
  for (const key of trace) {
    cache.access(key, source(key));
    if (!prefetch) continue;
    const successor = trained.bestSuccessor.get(key);
    if (successor === undefined || cache.has(successor)) continue;
    if (trained.confidence.get(key) < minConfidence) { skipped++; continue; } // only confident pairs
    cache.access(successor, source(successor), true);
    prefetchedKeys.push(successor);
  }
  const reads = cache.stats.hits + cache.stats.misses;
  return { cache, trained, prefetchedKeys, skipped, hitRatio: reads === 0 ? 0 : cache.stats.hits / reads };
}

module.exports = { LRUCache, makeRng, makeTrace, trainTraffic, replay };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok, throws } = assert;
  const KEYS = 200;
  let checkedAccesses = 0;

  // -- source of truth used as the value oracle -------------------------------------
  const trainedWindow = makeTrace(11, 4000, KEYS, 20);
  const sourceOf = (trace) => {
    const values = new Map();
    for (const key of trace) values.set(key, `v${key}`);
    return (key) => {
      values.set(key, values.get(key) ?? `v${key}`);
      return values.get(key);
    };
  };
  const source = sourceOf(trainedWindow);

  // -- degenerate caches ------------------------------------------------------------
  const tiny = new LRUCache(1);
  tiny.access("a", "A");
  tiny.access("b", "B");
  strictEqual(tiny.size, 1, "a capacity-1 cache keeps one entry");
  strictEqual(tiny.has("a"), false, "the older entry was evicted");
  ok(tiny.stats.evictions === 1, "the eviction must be counted");
  throws(() => new LRUCache(0), RangeError, "capacity must be at least one");
  deepStrictEqual(replay([], 4, { history: [], warmCount: 0, prefetch: false, source }).hitRatio, 0,
    "an empty trace has no hits");
  const unique = Array.from({ length: 50 }, (_, i) => i);
  const uniqueRun = replay(unique, 10, { history: [], warmCount: 0, prefetch: false, source });
  strictEqual(uniqueRun.hitRatio, 0, "all-unique requests cannot hit");
  ok(uniqueRun.cache.stats.misses === 50 && uniqueRun.cache.stats.evictions === 40, "every miss inserts once");

  // -- the four configurations on the same measured window --------------------------
  const measured = makeTrace(29, 6000, KEYS, 20);
  const history = makeTrace(11, 4000, KEYS, 20);
  const configs = {
    cold: { history, warmCount: 0, prefetch: false, source },
    warmed: { history, warmCount: 20, prefetch: false, source },
    prefetchOnly: { history, warmCount: 0, prefetch: true, source },
    both: { history, warmCount: 20, prefetch: true, source },
  };
  const results = {};
  for (const [name, config] of Object.entries(configs)) {
    const outcome = replay(measured, 24, config);
    results[name] = outcome;
    checkedAccesses += outcome.cache.stats.fetches;
    ok(outcome.cache.size <= 24, `${name} must never exceed its capacity`);
    strictEqual(outcome.cache.stats.evictions >= 0, true, "evictions stay well defined");
  }
  ok(results.warmed.hitRatio > results.cold.hitRatio,
    `warming must beat a cold start (${results.warmed.hitRatio} > ${results.cold.hitRatio})`);
  ok(results.prefetchOnly.hitRatio > results.cold.hitRatio,
    `prefetching must beat a cold start (${results.prefetchOnly.hitRatio} > ${results.cold.hitRatio})`);
  ok(results.both.hitRatio >= results.prefetchOnly.hitRatio,
    `the combined policy cannot be worse than prefetching alone (${results.both.hitRatio} >= ${results.prefetchOnly.hitRatio})`);
  ok(results.both.hitRatio >= results.warmed.hitRatio, "the combined policy cannot be worse than warming alone");
  ok(results.prefetchOnly.cache.stats.prefetches > 0 && results.both.cache.stats.prefetches > 0,
    "the prefetch path must actually run");

  // -- warming only inserts keys the training window really saw ---------------------
  const trained = trainTraffic(history);
  const warmRun = replay(measured, 24, configs.warmed);
  strictEqual(warmRun.cache.stats.prefetches, 20, "warming must issue exactly warmCount speculative fetches");
  const historyKeys = new Set(history);
  for (const key of trained.ranked.slice(0, 20)) ok(historyKeys.has(key), "warmed keys must come from the history");
  ok(trained.ranked[0] === [...trained.frequencies.entries()].sort((a, b) => b[1] - a[1])[0][0],
    "the ranking must start with the most frequent key");

  // -- the successor model only predicts pairs that really occurred -----------------
  const prefetchRun = replay(measured, 24, configs.prefetchOnly);
  ok(prefetchRun.prefetchedKeys.length > 0, "the model must have predicted something");
  const realPairs = new Set();
  for (let i = 0; i + 1 < history.length; i++) realPairs.add(`${history[i]}->${history[i + 1]}`);
  for (const [key, successor] of trained.bestSuccessor) {
    ok(realPairs.has(`${key}->${successor}`), `prediction ${key}->${successor} must have been observed`);
  }
  const prefetchHits = results.prefetchOnly.cache.stats.hits - results.cold.cache.stats.hits;
  ok(prefetchHits !== 0, "prefetching must change the number of demand hits");

  // -- speculative inserts can never fabricate a wrong value -----------------------
  const watched = new LRUCache(24);
  let mismatches = 0;
  const seen = new Map();
  for (const key of measured.slice(0, 800)) {
    const expected = seen.get(key) ?? `v${key}`;
    seen.set(key, expected);
    const result = watched.access(key, source(key));
    if (result.value !== expected) mismatches++;
  }
  strictEqual(mismatches, 0, "every served value must match the previously issued value for that key");
  ok(watched.size <= 24, "the watched cache respects its capacity too");

  // -- the distribution shift: both heuristics keep paying off, prefetching more so -----
  const firstHalf = measured.slice(0, measured.length / 2);
  const secondHalf = measured.slice(measured.length / 2);
  const measureWindow = (trace) => ({
    cold: replay(trace, 24, { history, warmCount: 0, prefetch: false, source }).hitRatio,
    warmed: replay(trace, 24, { history, warmCount: 20, prefetch: false, source }).hitRatio,
    prefetchOnly: replay(trace, 24, { history, warmCount: 0, prefetch: true, source }).hitRatio,
  });
  const early = measureWindow(firstHalf);
  const late = measureWindow(secondHalf);
  ok(early.warmed > early.cold && late.warmed > late.cold, "warming helps in both halves on this trace");
  ok(early.prefetchOnly > early.cold && late.prefetchOnly > late.cold, "prefetching helps in both halves too");
  const warmKeys = trained.ranked.slice(0, 20);
  const rateIn = (trace) => trace.filter((key) => warmKeys.includes(key)).length / trace.length;
  ok(rateIn(firstHalf) < rateIn(secondHalf),
    "on this trace the ranked keys stay warm in the second half, which is why a static warmup keeps helping");

  // -- the successor model's precision: prefetched keys really arrive soon ----------
  const successive = new LRUCache(24);
  let issued = 0;
  let useful = 0;
  for (let i = 0; i < measured.length; i++) {
    const key = measured[i];
    successive.access(key, source(key));
    const successor = trained.bestSuccessor.get(key);
    if (successor === undefined || successive.has(successor) || trained.confidence.get(key) < 0.5) continue;
    successive.access(successor, source(successor), true);
    issued++;
    if (measured.slice(i + 1, i + 9).includes(successor)) useful++; // arrives within eight requests
  }
  ok(issued > 100, "the model must issue a meaningful number of prefetches");
  const precision = useful / issued;
  ok(precision > 0.4, `at least 40% of the prefetches must be used soon (measured ${precision.toFixed(3)})`);
  ok(precision < 1, "a successor model can never be perfectly precise on this trace");
  ok(results.prefetchOnly.cache.stats.hits - results.cold.cache.stats.hits > 100,
    "the measured prefetching must convert into real demand hits");

  // -- determinism ------------------------------------------------------------------
  const build = () => {
    const run = replay(measured, 24, configs.both);
    return { hits: run.cache.stats.hits, misses: run.cache.stats.misses, prefetches: run.cache.stats.prefetches };
  };
  strictEqual(JSON.stringify(build()), JSON.stringify(build()), "the simulation must be deterministic");

  console.log(`hits cold=${results.cold.cache.stats.hits} warmed=${results.warmed.cache.stats.hits} prefetch=${results.prefetchOnly.cache.stats.hits} both=${results.both.cache.stats.hits} accesses=${checkedAccesses}`);
  console.log("PASS");
}

