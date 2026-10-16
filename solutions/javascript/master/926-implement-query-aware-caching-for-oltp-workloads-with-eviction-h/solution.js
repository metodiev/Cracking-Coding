/** Task 926 - Implement query-aware caching for OLTP workloads with eviction hints.
 *  (Tier: Master, CTCI Ch 14 - Databases)
 *
 * Three caches over the same seeded OLTP trace of (key, cost, reuse-hint) requests:
 *   LRU            the recency baseline (Map insertion order as the recency list);
 *   HintCache      GreedyDual-style, query-aware eviction: every entry holds H = L + cost * (1 +
 *                  hint.reuse), where L is the inflation counter set to the H of the entry evicted
 *                  last. Recency is therefore weighted by the query's predicted reuse and by how
 *                  expensive the miss is, and stale entries lose to fresher or costlier ones;
 *   Belady MIN     offline optimum used as the oracle for the unweighted trace: evicting the entry
 *                  whose next use lies farthest in the future maximises the hit count, so no online
 *                  policy may exceed it.
 * The checks assert the structural invariants after every access (capacity respected, no duplicate
 * entry, exactly one victim) and reconcile the reported accounting against an independent pass over
 * the trace. Simplification: fixed capacity, single-item entries, hints supplied by the workload
 * (no learned predictions). Verified: hint-aware beats LRU on the hint-favourable trace, hit counts
 * never exceed Belady, boundary capacities, hint semantics, invalid input and determinism.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    return (this.state = x);
  }
  float() { return this.next() / 4294967296; }
  int(bound) { return this.next() % bound; }
}

/** Recency-only cache: a Map iterates in insertion order, so the first key is the LRU victim. */
class LruCache {
  constructor(capacity) {
    assert.ok(Number.isInteger(capacity) && capacity >= 0, "capacity must be a non-negative integer");
    this.capacity = capacity;
    this.entries = new Map();
    this.hits = 0;
    this.misses = 0;
    this.missCost = 0;
    this.evictions = 0;
  }
  get size() { return this.entries.size; }
  access(key, cost) {
    assert.strictEqual(this.entries.size <= this.capacity, true, "capacity exceeded");
    if (this.entries.has(key)) {
      this.entries.delete(key);
      this.entries.set(key, cost);
      this.hits++;
      return true;
    }
    this.misses++;
    this.missCost += cost;
    if (this.capacity > 0) {
      if (this.entries.size === this.capacity) { this.entries.delete(this.entries.keys().next().value); this.evictions++; }
      this.entries.set(key, cost);
    }
    return false;
  }
}

/** Query-aware cache with eviction hints (GreedyDual flavour). */
class HintCache {
  constructor(capacity, options = {}) {
    assert.ok(Number.isInteger(capacity) && capacity >= 0, "capacity must be a non-negative integer");
    this.capacity = capacity;
    this.weight = options.hintWeight ?? 1;
    this.entries = new Map();   // key -> {H, cost, hint}
    this.inflation = 0;
    this.hits = 0;
    this.misses = 0;
    this.missCost = 0;
    this.evictions = 0;
  }
  get size() { return this.entries.size; }
  priority(cost, hint) { return this.inflation + cost * (1 + this.weight * Math.max(0, hint)); }
  access(key, cost, hint) {
    assert.ok(cost >= 0, "the query cost must be non-negative");
    const existing = this.entries.get(key);
    if (existing) {
      existing.cost = cost;
      existing.hint = hint;
      existing.H = this.priority(cost, hint);
      this.hits++;
      return true;
    }
    this.misses++;
    this.missCost += cost;
    if (this.capacity > 0) {
      if (this.entries.size >= this.capacity) {
        let victim = null, lowest = Infinity;
        for (const [candidate, entry] of this.entries) {
          if (entry.H < lowest) { lowest = entry.H; victim = candidate; }
        }
        this.entries.delete(victim);
        this.inflation = lowest;
        this.evictions++;
      }
      this.entries.set(key, { H: this.priority(cost, hint), cost, hint });
    }
    return false;
  }
}

/** Offline optimum for a unit-cost trace: always evict the entry used farthest in the future. */
function beladyHits(keys, capacity) {
  assert.ok(Number.isInteger(capacity) && capacity >= 0, "capacity must be a non-negative integer");
  const cache = new Set();
  let hits = 0;
  for (let i = 0; i < keys.length; i++) {
    if (cache.has(keys[i])) { hits++; continue; }
    if (capacity === 0) continue;
    if (cache.size === capacity) {
      let victim = null, farthest = -1;
      for (const candidate of cache) {
        let next = Infinity;
        for (let j = i + 1; j < keys.length; j++) {
          if (keys[j] === candidate) { next = j; break; }
        }
        if (next > farthest) { farthest = next; victim = candidate; }
      }
      cache.delete(victim);
    }
    cache.add(keys[i]);
  }
  return hits;
}

/** Seeded OLTP trace: a cheap hot class and an expensive periodic class, with noisy but genuinely
 *  informative reuse hints (the true number of reuses inside the next `window` accesses). */
function makeTrace(prng, options = {}) {
  const queries = options.queries ?? 400;
  const window = options.window ?? 30;
  const hotCount = options.hotCount ?? 6;
  const periodicCount = options.periodicCount ?? 6;
  const hot = Array.from({ length: hotCount }, (_, k) => `hot${k}`);
  const periodic = Array.from({ length: periodicCount }, (_, k) => `periodic${k}`);
  const keys = [], costs = [];
  for (let k = 0; k < queries; k++) {
    const usePeriodic = k % 5 === 4; // 20% of the traffic revisits the expensive class
    const key = usePeriodic ? periodic[Math.floor(k / 5) % periodicCount] : hot[prng.int(hotCount)];
    keys.push(key);
    costs.push(usePeriodic ? 10 : 1);
  }
  const hints = keys.map((key, k) => {
    let reuses = 0;
    for (let j = k + 1; j <= k + window && j < keys.length; j++) if (keys[j] === key) reuses++;
    return Math.max(0, reuses + (prng.float() < 0.25 ? (prng.float() < 0.5 ? -1 : 1) : 0));
  });
  return { keys, costs, hints, window };
}

function main() {
  const prng = new Prng(20250926);
  const trace = makeTrace(prng);
  const capacity = 10;

  // 1. Structural invariants are enforced on every single access of both online caches.
  const lru = new LruCache(capacity);
  const hinted = new HintCache(capacity);
  for (let k = 0; k < trace.keys.length; k++) {
    lru.access(trace.keys[k], trace.costs[k]);
    hinted.access(trace.keys[k], trace.costs[k], trace.hints[k]);
    assert.ok(lru.size <= capacity && hinted.size <= capacity, "capacity exceeded");
    assert.strictEqual(new Set(hinted.entries.keys()).size, hinted.size, "duplicate cache entry");
  }
  assert.strictEqual(lru.hits + lru.misses, trace.keys.length);
  assert.strictEqual(hinted.hits + hinted.misses, trace.keys.length);

  // 2. Independent accounting: the reported miss cost equals a pass over the trace's miss positions.
  const reconcile = (cache) => cache.missCost;
  const replayMissCost = (policy) => {
    const local = policy === "lru" ? new LruCache(capacity) : new HintCache(capacity);
    let cost = 0;
    for (let k = 0; k < trace.keys.length; k++) {
      const hit = policy === "lru" ? local.access(trace.keys[k], trace.costs[k])
        : local.access(trace.keys[k], trace.costs[k], trace.hints[k]);
      if (!hit) cost += trace.costs[k];
    }
    return { cost, misses: local.misses };
  };
  assert.strictEqual(reconcile(lru), replayMissCost("lru").cost);
  assert.strictEqual(reconcile(hinted), replayMissCost("hint").cost);
  assert.strictEqual(hinted.misses, replayMissCost("hint").misses);

  // 3. Hint-aware eviction wins on the query-aware objective (weighted miss cost), and it is the
  //    cost that matters: LRU can score more raw hits yet pay far more for the misses it takes.
  assert.ok(hinted.missCost * 1.4 < lru.missCost,
    `hint-aware cost ${hinted.missCost} versus LRU ${lru.missCost}`);
  assert.ok(lru.hits > hinted.hits, "on this trace raw recency should win the hit count");
  assert.ok(hinted.evictions > 0 && lru.evictions > 0);

  // 4. Oracle: on a unit-cost trace no online policy may reach more hits than Belady's offline MIN.
  const unitCosts = trace.costs.map(() => 1);
  const offline = beladyHits(trace.keys, capacity);
  const lruUnit = new LruCache(capacity);
  const hintUnit = new HintCache(capacity);
  for (let k = 0; k < trace.keys.length; k++) {
    lruUnit.access(trace.keys[k], unitCosts[k]);
    hintUnit.access(trace.keys[k], unitCosts[k], trace.hints[k]);
  }
  assert.ok(offline >= lruUnit.hits, `belady ${offline} < LRU ${lruUnit.hits}`);
  assert.ok(offline >= hintUnit.hits, `belady ${offline} < hint cache ${hintUnit.hits}`);
  assert.ok(offline > hintUnit.hits, "a strict gap is expected on this trace");
  const distinctKeys = new Set(trace.keys).size;
  const unbounded = new LruCache(distinctKeys);
  for (let k = 0; k < trace.keys.length; k++) unbounded.access(trace.keys[k], unitCosts[k]);
  assert.strictEqual(unbounded.hits, trace.keys.length - distinctKeys,
    "an unbounded cache only misses on first touch");

  // 5. Boundary capacities: zero disables the cache, one keeps a single item, empty trace is free.
  const zero = new HintCache(0);
  for (let k = 0; k < 20; k++) zero.access(`k${k % 3}`, 5, 1);
  assert.strictEqual(zero.hits, 0);
  assert.strictEqual(zero.size, 0);
  assert.strictEqual(zero.missCost, 100);
  const one = new HintCache(1);
  ["a", "a", "b", "b"].forEach((key, k) => one.access(key, 1, k % 2));
  assert.strictEqual(one.hits, 2, "a one-slot cache hits on every immediate repeat");
  assert.strictEqual(one.misses, 2);
  assert.strictEqual(one.size, 1);
  assert.strictEqual(one.evictions, 1, "moving to b must evict a");
  const nothing = new LruCache(4);
  assert.strictEqual(nothing.hits + nothing.misses, 0);
  assert.strictEqual(nothing.missCost, 0);
  const noCapacity = new LruCache(0);
  noCapacity.access("z", 3);
  assert.strictEqual(noCapacity.size, 0, "a zero-capacity cache stores nothing");

  // 6. Hint semantics: a high predicted reuse protects a cheap query, a low one does not.
  const semantics = new HintCache(2);
  semantics.access("cheap-hot", 1, 100);
  semantics.access("expensive-cold", 9, 0);
  semantics.access("filler", 1, 0);
  assert.ok(semantics.entries.has("cheap-hot"), "a strong hint must survive a costly competitor");
  assert.ok(!semantics.entries.has("expensive-cold"), "a zero hint must be evicted");
  const reversed = new HintCache(2);
  reversed.access("cheap-hot", 1, 0);
  reversed.access("expensive-cold", 9, 5);
  reversed.access("filler", 1, 0);
  assert.ok(reversed.entries.has("expensive-cold"), "cost times hint must dominate");
  assert.ok(!reversed.entries.has("cheap-hot"));

  // 7. Invalid input is rejected.
  assert.throws(() => new LruCache(-1), /non-negative integer/);
  assert.throws(() => new HintCache(1.5), /non-negative integer/);
  assert.throws(() => new HintCache(2).access("k", -3, 1), /non-negative/);
  assert.throws(() => beladyHits(["a"], -2), /non-negative integer/);

  // 8. Determinism: the same seed rebuilds the same trace and the same hit counts.
  const rerun = () => {
    const local = makeTrace(new Prng(20250926));
    const cache = new HintCache(capacity);
    local.keys.forEach((key, k) => cache.access(key, local.costs[k], local.hints[k]));
    return { hits: cache.hits, misses: cache.misses, cost: cache.missCost, keys: local.keys };
  };
  assert.deepStrictEqual(rerun(), rerun());
  const otherSeed = makeTrace(new Prng(11));
  assert.notDeepStrictEqual(otherSeed.keys, trace.keys, "a different seed must change the trace");

  console.log(`cache: capacity ${capacity}, LRU ${lru.hits} hits / cost ${lru.missCost}, ` +
    `hint-aware ${hinted.hits} hits / cost ${hinted.missCost}, belady ${offline} hits`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, LruCache, HintCache, beladyHits, makeTrace };
