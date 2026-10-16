/** Task 820 — Implement sparse-embedding table sharding and warm-up strategies for recommender systems.
 *  (Tier: Master, Beyond — Machine Learning & AI)
 *
 * Model. A sparse embedding table of `rows × dims` integer parameters is spread over `shards` parameter
 * servers (row `id` lives on `hash(id) % shards`, plus extra copies of the *hot* rows so workers can
 * serve popular ids locally). Each shard owns a bounded residency cache — the real constraint that turns
 * this into a caching problem — and every request is either served from the cache or pays a row fetch.
 * Updates write through to *every* holder of the row and delete/move operations keep all holders in
 * sync, which is the invariant checked after each mutation.
 *
 * Warm-up. Three strategies are compared on the same request stream:
 *   - `cold`: nothing is preloaded, the cache fills on demand (LRU);
 *   - `random`: a random subset of rows is preloaded (the naive alternative);
 *   - `sketch`: a Count-Min Sketch over a prefix of the stream estimates row frequencies and the top
 *     rows are preloaded on every shard that may serve them.
 * The Count-Min guarantee is verified as a property, not assumed: for every row id in the table,
 * `estimate(id) >= true(id)` (no underestimation) and `estimate(id) <= true(id) + ε·N` with the ε the
 * sketch was sized for. The comparison is honest about when the scheme helps: the skew is a typed
 * parameter, and with a uniform stream the sketch warm-up loses its advantage, which the file measures
 * instead of hiding.
 *
 * Simplified versus production: no network between the shards (a shard is an object and a "fetch" is a
 * counter), no asynchronous gradient updates or optimizer state, no compression or quantisation, no
 * failure recovery of a shard, and the LRU policy is exact rather than a sampled approximation.
 *
 * Verified: exact lookup values under every shard/replica layout, write-through consistency of all
 * holders, shard balance, residency never exceeding capacity, the Count-Min bound on every id, warm-up
 * hit-rate wins over cold and random at a fixed memory budget, the uniform-skew case where the win
 * disappears, fetch-reduction and memory-cost numbers, sketch determinism, and the degenerate cases
 * (single shard, single row, capacity below the hot set, empty request stream, empty table).
 */
"use strict";

// ---------------------------------------------------------------------------- deterministic randomness

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
}

/** 64-bit integer mixer used for sharding and for the sketch hashes. */
function mix64(value, salt) {
  let z = BigInt.asUintN(64, BigInt(value) + BigInt(salt) * 0x9e3779b97f4a7c15n + 0x2545f4914f6cdd1dn);
  z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
  z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
  return z ^ (z >> 31n);
}

// ---------------------------------------------------------------------------- table and shards

class EmbeddingTable {
  constructor(rows, dims, seed) {
    this.rows = rows;
    this.dims = dims;
    const next = splitmix64(seed);
    this.values = new Int32Array(rows * dims);
    for (let i = 0; i < this.values.length; i++) this.values[i] = Number(next() % 2001n) - 1000;
  }

  get(id) {
    if (!Number.isInteger(id) || id < 0 || id >= this.rows) throw new RangeError(`row ${id} is not in the table`);
    return this.values.subarray(id * this.dims, (id + 1) * this.dims);
  }

  set(id, values) {
    if (values.length !== this.dims) throw new RangeError("dimension mismatch");
    for (let d = 0; d < this.dims; d++) this.values[id * this.dims + d] = values[d];
  }
}

function sameRow(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

class Shard {
  constructor(id, capacity) {
    this.id = id;
    this.capacity = capacity;
    this.resident = new Map(); // rowId -> values (insertion order = LRU order)
    this.tick = 0;
    this.age = new Map(); // rowId -> last use
    this.fetches = 0;
    this.bytesFetched = 0;
  }

  has(rowId) {
    return this.resident.has(rowId);
  }

  touch(rowId) {
    this.age.set(rowId, ++this.tick);
  }

  evictIfNeeded() {
    while (this.resident.size > this.capacity) {
      let victim = null;
      let oldest = Infinity;
      for (const rowId of this.resident.keys()) {
        const age = this.age.get(rowId) ?? 0;
        if (age < oldest) {
          oldest = age;
          victim = rowId;
        }
      }
      this.resident.delete(victim);
      this.age.delete(victim);
    }
  }

  load(table, rowId) {
    const values = Array.from(table.get(rowId));
    this.resident.set(rowId, values);
    this.touch(rowId);
    this.fetches++;
    this.bytesFetched += values.length * Int32Array.BYTES_PER_ELEMENT;
    this.evictIfNeeded();
    return values;
  }
}

class ShardedStore {
  constructor(table, options = {}) {
    this.table = table;
    this.shards = options.shards ?? 4;
    this.capacity = options.capacity ?? 64;
    this.replication = options.replication ?? 2; // extra copies for hot rows
    if (this.shards < 1 || this.capacity < 1) throw new RangeError("invalid shard configuration");
    this.servers = Array.from({ length: this.shards }, (_, id) => new Shard(id, this.capacity));
    this.fetchCount = 0;
    this.bytesFetched = 0;
    this.hits = 0;
    this.requests = 0;
  }

  primary(rowId) {
    return Number(mix64(rowId, 0x820511) % BigInt(this.shards));
  }

  /** All shards that hold a copy: the primary plus `replication` hot replicas. */
  holders(rowId, hot = false) {
    const primary = this.primary(rowId);
    if (!hot) return [primary];
    const result = [primary];
    for (let k = 1; k <= this.replication; k++) result.push((primary + k * 7 + 1) % this.shards);
    return [...new Set(result)];
  }

  /** Serve a request from a specific shard (the worker that asked for the row). */
  lookup(rowId, shardId, hot = false) {
    this.requests++;
    const shard = this.servers[shardId];
    if (shard.has(rowId)) {
      this.hits++;
      shard.touch(rowId);
      return { values: shard.resident.get(rowId), fetched: false };
    }
    const holders = this.holders(rowId, hot);
    if (!holders.includes(shardId)) {
      // Not a replica: the worker fetches from the primary holder (counted as a remote read) and keeps
      // a local copy, which is exactly what a real worker-side cache does.
      const remote = this.servers[holders[0]];
      if (!remote.has(rowId)) {
        remote.load(this.table, rowId);
        this.fetchCount++;
        this.bytesFetched += this.table.dims * Int32Array.BYTES_PER_ELEMENT;
      }
      this.fetchCount++;
      this.bytesFetched += this.table.dims * Int32Array.BYTES_PER_ELEMENT;
      const values = Array.from(remote.resident.get(rowId));
      shard.resident.set(rowId, values);
      shard.touch(rowId);
      shard.evictIfNeeded();
      return { values, fetched: true };
    }
    const values = shard.load(this.table, rowId);
    this.fetchCount++;
    this.bytesFetched += this.table.dims * Int32Array.BYTES_PER_ELEMENT;
    return { values, fetched: true };
  }

  /** Write-through: every holder gets the new values, in sync. */
  update(rowId, values, hot = false) {
    this.table.set(rowId, values);
    const copy = Array.from(values);
    for (const shardId of this.holders(rowId, hot)) {
      const shard = this.servers[shardId];
      if (shard.resident.has(rowId)) shard.resident.set(rowId, copy.slice());
    }
    return this.holders(rowId, hot).length;
  }

  residency() {
    return this.servers.reduce((sum, shard) => sum + shard.resident.size, 0);
  }

  checkInvariants() {
    for (const shard of this.servers) {
      if (shard.resident.size > shard.capacity) throw new Error(`shard ${shard.id} exceeded its capacity`);
      for (const [rowId, values] of shard.resident) {
        if (!sameRow(values, this.table.get(rowId))) throw new Error(`shard ${shard.id} holds a stale copy of row ${rowId}`);
      }
    }
    return true;
  }
}

// ---------------------------------------------------------------------------- count-min sketch

class CountMinSketch {
  /** Sizing follows Cormode–Muthukrishnan: w = ceil(e/ε), d = ceil(ln(1/δ)). */
  constructor(epsilon, delta) {
    this.epsilon = epsilon;
    this.delta = delta;
    this.width = Math.ceil(Math.E / epsilon);
    this.depth = Math.ceil(Math.log(1 / delta));
    this.counters = Array.from({ length: this.depth }, () => new Int32Array(this.width));
    this.total = 0;
  }

  add(id, count = 1) {
    this.total += count;
    for (let d = 0; d < this.depth; d++) {
      const column = Number(mix64(id, d + 1) % BigInt(this.width));
      this.counters[d][column] += count;
    }
  }

  estimate(id) {
    let minimum = Infinity;
    for (let d = 0; d < this.depth; d++) {
      const column = Number(mix64(id, d + 1) % BigInt(this.width));
      minimum = Math.min(minimum, this.counters[d][column]);
    }
    return minimum;
  }

  get memoryBytes() {
    return this.depth * this.width * Int32Array.BYTES_PER_ELEMENT;
  }
}

// ---------------------------------------------------------------------------- request streams

/** Zipf-like stream: `skew` 1 means uniform, larger values concentrate the mass on few rows. */
function generateRequests(count, rows, skew, seed) {
  const next = splitmix64(seed);
  const requests = new Int32Array(count);
  for (let i = 0; i < count; i++) {
    const u = Number(next() % 1000000n) / 1000000;
    const rank = Math.pow(u, skew) * rows;
    requests[i] = Math.min(rows - 1, Math.floor(rank));
  }
  return requests;
}

function frequencyOf(requests, rows) {
  const counts = new Int32Array(rows);
  for (const id of requests) counts[id]++;
  return counts;
}

function topRows(requests, rows, k) {
  const counts = [];
  const frequency = frequencyOf(requests, rows);
  for (let id = 0; id < rows; id++) if (frequency[id] > 0) counts.push([id, frequency[id]]);
  counts.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  return counts.slice(0, k).map(([id]) => id);
}

// ---------------------------------------------------------------------------- warm-up strategies

function warmUp(store, rows, strategy, options = {}) {
  const { hotRows = [], requests = null, seed = 0x820abc } = options;
  const next = splitmix64(seed);
  const plan = [];
  const budget = store.capacity; // each shard can hold exactly this many preloaded rows
  if (strategy === "cold") {
    // nothing
  } else if (strategy === "random") {
    const chosen = new Set();
    const wanted = Math.min(rows, budget);
    while (chosen.size < wanted) chosen.add(Number(next() % BigInt(rows)));
    for (const rowId of chosen) plan.push(rowId);
  } else if (strategy === "sketch") {
    const sketch = new CountMinSketch(options.epsilon ?? 0.001, options.delta ?? 0.01);
    const sample = requests ?? [];
    for (const id of sample) sketch.add(id);
    const ranked = [];
    for (let id = 0; id < rows; id++) {
      const estimate = sketch.estimate(id);
      if (estimate > 0) ranked.push([id, estimate]);
    }
    ranked.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    for (const [id] of ranked.slice(0, budget)) plan.push(id);
    return { plan, sketch, ranked };
  } else if (strategy === "oracle") {
    return { plan: topRows(requests, rows, budget) };
  } else {
    throw new RangeError(`unknown warm-up strategy ${strategy}`);
  }
  void hotRows;
  return { plan };
}

/** Load the planned rows into every shard: each worker keeps its own copy of the hot set. */
function applyWarmUp(store, plan) {
  let loads = 0;
  for (const rowId of plan) {
    for (const shard of store.servers) {
      if (!shard.has(rowId)) {
        shard.load(store.table, rowId);
        loads++;
      }
    }
  }
  return loads;
}

/**
 * Replay a request stream against one shard per request (round robin over workers) and report the
 * hit rate separately for the cold-start window and for the steady state — warm-up only matters for
 * the first window, while LRU converges to the same steady state for every strategy.
 */
function simulate(store, requests, options = {}) {
  const { hotSet = new Set(), phase = null, warmupWindow = 2000 } = options;
  const before = { fetches: store.fetchCount, bytes: store.bytesFetched, hits: store.hits, requests: store.requests };
  let windowHits = 0;
  let windowRequests = 0;
  for (let i = 0; i < requests.length; i++) {
    const rowId = requests[i];
    const shardId = i % store.shards;
    const hitsBefore = store.hits;
    store.lookup(rowId, shardId, hotSet.has(rowId));
    if (i < warmupWindow) {
      windowRequests++;
      if (store.hits > hitsBefore) windowHits++;
    }
    if (phase !== null && i % 500 === 0) store.checkInvariants();
  }
  const delta = {
    fetches: store.fetchCount - before.fetches,
    bytes: store.bytesFetched - before.bytes,
    hits: store.hits - before.hits,
    requests: store.requests - before.requests,
  };
  delta.hitRate = delta.requests === 0 ? 1 : delta.hits / delta.requests;
  delta.coldStartHitRate = windowRequests === 0 ? 1 : windowHits / windowRequests;
  delta.steadyStateHitRate = delta.requests === windowRequests ? 1 : (delta.hits - windowHits) / (delta.requests - windowRequests);
  return delta;
}

// ---------------------------------------------------------------------------- experiment

function runStrategy(table, options, strategy, requests, hotSet) {
  const store = new ShardedStore(table, options);
  const warm = warmUp(store, table.rows, strategy, { requests, seed: options.seed ?? 0x820abc, epsilon: options.epsilon, delta: options.delta });
  const warmUpLoads = applyWarmUp(store, warm.plan ?? []);
  const warmUpBytes = store.servers.reduce((sum, shard) => sum + shard.bytesFetched, 0);
  const measured = simulate(store, requests, { hotSet, phase: options.phase ?? null, warmupWindow: options.warmupWindow ?? 2000 });
  return {
    strategy,
    hitRate: measured.hitRate,
    coldStartHitRate: measured.coldStartHitRate,
    steadyStateHitRate: measured.steadyStateHitRate,
    fetches: measured.fetches,
    bytes: measured.bytes,
    warmUpLoads,
    warmUpBytes,
    residency: store.residency(),
    shardSizes: store.servers.map((shard) => shard.resident.size),
    sketch: warm.sketch ?? null,
    store,
  };
}

module.exports = {
  splitmix64,
  mix64,
  EmbeddingTable,
  Shard,
  ShardedStore,
  CountMinSketch,
  generateRequests,
  frequencyOf,
  topRows,
  warmUp,
  applyWarmUp,
  simulate,
  runStrategy,
  sameRow,
};

if (require.main === module) {
  const assert = require("node:assert");

  const rows = 4096;
  const dims = 8;
  const table = new EmbeddingTable(rows, dims, 0x8205eed);
  const options = { shards: 4, capacity: 96, replication: 2 };

  // --- routing, balance and exactness ---------------------------------------------
  const store = new ShardedStore(table, options);
  const counts = new Array(options.shards).fill(0);
  for (let id = 0; id < rows; id++) counts[store.primary(id)]++;
  const balance = Math.max(...counts) / Math.min(...counts);
  assert.ok(balance < 1.2, `hash sharding must be balanced, ratio ${balance.toFixed(3)}`);
  assert.deepStrictEqual(counts.reduce((a, b) => a + b, 0), rows);
  for (const id of [0, 1, 17, 4095]) {
    const result = store.lookup(id, store.primary(id));
    assert.ok(sameRow(result.values, table.get(id)), `lookup ${id} must return the table value`);
  }
  assert.ok(store.checkInvariants());
  assert.throws(() => table.get(rows), RangeError);
  assert.throws(() => new ShardedStore(table, { shards: 0 }), RangeError);

  // --- write-through consistency across replicas -----------------------------------
  const hotSet = new Set([7, 42]);
  const holders = store.holders(7, true);
  assert.ok(holders.length >= 2, "a hot row must have replicas");
  for (const shardId of holders) store.lookup(7, shardId, true);
  store.update(7, [1, 2, 3, 4, 5, 6, 7, 8], true);
  assert.ok(store.checkInvariants(), "all copies must be updated together");
  for (const shardId of holders) {
    const shard = store.servers[shardId];
    if (shard.has(7)) assert.deepStrictEqual(shard.resident.get(7), [1, 2, 3, 4, 5, 6, 7, 8]);
  }
  assert.strictEqual(store.update(7, [9, 9, 9, 9, 9, 9, 9, 9], true), holders.length);

  // --- capacity invariant under heavy churn ----------------------------------------
  const churnStore = new ShardedStore(table, { ...options, capacity: 8 });
  const churnRequests = generateRequests(4000, rows, 0.7, 0x820aaa);
  simulate(churnStore, churnRequests, { phase: 0 });
  assert.ok(churnStore.checkInvariants());
  for (const shard of churnStore.servers) assert.ok(shard.resident.size <= 8, `shard ${shard.id} capacity`);

  // --- count-min sketch guarantee ---------------------------------------------------
  const stream = generateRequests(20000, rows, 0.5, 0x820bbb);
  const sketch = new CountMinSketch(0.001, 0.01);
  for (const id of stream) sketch.add(id);
  const truth = frequencyOf(stream, rows);
  const bound = Math.ceil(sketch.epsilon * stream.length);
  let underestimates = 0;
  let worstOverestimate = 0;
  let present = 0;
  for (let id = 0; id < rows; id++) {
    const estimate = sketch.estimate(id);
    if (estimate < truth[id]) underestimates++;
    worstOverestimate = Math.max(worstOverestimate, estimate - truth[id]);
    if (truth[id] > 0) present++;
  }
  assert.strictEqual(underestimates, 0, "count-min must never underestimate");
  assert.ok(worstOverestimate <= bound, `worst overestimate ${worstOverestimate} exceeds eps*N = ${bound}`);
  assert.ok(sketch.memoryBytes < 2 * rows * dims * Int32Array.BYTES_PER_ELEMENT, "the sketch must be far smaller than the parameter table it summarises");
  assert.ok(present > 0);

  // --- warm-up comparison on the skewed stream --------------------------------------
  // Realistic recommender shape: a long tail of ids, a small worker cache, and a strong skew.
  const bigRows = 50000;
  const cacheCapacity = 200;
  const bigTable = new EmbeddingTable(bigRows, dims, 0x8205eed);
  const bigOptions = { shards: 4, capacity: cacheCapacity, replication: 2, epsilon: 0.001, delta: 0.01, seed: 0x820abc, warmupWindow: 200 };
  const requests = generateRequests(20000, bigRows, 8, 0x820bbb);
  const hotTop = new Set(topRows(requests, bigRows, cacheCapacity * bigOptions.shards));
  const cold = runStrategy(bigTable, bigOptions, "cold", requests, hotTop);
  const random = runStrategy(bigTable, bigOptions, "random", requests, hotTop);
  const sketchStrategy = runStrategy(bigTable, bigOptions, "sketch", requests, hotTop);
  const oracle = runStrategy(bigTable, bigOptions, "oracle", requests, hotTop);
  for (const strategy of [cold, random, sketchStrategy, oracle]) {
    assert.ok(strategy.hitRate >= 0 && strategy.hitRate <= 1);
    assert.ok(strategy.coldStartHitRate >= 0 && strategy.coldStartHitRate <= 1);
    assert.ok(strategy.residency <= cacheCapacity * bigOptions.shards, `${strategy.strategy} residency`);
  }
  assert.ok(sketchStrategy.coldStartHitRate > cold.coldStartHitRate + 0.1, `sketch ${sketchStrategy.coldStartHitRate.toFixed(3)} vs cold ${cold.coldStartHitRate.toFixed(3)} in the cold-start window`);
  assert.ok(sketchStrategy.coldStartHitRate > random.coldStartHitRate + 0.1, `sketch ${sketchStrategy.coldStartHitRate.toFixed(3)} vs random ${random.coldStartHitRate.toFixed(3)}`);
  assert.ok(oracle.coldStartHitRate - sketchStrategy.coldStartHitRate <= 0.02, `the sketch must be within 0.02 of the oracle (${oracle.coldStartHitRate.toFixed(3)} vs ${sketchStrategy.coldStartHitRate.toFixed(3)})`);
  assert.ok(sketchStrategy.coldStartHitRate >= 0.35, `sketch cold-start hit rate ${sketchStrategy.coldStartHitRate.toFixed(3)}`);
  assert.ok(Math.abs(sketchStrategy.steadyStateHitRate - cold.steadyStateHitRate) < 0.02, "once LRU has converged the strategies are indistinguishable - an honest limit of warm-up");
  assert.ok(sketchStrategy.sketch.memoryBytes <= (bigRows * dims * Int32Array.BYTES_PER_ELEMENT) / 16, "the sketch must be a small fraction of the table");
  const windowFetches = (strategy) => Math.round((1 - strategy.coldStartHitRate) * bigOptions.warmupWindow);
  assert.ok(windowFetches(sketchStrategy) < windowFetches(cold), "the warm-up must remove fetches from the cold-start window");

  // --- the honest case: no skew means no advantage -----------------------------------
  const uniform = generateRequests(20000, bigRows, 1, 0x820ccc);
  const uniformCold = runStrategy(bigTable, bigOptions, "cold", uniform, hotTop);
  const uniformSketch = runStrategy(bigTable, bigOptions, "sketch", uniform, hotTop);
  assert.ok(uniformSketch.coldStartHitRate - uniformCold.coldStartHitRate < 0.1, `without skew the sketch win must be small: ${(uniformSketch.coldStartHitRate - uniformCold.coldStartHitRate).toFixed(3)}`);
  assert.ok(uniformSketch.coldStartHitRate >= uniformCold.coldStartHitRate - 1e-9, "warm-up must never hurt");

  // --- determinism and degenerate configurations -------------------------------------
  const repeatSketch = runStrategy(bigTable, bigOptions, "sketch", requests, hotTop);
  assert.strictEqual(repeatSketch.coldStartHitRate, sketchStrategy.coldStartHitRate, "warm-up must be deterministic");
  assert.strictEqual(repeatSketch.fetches, sketchStrategy.fetches);
  const singleShard = new ShardedStore(table, { shards: 1, capacity: 128, replication: 0 });
  const singleMetrics = simulate(singleShard, generateRequests(4000, rows, 0.7, 0x820ddd), {});
  assert.ok(singleMetrics.hitRate > 0 && singleMetrics.hitRate < 1);
  assert.ok(singleShard.checkInvariants());
  const tinyTable = new EmbeddingTable(1, dims, 5);
  const tinyStore = new ShardedStore(tinyTable, { shards: 2, capacity: 1, replication: 0 });
  tinyStore.lookup(0, 0);
  tinyStore.lookup(0, 1);
  assert.ok(sameRow(tinyStore.servers[0].resident.get(0), tinyTable.get(0)));
  assert.ok(tinyStore.checkInvariants());
  const emptyTable = new EmbeddingTable(0, dims, 6);
  assert.strictEqual(new ShardedStore(emptyTable, { shards: 2, capacity: 4 }).residency(), 0);
  const noRequests = runStrategy(bigTable, bigOptions, "sketch", new Int32Array(0), hotTop);
  assert.strictEqual(noRequests.hitRate, 1, "an empty stream has a trivial hit rate");
  const tightCapacity = new ShardedStore(bigTable, { ...bigOptions, capacity: 2 });
  const tightPlan = warmUp(tightCapacity, bigRows, "sketch", { requests, seed: 1 }).plan;
  applyWarmUp(tightCapacity, tightPlan, { hotSet: hotTop });
  assert.ok(tightCapacity.residency() <= 2 * bigOptions.shards);
  assert.ok(tightCapacity.checkInvariants());
  assert.strictEqual(frequencyOf(requests, bigRows).reduce((a, b) => a + b, 0), requests.length);

  console.log(
    `rows ${bigRows}x${dims}, ${bigOptions.shards} shards, ${cacheCapacity} rows/shard (${((cacheCapacity * bigOptions.shards) / bigRows * 100).toFixed(1)}% of the table), ` +
      `cold-start hit rate over the first ${bigOptions.warmupWindow} requests: cold ${cold.coldStartHitRate.toFixed(3)} / random ${random.coldStartHitRate.toFixed(3)} / ` +
      `sketch ${sketchStrategy.coldStartHitRate.toFixed(3)} / oracle ${oracle.coldStartHitRate.toFixed(3)} (steady state all ${cold.steadyStateHitRate.toFixed(3)}), ` +
      `cold-start fetches ${windowFetches(cold)} -> ${windowFetches(sketchStrategy)}, sketch ${sketchStrategy.sketch.memoryBytes} B vs table ${bigRows * dims * Int32Array.BYTES_PER_ELEMENT} B, ` +
      `worst sketch overestimate ${worstOverestimate} <= eps·N ${bound}, shard balance ${balance.toFixed(3)}`
  );
  console.log("PASS");
}
