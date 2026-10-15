/**
 * Task 353 - Implement concurrent hashmap simplified (sharding). (Tier: Intermediate, CTCI Ch 15)
 *
 * JavaScript runs single-threaded here, so this is the repo's documented task-195 async
 * simulation: each shard owns an AsyncMutex whose promise waiters stand in for a blocking
 * lock, and four virtual "thread" coroutines interleave at every await. The shard routing
 * is explicit (see the shardIndex policy), per-shard lock acquisitions are counted and
 * compared with an independent count derived from the operation log, and the final map is
 * compared key by key with a single-threaded reference map. Hash policy (identical in all
 * four languages): shard(key) = ((key % S) + S) % S for S = 4 shards, so -1 -> 3, -4 -> 0.
 */
"use strict";

const SHARDS = 4;
const THREADS = 4;
const KEYS = 50;

class AsyncMutex {
  constructor() {
    this.locked = false;
    this.waiters = [];
    this.acquisitions = 0;
    this.maxConcurrentHolders = 0;
    this.holders = 0;
  }

  async acquire() {
    while (this.locked) await this._wait();
    this.locked = true;
    this.holders += 1;
    this.maxConcurrentHolders = Math.max(this.maxConcurrentHolders, this.holders);
    this.acquisitions += 1;
  }

  release() {
    this.holders -= 1;
    this.locked = false;
    const resolve = this.waiters.shift();
    if (resolve) resolve();
  }

  _wait() {
    return new Promise((resolve) => this.waiters.push(resolve));
  }
}

class ShardedMap {
  constructor(numShards) {
    if (numShards <= 0) throw new Error("numShards must be positive");
    this.shards = [];
    this.locks = [];
    for (let i = 0; i < numShards; i++) {
      this.shards.push(new Map());
      this.locks.push(new AsyncMutex());
    }
    this.opLog = [];                              // key touched by each operation
  }

  shardIndex(key) {
    const count = this.shards.length;
    return ((key % count) + count) % count;
  }

  async _withShard(key, action) {
    const index = this.shardIndex(key);
    await this.locks[index].acquire();
    try {
      this.opLog.push(key);
      return action(this.shards[index]);
    } finally {
      this.locks[index].release();
    }
  }

  async put(key, value) {
    await this._withShard(key, (shard) => { shard.set(key, value); });
  }

  async get(key) {
    return this._withShard(key, (shard) => (shard.has(key) ? shard.get(key) : null));
  }

  async remove(key) {
    return this._withShard(key, (shard) => shard.delete(key));
  }

  size() {
    let total = 0;
    for (const shard of this.shards) total += shard.size;
    return total;
  }

  keys() {
    const result = [];
    for (const shard of this.shards) for (const key of shard.keys()) result.push(key);
    return result.sort((a, b) => a - b);
  }

  shardsContaining(key) {
    const result = [];
    for (let i = 0; i < this.shards.length; i++) if (this.shards[i].has(key)) result.push(i);
    return result;
  }
}

async function main() {
  const assert = require("node:assert");
  const mapping = new ShardedMap(SHARDS);

  // Phase 1 (single-threaded): keys 0..49 land in shards 0..3 with hand-checked counts.
  for (let key = 0; key < KEYS; key++) await mapping.put(key, key * 3);
  assert.deepStrictEqual(mapping.shards.map((shard) => shard.size), [13, 13, 12, 12],
    "hand-checked shard counts");

  // Every inserted key is in exactly one shard, and it is the policy shard.
  for (let key = 0; key < KEYS; key++) {
    assert.deepStrictEqual(mapping.shardsContaining(key), [mapping.shardIndex(key)],
      `key ${key} lives in exactly its policy shard`);
  }

  // Negative keys are normalised by the documented policy (hand-checked: -1 -> 3, -4 -> 0).
  assert.strictEqual(mapping.shardIndex(-1), 3, "negative key policy: -1");
  assert.strictEqual(mapping.shardIndex(-4), 0, "negative key policy: -4");
  await mapping.put(-1, 999);
  assert.deepStrictEqual(mapping.shardsContaining(-1), [3], "negative key lands in one shard");
  assert.strictEqual(await mapping.get(-1), 999, "negative key round-trips");
  assert.strictEqual(await mapping.remove(-1), true, "negative key removed");
  assert.strictEqual(await mapping.remove(-1), false, "second remove of a missing key is false");
  assert.strictEqual(mapping.size(), KEYS, "size restored after the negative-key check");

  // Phase 2 (simulated concurrency): 4 virtual threads upsert keys 0..49 (same value per key
  // in every thread, so any interleaving ends in the same state) and add disjoint 100..199.
  const workers = [];
  for (let threadIndex = 0; threadIndex < THREADS; threadIndex++) {
    workers.push((async () => {
      for (let key = 0; key < KEYS; key++) await mapping.put(key, key * 3 + 1000);
      for (let key = 100 + threadIndex * 25; key < 100 + (threadIndex + 1) * 25; key++) {
        await mapping.put(key, key + 1);
      }
    })());
  }
  await Promise.all(workers);

  assert.strictEqual(mapping.size(), 150, "every concurrent upsert persisted exactly once");
  for (let key = 0; key < KEYS; key++) {
    assert.strictEqual(await mapping.get(key), key * 3 + 1000, `overlapping upsert at ${key}`);
    assert.deepStrictEqual(mapping.shardsContaining(key), [mapping.shardIndex(key)]);
  }
  for (let key = 100; key < 200; key++) {
    assert.strictEqual(await mapping.get(key), key + 1, `disjoint upsert at ${key}`);
    assert.deepStrictEqual(mapping.shardsContaining(key), [mapping.shardIndex(key)]);
  }

  // Lock accounting: each operation acquired exactly its key's shard lock once, and no
  // shard lock ever had two holders at the same time.
  const expectedAcquisitions = new Array(SHARDS).fill(0);
  for (const key of mapping.opLog) expectedAcquisitions[mapping.shardIndex(key)] += 1;
  for (let i = 0; i < SHARDS; i++) {
    assert.strictEqual(mapping.locks[i].acquisitions, expectedAcquisitions[i],
      `shard ${i} acquired exactly once per routed operation`);
    assert.strictEqual(mapping.locks[i].maxConcurrentHolders, 1,
      `shard ${i} lock never had two holders`);
  }

  // Reference map built single-threaded with the same logical operations.
  const reference = new Map();
  for (let key = 0; key < KEYS; key++) reference.set(key, key * 3);
  for (let key = 0; key < KEYS; key++) reference.set(key, key * 3 + 1000);
  for (let t = 0; t < THREADS; t++) {
    for (let key = 100 + t * 25; key < 125 + t * 25; key++) reference.set(key, key + 1);
  }

  assert.deepStrictEqual(mapping.keys(), [...reference.keys()].sort((a, b) => a - b),
    "same key set as the reference");
  assert.strictEqual(mapping.size(), reference.size, "same size as the reference");
  for (let key = -10; key < 260; key++) {
    const expected = reference.has(key) ? reference.get(key) : null;
    assert.strictEqual(await mapping.get(key), expected, `reference match at ${key}`);
  }

  // Removal path, then a full re-comparison against the reference.
  assert.strictEqual(await mapping.remove(150), true, "remove existing key");
  assert.strictEqual(await mapping.get(150), null, "removed key reads as missing");
  assert.strictEqual(await mapping.remove(150), false, "remove missing key");
  reference.delete(150);
  assert.strictEqual(mapping.size(), reference.size, "size after removal");
  for (let key = -10; key < 260; key++) {
    const expected = reference.has(key) ? reference.get(key) : null;
    assert.strictEqual(await mapping.get(key), expected, `post-removal match at ${key}`);
  }
}

module.exports = { ShardedMap, AsyncMutex };

if (require.main === module) {
  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
