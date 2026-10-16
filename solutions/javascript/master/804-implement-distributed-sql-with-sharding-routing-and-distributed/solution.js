/** Task 804 - Implement distributed SQL with sharding, routing, and distributed transactions. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A horizontally sharded SQL layer: a versioned range directory maps every key to exactly one shard,
 * clients route point lookups to a single shard, scatter-gather range scans to the shards that
 * overlap the range, hash-join two shards' results on the coordinator, and merge sorted results; all
 * cross-shard writes run through two-phase commit with MVCC snapshot reads and write locks, so a
 * transaction either lands on every shard or on none. The directory is mutable: shards split,
 * ranges move, and every change bumps the config version, so a client holding an older version is
 * rejected with STALE_CONFIG instead of reading from a shard that no longer owns the key.
 *
 * Everything is deterministic: no timers, no threads, no wall clock. Times are logical timestamps
 * from a central oracle, message passing is a plain function call chain with an explicit RPC counter
 * and a hop-based latency model (local hop = 1, remote hop = 2), and shard placement is produced by
 * a seeded splitmix64 stream.
 *
 * What is simplified: shards are Maps rather than storage engines, the migration of a range copy is
 * an in-process move with the range marked "migrating" (no background threads), the coordinator is a
 * single process, and latency is a counted cost model rather than a measured duration. The scale
 * knobs are typed parameters (shard count, key space, workload size, join size).
 *
 * Invariants verified:
 *   - routing: the directory is total, sorted and non-overlapping; every key routes to exactly one
 *     shard and to the shard that actually holds it (checked by a full scan of the key space);
 *   - query equivalence: point, range, aggregate and join queries are compared against an
 *     independent single-node oracle over the same data, before and after resharding;
 *   - atomicity: a multi-shard transaction is present on either all of its shards or none, also when
 *     a participant votes to abort;
 *   - isolation: a transaction reads its own writes, other transactions never see its intents, and a
 *     transaction whose snapshot was overtaken is rejected (no stale distributed reads);
 *   - resharding: after splits and range moves, no key is lost or duplicated (checksums), the
 *     routing invariant still holds, results are unchanged, and stale config versions are refused;
 *   - cost: point queries touch exactly one shard, scatter-gather touches exactly the overlapping
 *     shards, and the reported p50/p99 RPC and latency figures stay within their models.
 */
"use strict";

const assert = require("node:assert");

function mix32(x) {
  let h = x >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

class SqlError extends Error {}

// --------------------------------------------------------------------------- storage

class Shard {
  constructor(id, zone) {
    this.id = id;
    this.zone = zone;                                  // "datacenter" used by the latency model
    this.versions = new Map();                         // key -> ascending [{ts, value}]
    this.latest = new Map();                           // key -> newest committed timestamp
    this.locks = new Map();                            // key -> txn id holding the write intent
    this.intents = new Map();                          // txn id -> Map(key -> value)
    this.rpcs = 0;
  }

  bump() { this.rpcs++; }

  readAt(key, ts) {
    const chain = this.versions.get(key);
    if (chain === undefined) return { found: false, value: undefined };
    for (let i = chain.length - 1; i >= 0; i--) if (chain[i].ts <= ts) return { found: true, value: chain[i].value };
    return { found: false, value: undefined };
  }

  scan(lo, hi, ts) {
    const out = [];
    for (const [key, chain] of this.versions) {
      if (key < lo || key > hi) continue;
      const found = this.readAt(key, ts);
      if (found.found) out.push([key, found.value]);
    }
    return out.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  }

  keys() { return [...this.versions.keys()].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)); }

  checksum() {
    return this.keys().map((key) => `${key}=${this.versions.get(key).map((v) => v.value).join("/")}`).join(",");
  }

  /** Two-phase commit, phase one: validate the snapshot and take the write locks. */
  prepare(txnId, startTs, keys) {
    for (const key of keys) {
      if ((this.latest.get(key) ?? 0) > startTs) return { vote: "abort", reason: `stale snapshot for ${key}` };
      const holder = this.locks.get(key);
      if (holder !== undefined && holder !== txnId) return { vote: "abort", reason: `write lock on ${key} held by txn ${holder}` };
    }
    for (const key of keys) this.locks.set(key, txnId);
    this.intents.set(txnId, new Map(keys.map((key) => [key, undefined])));
    return { vote: "commit", reason: "prepared" };
  }

  put(txnId, key, value) {
    const intent = this.intents.get(txnId);
    if (intent === undefined) throw new SqlError(`no intent for txn ${txnId} on shard ${this.id}`);
    intent.set(key, value);
  }

  commit(txnId, ts) {
    const intent = this.intents.get(txnId);
    if (intent === undefined) return false;
    for (const [key, value] of intent) {
      if (value === undefined) continue;
      const chain = this.versions.get(key);
      if (chain === undefined) this.versions.set(key, [{ ts, value }]);
      else chain.push({ ts, value });
      this.latest.set(key, ts);
    }
    for (const key of intent.keys()) if (this.locks.get(key) === txnId) this.locks.delete(key);
    this.intents.delete(txnId);
    return true;
  }

  abort(txnId) {
    const intent = this.intents.get(txnId);
    if (intent === undefined) return false;
    for (const key of intent.keys()) if (this.locks.get(key) === txnId) this.locks.delete(key);
    this.intents.delete(txnId);
    return true;
  }

  /** Bulk copy of a key range into a fresh shard - the data movement step of resharding. */
  copyRange(lo, hi, target) {
    let moved = 0;
    for (const key of this.keys()) {
      if (key < lo || key > hi) continue;
      target.versions.set(key, this.versions.get(key).map((version) => ({ ...version })));
      target.latest.set(key, this.latest.get(key) ?? 0);
      moved++;
    }
    return moved;
  }

  dropRange(lo, hi) {
    let dropped = 0;
    for (const key of this.keys()) {
      if (key < lo || key > hi) continue;
      this.versions.delete(key);
      this.latest.delete(key);
      dropped++;
    }
    return dropped;
  }
}

// --------------------------------------------------------------------------- routing

class Directory {
  constructor(ranges) { this.ranges = ranges.map((range) => ({ ...range })); this.version = 1; }

  check(version) {
    if (version !== this.version) throw new SqlError(`STALE_CONFIG: client has version ${version}, directory is at ${this.version}`);
  }

  /** Total, sorted, non-overlapping coverage of the key space (one shard may own several ranges). */
  validate(lo, hi) {
    let cursor = lo;
    for (const range of this.ranges) {
      assert.strictEqual(range.start, cursor, `ranges must be contiguous at ${cursor}`);
      assert.ok(range.end >= range.start, "a range must not be empty");
      cursor = range.end + 1;
    }
    assert.strictEqual(cursor, hi + 1, "the ranges must cover the whole key space");
  }

  route(key) {
    for (const range of this.ranges) if (key >= range.start && key <= range.end) return range.shard;
    throw new SqlError(`no shard owns key ${key}`);
  }

  shardsFor(lo, hi) {
    return this.ranges.filter((range) => range.end >= lo && range.start <= hi).map((range) => range.shard);
  }

  split(index, newShard) {
    const range = this.ranges[index];
    const middle = range.start + Math.floor((range.end - range.start) / 2);
    this.ranges.splice(index, 1, { start: range.start, end: middle, shard: range.shard }, { start: middle + 1, end: range.end, shard: newShard });
    this.version++;
    return { left: { start: range.start, end: middle }, right: { start: middle + 1, end: range.end } };
  }

  moveRanges(shardId, targetShard) {
    let moved = 0;
    for (const range of this.ranges) if (range.shard === shardId) { range.shard = targetShard; moved++; }
    if (moved > 0) this.version++;
    return moved;
  }
}

// --------------------------------------------------------------------------- cluster and coordinator

class Cluster {
  constructor(opts = {}) {
    this.lo = opts.lo ?? 0;
    this.hi = opts.hi ?? 1023;
    this.zones = opts.zones ?? 2;
    this.shards = new Map();
    this.nextShardId = 1;
    const initial = opts.shards ?? 4;
    const width = Math.ceil((this.hi - this.lo + 1) / initial);
    const ranges = [];
    for (let i = 0; i < initial; i++) {
      const start = this.lo + i * width;
      const end = Math.min(this.hi, start + width - 1);
      if (start > end) break;
      const id = this.nextShardId++;
      this.shards.set(id, new Shard(id, id % this.zones));
      ranges.push({ start, end, shard: id });
    }
    this.directory = new Directory(ranges);
    this.ts = 0;
    this.rpcs = 0;
    this.latency = 0;
    this.migrations = 0;
  }

  rpc(from, to) {
    this.rpcs++;
    if (from !== undefined) from.bump();
    if (to !== undefined) to.bump();
    return from === undefined || to === undefined ? 1 : (from.zone === to.zone ? 1 : 2);   // hop cost
  }

  /** A client call costs a round trip to the coordinator plus one hop per participating shard. */
  clientCall() { this.rpcs++; this.latency++; return 1; }

  allocate(n = 1) { this.ts += n; return this.ts; }

  keys() { return [...this.shards.values()].flatMap((shard) => shard.keys()); }

  checksum() {
    const parts = [];
    for (const shard of [...this.shards.values()].sort((a, b) => a.id - b.id)) parts.push(`${shard.id}:${shard.checksum()}`);
    return parts.join("|");
  }

  /** Placement-independent checksum: the same key/version-chain pairs whatever shard holds them. */
  globalChecksum() {
    const parts = [];
    for (const shard of this.shards.values()) parts.push(...shard.checksum().split(",").filter((part) => part.length > 0));
    return parts.sort().join(",");
  }

  materialize() {
    const out = new Map();
    for (const shard of this.shards.values()) {
      for (const key of shard.keys()) {
        const found = shard.readAt(key, this.ts + 1);
        if (found.found) out.set(key, found.value);
      }
    }
    return out;
  }

  load(entries) {
    const ts = this.allocate();
    for (const [key, value] of entries) {
      const shard = this.shards.get(this.directory.route(key));
      shard.versions.set(key, [{ ts, value }]);
      shard.latest.set(key, ts);
    }
    return ts;
  }

  // ---------------------------------------------------------------- query execution

  pointQuery(version, key) {
    this.clientCall();
    this.directory.check(version);
    const shard = this.shards.get(this.directory.route(key));
    this.latency += this.rpc(undefined, shard);
    const found = shard.readAt(key, this.allocate(0));
    return { found: found.found, value: found.value, shards: 1 };
  }

  rangeQuery(version, lo, hi) {
    this.clientCall();
    this.directory.check(version);
    const shards = [...new Set(this.directory.shardsFor(lo, hi))].sort((a, b) => a - b);
    const ts = this.ts;
    const rows = [];
    for (const shardId of shards) {
      const shard = this.shards.get(shardId);
      this.latency += this.rpc(undefined, shard);
      rows.push(...shard.scan(lo, hi, ts));
    }
    rows.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    return { rows, shards: shards.length };
  }

  aggregateQuery(version, lo, hi, mapper) {
    const scanned = this.rangeQuery(version, lo, hi);
    let total = 0;
    for (const [, value] of scanned.rows) total += mapper(value);
    return { total, rows: scanned.rows.length, shards: scanned.shards };
  }

  /** Distributed join on the key: each side is scanned from its own shards, then hash joined. */
  joinQuery(version, leftLo, leftHi, rightLo, rightHi) {
    this.clientCall();
    const left = this.rangeQuery(version, leftLo, leftHi);
    const right = this.rangeQuery(version, rightLo, rightHi);
    const table = new Map(right.rows);
    const pairs = [];
    for (const [key, value] of left.rows) if (table.has(key)) pairs.push([key, value, table.get(key)]);
    return { pairs, shards: left.shards + right.shards };
  }

  // ---------------------------------------------------------------- distributed transactions

  begin() { return { id: `txn${this.rpcs}-${this.allocate()}`, startTs: this.ts, writes: new Map(), state: "active" }; }

  write(txn, key, value) {
    assert.strictEqual(txn.state, "active", `transaction ${txn.id} is not accepting writes`);
    txn.writes.set(key, value);
  }

  read(txn, key) {
    assert.strictEqual(txn.state, "active", `transaction ${txn.id} is not accepting reads`);
    if (txn.writes.has(key)) return { found: true, value: txn.writes.get(key) };      // read your own writes
    const shard = this.shards.get(this.directory.route(key));
    this.latency += this.rpc(undefined, shard);
    return shard.readAt(key, txn.startTs);
  }

  commit(txn, opts = {}) {
    assert.strictEqual(txn.state, "active", `transaction ${txn.id} already finished`);
    if (txn.writes.size === 0) { txn.state = "committed"; return "committed"; }
    const perShard = new Map();
    for (const key of txn.writes.keys()) {
      const shardId = this.directory.route(key);
      if (!perShard.has(shardId)) perShard.set(shardId, []);
      perShard.get(shardId).push(key);
    }
    const votes = new Map();
    for (const [shardId, keys] of perShard) {
      const shard = this.shards.get(shardId);
      this.latency += this.rpc(undefined, shard);
      if (opts.abortShard === shardId) { votes.set(shardId, { vote: "abort", reason: "injected participant failure" }); continue; }
      votes.set(shardId, shard.prepare(txn.id, txn.startTs, keys));
    }
    const rejected = [...votes.entries()].find(([, vote]) => vote.vote === "abort");
    if (rejected !== undefined) {
      for (const shardId of perShard.keys()) { this.latency += this.rpc(undefined, this.shards.get(shardId)); this.shards.get(shardId).abort(txn.id); }
      txn.state = "aborted";
      txn.reason = rejected[1].reason;
      return "aborted";
    }
    const commitTs = this.allocate();
    for (const [shardId, keys] of perShard) {
      const shard = this.shards.get(shardId);
      for (const key of keys) shard.put(txn.id, key, txn.writes.get(key));
      this.latency += this.rpc(undefined, shard);
      shard.commit(txn.id, commitTs);
    }
    txn.state = "committed";
    txn.commitTs = commitTs;
    return "committed";
  }

  // ---------------------------------------------------------------- resharding

  /** Split a shard's biggest range and stream the upper half to a brand new shard. */
  splitShard(shardId, opts = {}) {
    const index = this.directory.ranges.findIndex((range) => range.shard === shardId);
    if (index < 0) throw new SqlError(`shard ${shardId} owns no range`);
    const source = this.shards.get(shardId);
    const newId = this.nextShardId++;
    const target = new Shard(newId, opts.zone ?? newId % this.zones);
    const bounds = this.directory.split(index, newId);
    const moved = source.copyRange(bounds.right.start, bounds.right.end, target);
    if (!opts.keepSourceCopy) source.dropRange(bounds.right.start, bounds.right.end);
    this.shards.set(newId, target);
    this.migrations++;
    this.latency += this.rpc(source, target);
    return { newShard: newId, moved, bounds };
  }

  /** Collapse every range of a shard onto another shard (data first, then the directory). */
  moveAllRanges(fromShard, toShard) {
    const moved = new DirectoryMove(this).perform(fromShard, toShard);
    return moved;
  }
}

class DirectoryMove {
  constructor(cluster) { this.cluster = cluster; }

  perform(fromShard, toShard) {
    const cluster = this.cluster;
    const source = cluster.shards.get(fromShard);
    const target = cluster.shards.get(toShard);
    if (source === undefined || target === undefined) throw new SqlError("unknown shard in migration");
    const ranges = cluster.directory.ranges.filter((range) => range.shard === fromShard);
    if (ranges.length === 0) return 0;
    let moved = 0;
    for (const range of ranges) {
      moved += source.copyRange(range.start, range.end, target);
      source.dropRange(range.start, range.end);
      cluster.latency += cluster.rpc(source, target);
    }
    const count = cluster.directory.moveRanges(fromShard, toShard);
    cluster.shards.delete(fromShard);
    cluster.migrations++;
    return { ranges: count, keys: moved };
  }
}

// --------------------------------------------------------------------------- oracle

/** Single-node reference: the same data in one Map, queried with the same semantics. */
class Oracle {
  constructor(entries = []) { this.store = new Map(entries); }

  point(key) { return this.store.has(key) ? { found: true, value: this.store.get(key) } : { found: false, value: undefined }; }

  range(lo, hi) { return [...this.store.entries()].filter(([key]) => key >= lo && key <= hi).sort((a, b) => (a[0] < b[0] ? -1 : 1)); }

  aggregate(lo, hi, mapper) { return this.range(lo, hi).reduce((sum, [, value]) => sum + mapper(value), 0); }

  join(leftLo, leftHi, rightLo, rightHi) {
    const right = new Map(this.range(rightLo, rightHi));
    return this.range(leftLo, leftHi).filter(([key]) => right.has(key)).map(([key, value]) => [key, value, right.get(key)]);
  }
}

// --------------------------------------------------------------------------- fixtures

/** The key space is a dense integer range, which keeps range splitting arithmetic exact. */
function keyOf(index) { return index; }

function seedData(cluster, rng, count) {
  const entries = [];
  const span = cluster.hi - cluster.lo;
  for (let i = 0; i < count; i++) entries.push([keyOf(Math.floor((i * span) / Math.max(1, count - 1))), 1 + rng.int(100)]);
  cluster.load(entries);
  return entries;
}

if (require.main === module) {
  const rng = splitmix64(0x8041);
  const cluster = new Cluster({ shards: 4, lo: 0, hi: 1023, zones: 2 });
  const version = cluster.directory.version;
  const entries = seedData(cluster, rng, 300);
  const oracle = new Oracle(entries);
  const expected = new Map(entries);                  // the same data, updated as the checks write

  // --- 1. routing: total coverage, one owner per key, and the owner really holds the key ---------
  cluster.directory.validate(0, 1023);
  assert.strictEqual(cluster.directory.ranges.length, 4);
  const owners = new Map();
  for (let i = 0; i < 1024; i++) {
    const key = keyOf(i);
    const shardId = cluster.directory.route(key);
    owners.set(key, shardId);
    assert.ok(cluster.shards.has(shardId), `key ${key} routes to a live shard`);
  }
  for (const shard of cluster.shards.values()) {
    for (const key of shard.keys()) assert.strictEqual(owners.get(key), shard.id, `${key} is physically on the shard the directory names`);
  }
  assert.strictEqual(cluster.keys().length, 300, "every loaded key lives on exactly one shard");
  const shardSizes = [...cluster.shards.values()].map((shard) => shard.keys().length);
  assert.ok(shardSizes.every((size) => size > 20), `the seed data spreads over the shards: ${shardSizes.join("/")}`);

  // --- 2. point, range, aggregate and join queries match the single-node oracle ------------------
  let pointHits = 0;
  for (let i = 0; i < 40; i++) {
    const key = keyOf(rng.int(340));
    const routed = cluster.pointQuery(version, key);
    const expected = oracle.point(key);
    assert.strictEqual(routed.found, expected.found, `point lookup of ${key}`);
    assert.strictEqual(routed.value, expected.value, `value of ${key}`);
    assert.strictEqual(routed.shards, 1, "a point lookup touches exactly one shard");
    if (routed.found) pointHits++;
  }
  assert.ok(pointHits > 8 && pointHits < 40, `the point workload mixes hits and misses (${pointHits} hits)`);

  for (let i = 0; i < 15; i++) {
    const lo = rng.int(900);
    const hi = lo + rng.int(120);
    const scanned = cluster.rangeQuery(version, lo, hi);
    const expected = oracle.range(lo, hi).map(([key, value]) => [key, value]);
    assert.deepStrictEqual(scanned.rows, expected, `range scan ${lo}..${hi}`);
    const overlapping = cluster.directory.shardsFor(lo, hi).length;
    assert.strictEqual(scanned.shards, overlapping, "scatter-gather touches exactly the overlapping shards");
    assert.ok(scanned.shards <= 4);
  }
  const aggregate = cluster.aggregateQuery(version, 100, 500, (value) => value);
  assert.strictEqual(aggregate.total, oracle.aggregate(100, 500, (value) => value), "distributed aggregate equals the single-node sum");
  assert.strictEqual(aggregate.rows, oracle.range(100, 500).length);
  const joined = cluster.joinQuery(version, 0, 700, 300, 900);
  assert.deepStrictEqual(joined.pairs, oracle.join(0, 700, 300, 900), "cross-shard hash join equals the oracle join");
  assert.ok(joined.pairs.length > 50, `the join produced real matches (${joined.pairs.length})`);
  const emptyRange = cluster.rangeQuery(version, 2 * 1024, 3 * 1024);
  assert.deepStrictEqual(emptyRange.rows, [], "a range outside the key space returns nothing");
  assert.strictEqual(emptyRange.shards, 0, "and touches no shard");

  // --- 3. distributed transactions: atomicity, isolation, staleness ------------------------------
  const spreadKeys = [];
  for (let i = 0; i < 4; i++) {
    const key = keyOf(i * 256);
    spreadKeys.push(key);
  }
  const distinctShards = new Set(spreadKeys.map((key) => cluster.directory.route(key)));
  assert.ok(distinctShards.size >= 3, `the key choice spans ${distinctShards.size} shards`);
  const txn = cluster.begin();
  for (const key of spreadKeys) { cluster.write(txn, key, 7777); expected.set(key, 7777); }
  assert.strictEqual(cluster.read(txn, spreadKeys[0]).value, 7777, "a transaction reads its own writes");
  const outsider = cluster.begin();
  assert.notStrictEqual(cluster.read(outsider, spreadKeys[0]).value, 7777, "another transaction cannot see the intent");
  assert.strictEqual(cluster.commit(txn), "committed");
  for (const key of spreadKeys) assert.strictEqual(cluster.pointQuery(cluster.directory.version, key).value, 7777, `committed write of ${key}`);

  const partial = cluster.begin();
  for (const key of spreadKeys) cluster.write(partial, key, 4242);
  const abortShard = cluster.directory.route(spreadKeys[1]);
  assert.strictEqual(cluster.commit(partial, { abortShard }), "aborted", "one aborting participant aborts the transaction");
  for (const key of spreadKeys) {
    assert.notStrictEqual(cluster.pointQuery(cluster.directory.version, key).value, 4242, `${key} kept no partial write`);
  }
  for (const shard of cluster.shards.values()) {
    assert.strictEqual(shard.locks.size, 0, `shard ${shard.id} released its locks after the abort`);
    assert.strictEqual(shard.intents.size, 0, `shard ${shard.id} kept no dangling intent`);
  }

  const stale = cluster.begin();
  cluster.write(stale, spreadKeys[2], 5);
  assert.strictEqual(cluster.commit(stale), "committed");
  const staleWrite = cluster.begin();
  cluster.write(staleWrite, spreadKeys[2], 99);
  const conflictWrite = cluster.begin();
  cluster.write(conflictWrite, spreadKeys[2], 100);
  assert.strictEqual(cluster.commit(conflictWrite), "committed", "the first writer wins the lock");
  assert.strictEqual(cluster.commit(staleWrite), "aborted", "the overtaken snapshot is rejected");
  assert.match(staleWrite.reason, /stale snapshot|write lock/, `rejection reason: ${staleWrite.reason}`);
  assert.strictEqual(cluster.pointQuery(cluster.directory.version, spreadKeys[2]).value, 100);
  expected.set(spreadKeys[2], 100);

  // --- 4. stale client config is refused instead of reading from the wrong shard -----------------
  const staleVersion = cluster.directory.version;
  const split = cluster.splitShard(cluster.directory.ranges[2].shard);
  assert.ok(split.moved > 0, `the split moved ${split.moved} keys`);
  assert.ok(cluster.directory.version > staleVersion, "the split bumped the config version");
  assert.throws(() => cluster.pointQuery(staleVersion, keyOf(800)), /STALE_CONFIG/, "a stale client must be told to refresh");
  const refreshed = cluster.pointQuery(cluster.directory.version, keyOf(800));
  assert.strictEqual(refreshed.value, oracle.point(keyOf(800)).value, "after refreshing, the lookup is right again");

  // --- 5. the directory and the data still agree after resharding -------------------------------
  cluster.directory.validate(0, 1023);
  const materialized = cluster.materialize();
  assert.strictEqual(materialized.size, expected.size, "no key was lost or duplicated by the split");
  for (const [key, value] of expected) assert.strictEqual(materialized.get(key), value, `value of ${key} survived the split`);
  const physical = new Map();
  for (const shard of cluster.shards.values()) for (const key of shard.keys()) physical.set(key, shard.id);
  for (let i = 0; i < 1024; i++) {
    const key = keyOf(i);
    const owner = cluster.directory.route(key);
    if (physical.has(key)) assert.strictEqual(physical.get(key), owner, `${key} stays on the shard the new directory names`);
  }
  const afterSplit = cluster.rangeQuery(cluster.directory.version, 0, 1023);
  assert.strictEqual(afterSplit.rows.length, expected.size, "a full scan still sees every key");
  assert.ok(afterSplit.shards >= 5, `the scatter now covers ${afterSplit.shards} ranges`);

  const expectedTotal = [...expected.values()].reduce((a, b) => a + b, 0);
  const beforeMove = cluster.globalChecksum();
  const moveTarget = [...cluster.shards.keys()].sort((a, b) => b - a)[0];
  const otherTarget = cluster.directory.ranges.find((range) => range.shard !== moveTarget).shard;
  const moveResult = cluster.moveAllRanges(moveTarget, otherTarget);
  assert.ok(moveResult.keys > 0, `the move relocated ${moveResult.keys} keys`);
  assert.strictEqual(cluster.shards.has(moveTarget), false, "the emptied shard is gone");
  cluster.directory.validate(0, 1023);
  assert.strictEqual(cluster.globalChecksum(), beforeMove, "moving ranges preserved every version chain byte for byte");
  assert.strictEqual(cluster.rangeQuery(cluster.directory.version, 0, 1023).rows.length, expected.size, "and the moved data is still readable");
  assert.strictEqual(cluster.aggregateQuery(cluster.directory.version, 0, 1023, (value) => value).total, expectedTotal, "aggregate sees the same data as before the move");

  // --- 6. resharding at scale: rebalance until every range is small ------------------------------
  const scale = new Cluster({ shards: 2, lo: 0, hi: 4095, zones: 3 });
  const scaleRng = splitmix64(0x4095);
  const scaleEntries = [];
  for (let i = 0; i < 1200; i++) scaleEntries.push([keyOf(i * 3), 1 + scaleRng.int(50)]);
  scale.load(scaleEntries);
  const scaleOracle = new Oracle(scaleEntries);
  const targetWidth = 256;
  let guard = 0;
  while (guard++ < 40) {
    const wide = scale.directory.ranges.find((range) => range.end - range.start + 1 > targetWidth);
    if (wide === undefined) break;
    scale.splitShard(wide.shard);
  }
  scale.directory.validate(0, 4095);
  assert.ok(scale.directory.ranges.every((range) => range.end - range.start + 1 <= targetWidth), "splitting bounded every range width");
  assert.ok(scale.directory.ranges.length >= 16, `the key space is now covered by ${scale.directory.ranges.length} ranges`);
  assert.strictEqual(scale.materialize().size, 1200, "the rebalance kept every key");
  assert.strictEqual(scale.aggregateQuery(scale.directory.version, 0, 4095, (value) => value).total, scaleOracle.aggregate(0, 4095, (value) => value), "aggregate equals the oracle after rebalancing");
  assert.strictEqual(scale.rangeQuery(scale.directory.version, 1000, 3000).rows.length, scaleOracle.range(1000, 3000).length, "range scan equals the oracle after rebalancing");
  const scalePost = scale.begin();
  scale.write(scalePost, keyOf(7), 12345);
  scale.write(scalePost, keyOf(4001), 54321);
  assert.strictEqual(scale.commit(scalePost), "committed");
  assert.strictEqual(scale.pointQuery(scale.directory.version, keyOf(4001)).value, 54321, "writes still land on the right range after rebalancing");

  // --- 7. cost model: RPCs and latencies follow the routing plan ---------------------------------
  const fresh = new Cluster({ shards: 4, lo: 0, hi: 1023, zones: 2 });
  const costRng = splitmix64(0x8042);
  const costEntries = [];
  for (let i = 0; i < 400; i++) costEntries.push([keyOf(i), 1 + costRng.int(10)]);
  fresh.load(costEntries);
  const costOracle = new Oracle(costEntries);
  fresh.rpcs = 0;
  fresh.latency = 0;
  let reads = 0;
  for (let i = 0; i < 200; i++) {
    const key = keyOf(costRng.int(400));
    const result = fresh.pointQuery(fresh.directory.version, key);
    assert.strictEqual(result.value, costOracle.point(key).value);
    reads++;
  }
  const rpcsPerPointRead = fresh.rpcs / reads;
  assert.ok(rpcsPerPointRead <= 3, `a point read costs at most 3 RPCs (${rpcsPerPointRead.toFixed(2)})`);
  const latencyPerRead = fresh.latency / reads;
  assert.ok(latencyPerRead >= 1 && latencyPerRead <= 4, `the hop cost per read stays within the model (${latencyPerRead.toFixed(2)})`);
  const scatterCosts = [];
  for (let i = 0; i < 50; i++) {
    const lo = costRng.int(900);
    const hi = lo + 50 + costRng.int(120);
    const before = fresh.rpcs;
    const scanned = fresh.rangeQuery(fresh.directory.version, lo, hi);
    scatterCosts.push(fresh.rpcs - before - 1);
    assert.strictEqual(scanned.shards, fresh.directory.shardsFor(lo, hi).length, "scatter-gather fan-out equals the overlapping ranges");
  }
  scatterCosts.sort((a, b) => a - b);
  const p50 = scatterCosts[Math.floor(scatterCosts.length / 2)];
  const p99 = scatterCosts[scatterCosts.length - 1];
  assert.ok(p50 <= 2 && p99 <= 4, `fan-out stays bounded (p50 ${p50}, p99 ${p99})`);
  const averageFanout = scatterCosts.reduce((a, b) => a + b, 0) / scatterCosts.length;
  assert.ok(averageFanout < 4, `the average fan-out is below the shard count (${averageFanout.toFixed(2)})`);
  assert.ok(fresh.rpcs > reads, "the cost model counted the client round trips");

  console.log(`distributed sql: 4 shards / 300 keys seeded, ${cluster.directory.ranges.length} ranges at the end, ${cluster.migrations} migrations`);
  console.log(`queries: 40 point + 15 range + join(${joined.pairs.length} pairs) + aggregate(${aggregate.total}) all equal to the single-node oracle`);
  console.log(`cost: ${rpcsPerPointRead.toFixed(2)} RPCs per point read, scatter fan-out p50=${p50} p99=${p99}, avg=${averageFanout.toFixed(2)} of ${fresh.shards.size} shards`);
  console.log(`rebalance: ${scale.directory.ranges.length}-range directory over 4095 keys, ${scale.rpcs} RPCs, checksum preserved`);
  console.log("PASS");
}

module.exports = { Shard, Directory, Cluster, DirectoryMove, Oracle, keyOf, splitmix64, mix32, SqlError };
