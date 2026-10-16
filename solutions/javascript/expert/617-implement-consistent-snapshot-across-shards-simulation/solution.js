/** Task 617 - Implement consistent snapshot across shards simulation. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A sharded key-value store where keys *move* between shards: a migration removes the key from
 * the source shard and sends it to the target shard as an in-band message, so at any real moment
 * a key lives in exactly one of three places - in a shard's store, or inside a message that is
 * still in its channel. Taking a snapshot of a sharded store is exactly the Chandy-Lamport
 * problem at shard granularity, driven here by a coordinator that seals every shard: a shard
 * records its store on its seal, starts recording every incoming channel except the one the seal
 * arrived on, and forwards a seal marker on all of its outgoing channels; when a channel's own
 * marker arrives the recording stops. Messages sent before the sender sealed and delivered after
 * the receiver sealed are therefore captured in the channel state, and a migration that crosses
 * the cut is counted exactly once.
 *
 * What is simplified: one coordinator seals all shards at once (a real deployment would let any
 * shard or the coordinator initiate), channels are explicit FIFO queues rather than a network,
 * and the store is a plain Map, i.e. there is no replication per shard. Verified: every shard
 * seals exactly once, the union of the shard snapshots plus the recorded channel messages contains
 * every key exactly once (no key is lost or duplicated - the property a non-coordinated snapshot
 * cannot give), no message crosses the cut backwards, every message sent before a sender's seal
 * is accounted for, and a deliberately naive snapshot built from staggered per-shard reads is
 * rejected by the very same checker.
 */
"use strict";

class Shard {
  constructor(id) {
    this.id = id;
    this.store = new Map();
    this.sealed = false;
    this.sealedAt = null;
    this.snapshot = new Map();
  }
}

class CrossShardStore {
  constructor(shardCount, edges) {
    if (!Number.isInteger(shardCount) || shardCount <= 0) throw new RangeError("shard count must be positive");
    this.shards = Array.from({ length: shardCount }, (_, id) => new Shard(id));
    this.channels = edges.map(([from, to]) => {
      if (!Number.isInteger(from) || !Number.isInteger(to) || from === to ||
          from < 0 || from >= shardCount || to < 0 || to >= shardCount) {
        throw new RangeError(`bad channel ${from}->${to}`);
      }
      return { from, to, queue: [], recording: false, recorded: [], sealedAtReceiver: null, stoppedAt: null };
    });
    this.now = 0;
    this.migrations = [];
    this.histories = this.shards.map(() => [{ time: 0, entries: [] }]);
  }

  #shard(id) {
    if (!Number.isInteger(id) || id < 0 || id >= this.shards.length) throw new RangeError(`bad shard ${id}`);
    return this.shards[id];
  }

  channelOf(from, to) {
    const channel = this.channels.find((c) => c.from === from && c.to === to);
    if (channel === undefined) throw new RangeError(`no channel ${from}->${to}`);
    return channel;
  }

  /** Data-local insert; keys are strings so the snapshot can be compared as a multiset. */
  put(shardId, key, value) {
    const shard = this.#shard(shardId);
    if (typeof key !== "string" || key.length === 0) throw new TypeError("keys must be non-empty strings");
    if (shard.store.has(key)) throw new Error(`key ${key} already on shard ${shardId}`);
    shard.store.set(key, value);
    this.#note(shard);
    return shard;
  }

  /** Move a key: remove it locally and send it through the FIFO channel to the target shard. */
  migrate(from, to, key) {
    const source = this.#shard(from);
    const channel = this.channelOf(from, to);            // validate before touching the store
    if (!source.store.has(key)) throw new Error(`key ${key} is not on shard ${from}`);
    const value = source.store.get(key);
    source.store.delete(key);
    this.#note(source);
    const message = { kind: "migrate", id: this.migrations.length, from, to, key, value, sentAt: ++this.now, deliveredAt: null };
    this.migrations.push(message);
    channel.queue.push(message);
    return message;
  }

  #note(shard) {
    this.histories[shard.id].push({ time: this.now, entries: [...shard.store.entries()].sort() });
  }

  storeAt(shardId, time) {
    const history = this.histories[this.#shard(shardId).id];
    let entries = [];
    for (const entry of history) if (entry.time <= time) entries = entry.entries;
    return new Map(entries);
  }

  #pushSeal(channel) {
    channel.queue.push({ kind: "seal", from: channel.from, to: channel.to, sentAt: ++this.now });
  }

  /** The coordinator seals one shard: record the store, record incoming channels, forward seals. */
  seal(shardId) {
    const shard = this.#shard(shardId);
    if (shard.sealed) throw new Error(`shard ${shardId} already sealed`);
    shard.sealed = true;
    shard.sealedAt = ++this.now;
    shard.snapshot = new Map(shard.store);
    for (const channel of this.channels) {
      if (channel.to === shard.id) {
        channel.recording = true;
        channel.sealedAtReceiver = shard.sealedAt;
      }
    }
    for (const channel of this.channels) if (channel.from === shard.id) this.#pushSeal(channel);
    return shard;
  }

  sealAll() { return this.shards.map((s) => this.seal(s.id)); }

  /** Deliver the head of a channel's FIFO; returns the message, or null when the queue is empty. */
  deliver(from, to) {
    const channel = this.channelOf(from, to);
    if (channel.queue.length === 0) return null;
    const message = channel.queue.shift();
    const at = ++this.now;
    if (message.kind === "seal") {
      message.deliveredAt = at;
      const shard = this.#shard(to);
      if (shard.sealed && channel.recording) {          // the channel's own marker closes it
        channel.recording = false;
        channel.stoppedAt = at;
      } else if (!shard.sealed) {
        this.seal(to);                                   // a seal that arrives first behaves like an initiator
      }
      return message;
    }
    message.deliveredAt = at;
    if (channel.recording) channel.recorded.push(message);
    const target = this.#shard(to);
    if (target.store.has(message.key)) throw new Error(`key ${message.key} already on shard ${to}`);
    target.store.set(message.key, message.value);
    this.#note(target);
    return message;
  }

  pending() { return this.channels.reduce((acc, c) => acc + c.queue.length, 0); }

  isComplete() { return this.shards.every((s) => s.sealed) && this.channels.every((c) => !c.recording); }

  snapshot() {
    return {
      shards: this.shards.map((s) => ({ id: s.id, entries: [...s.snapshot.entries()].sort(), sealedAt: s.sealedAt })),
      channels: this.channels.map((c) => ({
        from: c.from, to: c.to, messages: c.recorded.slice(), sealedAtReceiver: c.sealedAtReceiver, stoppedAt: c.stoppedAt,
      })),
    };
  }
}

/** Checks the Chandy-Lamport cut conditions and the exactly-once bookkeeping of every key. */
function verifySnapshot(sim, snapshot, allKeys) {
  const violations = [];
  for (const shard of snapshot.shards) {
    if (shard.sealedAt === null || shard.sealedAt === undefined) violations.push(`shard ${shard.id} never sealed`);
  }
  const channelSets = new Map(snapshot.channels.map((c) => [`${c.from}->${c.to}`, new Set(c.messages.map((m) => m.id))]));
  const sealedAt = new Map(snapshot.shards.map((s) => [s.id, s.sealedAt]));
  for (const message of sim.migrations) {
    const sentBefore = message.sentAt < sealedAt.get(message.from);
    const deliveredBefore = message.deliveredAt !== null && message.deliveredAt < sealedAt.get(message.to);
    const recorded = channelSets.get(`${message.from}->${message.to}`).has(message.id);
    if (sentBefore) {
      if (message.deliveredAt === null) violations.push(`migration ${message.id} was sent before its source sealed but is still in flight`);
      else if (!deliveredBefore && !recorded) violations.push(`migration ${message.id} crosses the cut but is missing from channel ${message.from}->${message.to}`);
      else if (deliveredBefore && recorded) violations.push(`migration ${message.id} was applied before the target sealed yet recorded in the channel`);
    } else {
      if (recorded) violations.push(`migration ${message.id} was sent after its source sealed but recorded in a channel`);
      if (deliveredBefore) violations.push(`migration ${message.id} crosses the cut backwards`);
    }
  }
  const places = new Map(allKeys.map((key) => [key, []]));
  for (const shard of snapshot.shards) {
    for (const [key] of shard.entries) places.get(key).push(`shard ${shard.id}`);
  }
  for (const channel of snapshot.channels) {
    for (const message of channel.messages) places.get(message.key).push(`channel ${channel.from}->${channel.to}`);
  }
  for (const [key, where] of places) {
    if (where.length !== 1) violations.push(`key ${key} appears ${where.length} times in the snapshot (${where.join(", ") || "nowhere"})`);
  }
  if (violations.length > 0) throw new Error(violations[0]);
  return true;
}

/** A naive cut: each shard's store is read at its own time and no channel state is recorded. */
function naiveSnapshot(sim, times) {
  return {
    shards: sim.shards.map((s) => ({ id: s.id, entries: [...sim.storeAt(s.id, times[s.id]).entries()].sort(), sealedAt: times[s.id] })),
    channels: sim.channels.map((c) => ({ from: c.from, to: c.to, messages: [], sealedAtReceiver: null, stoppedAt: null })),
  };
}

function xorshift32(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= (state << 13) >>> 0; state >>>= 0;
    state ^= state >>> 17;
    state ^= (state << 5) >>> 0; state >>>= 0;
    return state / 4294967296;
  };
}

module.exports = { CrossShardStore, verifySnapshot, naiveSnapshot };

if (require.main === module) {
  const assert = require("node:assert");
  const keysOf = (store) => store.shards.flatMap((s) => [...s.store.keys()]);

  // --- three shards in a ring, keys migrating while the snapshot is taken ---
  const store = new CrossShardStore(3, [[0, 1], [1, 2], [2, 0]]);
  store.put(0, "a", 1);
  store.put(0, "b", 2);
  store.put(1, "c", 3);
  store.put(2, "d", 4);
  const allKeys = ["a", "b", "c", "d"];
  const inFlight = store.migrate(0, 1, "a");            // sent before shard 0 seals: must land in a channel state
  store.sealAll();
  store.migrate(0, 1, "b");                             // sent after shard 0 sealed: a future event
  for (let step = 0; step < 60 && (store.pending() > 0 || !store.isComplete()); step++) {
    const [from, to] = [[0, 1], [1, 2], [2, 0]][step % 3];
    store.deliver(from, to);
  }
  assert.ok(store.isComplete(), "the coordinated snapshot must complete");
  assert.strictEqual(store.pending(), 0, "all seals and messages were delivered");
  const snapshot = store.snapshot();
  verifySnapshot(store, snapshot, allKeys);
  assert.deepStrictEqual(snapshot.shards[1].entries, [["c", 3]], "shard 1 sealed before any migration landed");
  assert.deepStrictEqual(snapshot.channels[0].messages.map((m) => m.key), ["a"], "the crossing migration is recorded once");
  assert.strictEqual(inFlight.deliveredAt !== null, true);
  const places = snapshot.shards.flatMap((s) => s.entries.map(([k]) => k))
    .concat(snapshot.channels.flatMap((c) => c.messages.map((m) => m.key)));
  assert.deepStrictEqual(places.sort(), [...allKeys].sort(), "every key appears exactly once");

  // --- the same cut without coordination loses or duplicates keys ---
  const naiveLost = naiveSnapshot(store, [inFlight.sentAt, inFlight.sentAt, inFlight.sentAt]);
  assert.throws(() => verifySnapshot(store, naiveLost, allKeys), /appears 0 times/,
    "a snapshot read while a migration is in flight loses that key");
  const naiveSplit = naiveSnapshot(store, [inFlight.sentAt - 1, store.now, store.now]);
  assert.throws(() => verifySnapshot(store, naiveSplit, allKeys), /appears 2 times|crosses the cut backwards/,
    "a staggered cut is rejected by the same checker");
  const duplicated = store.snapshot();
  duplicated.shards[1].entries = [["a", 1], ...duplicated.shards[1].entries];
  assert.throws(() => verifySnapshot(store, duplicated, allKeys), /appears 2 times/,
    "a key held in two places must be reported");

  // --- randomised migrations on a four-shard ring ---
  const next = xorshift32(0xbeef);
  const busy = new CrossShardStore(4, [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]);
  const busyKeys = [];
  for (let i = 0; i < 24; i++) {
    const key = `k${i}`;
    busy.put(i % 4, key, i);
    busyKeys.push(key);
  }
  for (let step = 0; step < 14; step++) {
    const key = busyKeys[Math.floor(next() * busyKeys.length)];
    const owner = busy.shards.findIndex((s) => s.store.has(key));
    if (owner === -1) continue;                                    // the key is in flight
    const targets = busy.channels.filter((c) => c.from === owner).map((c) => c.to);
    busy.migrate(owner, targets[Math.floor(next() * targets.length)], key);
  }
  busy.sealAll();
  for (let step = 0; step < 30; step++) {
    const key = busyKeys[Math.floor(next() * busyKeys.length)];
    const owner = busy.shards.findIndex((s) => s.store.has(key));
    if (owner === -1) continue;
    const targets = busy.channels.filter((c) => c.from === owner).map((c) => c.to);
    busy.migrate(owner, targets[Math.floor(next() * targets.length)], key);
  }
  const busyEdges = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]];
  for (let step = 0; step < 600 && (busy.pending() > 0 || !busy.isComplete()); step++) busy.deliver(...busyEdges[step % busyEdges.length]);
  assert.ok(busy.isComplete(), "the busy schedule must finish its snapshot");
  assert.strictEqual(busy.pending(), 0, "the busy store drained completely");
  const busySnapshot = busy.snapshot();
  verifySnapshot(busy, busySnapshot, busyKeys);
  assert.strictEqual(keysOf(busy).length, busyKeys.length, "migrations never duplicate a key");
  assert.ok(busySnapshot.channels.some((c) => c.messages.length > 0), "the busy cut must catch messages in flight");

  // --- degenerate and error cases ---
  const alone = new CrossShardStore(1, []);
  alone.put(0, "x", 1);
  alone.sealAll();
  assert.ok(alone.isComplete());
  verifySnapshot(alone, alone.snapshot(), ["x"]);
  assert.throws(() => alone.seal(0), /already sealed/);
  assert.throws(() => alone.put(0, "x", 2), /already on shard/);
  assert.throws(() => alone.migrate(0, 0, "x"), RangeError);
  assert.throws(() => new CrossShardStore(2, [[0, 1]]).migrate(0, 1, "k"), /not on shard/);
  assert.strictEqual(new CrossShardStore(2, [[0, 1]]).deliver(0, 1), null);
  assert.throws(() => new CrossShardStore(2, [[1, 1]]), RangeError);
  assert.throws(() => alone.put(0, "", 1), TypeError);

  console.log(`shards=${busy.shards.length} migrations=${busy.migrations.length} channels=${busy.channels.length}`);
  console.log("PASS");
}
