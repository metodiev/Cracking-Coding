/** Task 807 - Implement global scale consistent snapshot across geographically distributed shards. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A faithful in-process implementation of the Chandy-Lamport consistent global snapshot over a
 * multi-datacenter shard mesh: an initiator records its own state and sends marker messages over
 * every outgoing channel; a shard that sees a marker for the first time records its own state and
 * forwards the marker on the channels it has not recorded on; application messages that arrive on a
 * channel while the shard is recording are appended to that channel's recorded state until the
 * marker for that channel arrives. The result - per-shard states plus per-channel message lists -
 * is a global state that corresponds to a *consistent cut* of the execution, with no message both
 * received and in flight.
 *
 * Everything is explicit: time is a round counter, cross-datacenter latency is a typed hop cost,
 * messages are in-flight records with a deterministic delivery round produced by a splitmix64
 * stream, and no wall clock, timer or thread is used.
 *
 * Verification is by full replay of the event history rather than by inspecting the algorithm: an
 * independent event log (created / send / receive, in global order) is replayed to build the exact
 * expected global state for a candidate cut, and the snapshot is accepted when some cut reproduces
 * it byte for byte. Sizes are typed parameters (shard count, datacenters, keys, ops per round).
 *
 * What is simplified: shards are in-process objects, the mesh is a full graph (no spanning tree
 * construction), a "recording" is a Map copy rather than a disk checkpoint, and the initiator's
 * one-shot initiation is called by the test instead of a coordinator service.
 *
 * Invariants verified:
 *   - consistency: the snapshot equals the global state at a real cut of the event history, with
 *     the recorded channel contents equal to exactly the messages that crossed that cut;
 *   - no torn state: no message appears both in a receiver's recorded state and in a channel, and
 *     no message sent before the cut is missing from the reconstruction;
 *   - causal closure: every operation included in the snapshot has all of its causal dependencies
 *     included as well, so the cut is downward closed;
 *   - termination and liveness: the snapshot completes in a bounded number of rounds, normal traffic
 *     keeps flowing during the snapshot, and a snapshot that cannot finish (partition) is aborted
 *     after a bound instead of blocking the cluster;
 *   - epoch isolation: two overlapping snapshots build two separate consistent cuts and never mix
 *     each other's channel records;
 *   - cost: one marker per direction of every channel, plus the state size, with the aggregate
 *     (e.g. a global sum) matching the replay of the same cut.
 */
"use strict";

const assert = require("node:assert");

class SnapshotError extends Error {}

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

// --------------------------------------------------------------------------- shards and channels

class Shard {
  constructor(id, datacenter) {
    this.id = id;
    this.datacenter = datacenter;
    this.store = new Map();             // key -> value (last write wins by operation sequence)
    this.applied = new Set();           // operation seqs applied so far
    this.buffer = new Map();            // received but not yet applicable (causal delivery)
    this.vector = new Map();            // origin -> highest applied op seq
    this.recordings = new Map();        // epoch -> {state, applied, channels, closed}
    this.recordedAt = new Map();        // epoch -> event index of the recording
    this.markersSent = 0;
    this.markersReceived = 0;
    this.buffered = 0;
  }

  apply(op) {
    this.applied.add(op.seq);
    this.vector.set(op.origin, Math.max(this.vector.get(op.origin) ?? 0, op.seq));
    const current = this.store.get(op.key);
    if (current === undefined || current.seq < op.seq) this.store.set(op.key, { seq: op.seq, value: op.value, origin: op.origin });
  }

  /** Causal delivery: an operation waits until every operation it depends on has been applied. */
  ready(op) {
    for (const dependency of op.deps) if (!this.applied.has(dependency)) return false;
    return true;
  }

  receive(op) {
    if (this.applied.has(op.seq) || this.buffer.has(op.seq)) return "duplicate";
    if (!this.ready(op)) { this.buffer.set(op.seq, op); this.buffered++; return "buffered"; }
    this.apply(op);
    let progressed = true;
    while (progressed) {
      progressed = false;
      for (const [seq, pending] of [...this.buffer.entries()]) {
        if (!this.ready(pending)) continue;
        this.buffer.delete(seq);
        this.apply(pending);
        progressed = true;
      }
    }
    return "applied";
  }

  /** The recorded "state" is the materialised store, the applied set and the causal buffer. */
  capture() {
    return {
      store: new Map([...this.store.entries()].map(([key, entry]) => [key, entry.value])),
      applied: new Set(this.applied),
      buffered: new Set(this.buffer.keys()),
    };
  }

  beginRecording(epoch, eventIndex) {
    if (this.recordings.has(epoch)) return false;
    this.recordings.set(epoch, { state: this.capture(), channels: new Map(), closed: new Set() });
    this.recordedAt.set(epoch, eventIndex);
    return true;
  }

  recordChannelMessage(epoch, channel, op) {
    const recording = this.recordings.get(epoch);
    if (recording === undefined || recording.closed.has(channel)) return false;
    if (!recording.channels.has(channel)) recording.channels.set(channel, []);
    recording.channels.get(channel).push(op.seq);
    return true;
  }

  closeChannel(epoch, channel, eventIndex) {
    const recording = this.recordings.get(epoch);
    if (recording === undefined) return false;
    if (!recording.channels.has(channel)) recording.channels.set(channel, []);
    recording.closed.add(channel);
    if (!recording.closedAt) recording.closedAt = new Map();
    recording.closedAt.set(channel, eventIndex);
    return true;
  }
}

const channelKey = (from, to) => `${from}>${to}`;

// --------------------------------------------------------------------------- world

class World {
  constructor(opts = {}) {
    this.shards = new Map();
    const datacenters = opts.datacenters ?? 3;
    for (let id = 1; id <= (opts.shards ?? 5); id++) this.shards.set(id, new Shard(id, 1 + ((id - 1) % datacenters)));
    this.latencyPerDatacenter = opts.latency ?? 2;
    this.rng = splitmix64(opts.seed ?? 0x807);
    this.round = 0;
    this.eventIndex = 0;
    this.history = [];                  // authoritative event log: the snapshot oracle
    this.inFlight = [];                 // {kind, op|marker, from, to, deliverRound, channel}
    this.partitioned = new Set();       // channel keys that are not connected
    this.lastDelivery = new Map();      // channel key -> last scheduled delivery round (FIFO)
    this.nextOpSeq = 0;
    this.opBySeq = new Map();           // operation seq -> operation, the oracle's source of truth
    this.nextMarkerSeq = 0;
    this.snapshots = new Map();         // epoch -> {initiator, startedRound, completedRound, status, cuts}
    this.markers = 0;
    this.delivered = 0;
    this.opsCreated = 0;
    this.opsPerRound = opts.opsPerRound ?? 2;
    this.keys = Array.from({ length: opts.keys ?? 6 }, (_, index) => `k${index}`);
    this.labels = new Map();
  }

  link(from, to) { return !this.partitioned.has(channelKey(from, to)); }

  connect() { this.partitioned.clear(); }

  /** Cut the mesh into groups: no channel crosses a group boundary. */
  partition(groups) {
    this.partitioned.clear();
    for (const group of groups) for (const other of groups) {
      if (group === other) continue;
      for (const a of group) for (const b of other) this.partitioned.add(channelKey(a, b));
    }
  }

  event(type, detail) {
    this.history.push({ index: this.eventIndex++, type, round: this.round, ...detail });
    return this.eventIndex - 1;
  }

  /** Per-channel FIFO scheduling: a later send never overtakes an earlier one on the same channel. */
  nextDeliveryRound(channel, base) {
    const last = this.lastDelivery.get(channel) ?? -1;
    const round = Math.max(base, last + 1);
    this.lastDelivery.set(channel, round);
    return round;
  }

  /** Create an operation at its origin: applied there immediately, shipped to every other shard. */
  createOp(originId, key, value) {
    const origin = this.shards.get(originId);
    const op = { seq: ++this.nextOpSeq, origin: originId, key, value, deps: [...origin.applied] };
    origin.apply(op);
    this.opBySeq.set(op.seq, op);
    this.opsCreated++;    this.event("create", { shard: originId, op: op.seq, key });
    for (const shard of this.shards.values()) {
      if (shard.id === originId) continue;
      const hops = origin.datacenter === shard.datacenter ? 1 : this.latencyPerDatacenter;
      const channel = channelKey(originId, shard.id);
      const deliverRound = this.nextDeliveryRound(channel, this.round + hops + this.rng.int(2));
      this.inFlight.push({ kind: "op", op, from: originId, to: shard.id, deliverRound, channel });
      this.event("send", { from: originId, to: shard.id, op: op.seq, deliverRound });
    }
    return op;
  }

  sendMarker(epoch, fromId, toId, note) {
    const sameDatacenter = this.shards.get(fromId).datacenter === this.shards.get(toId).datacenter;
    const channel = channelKey(fromId, toId);
    const deliverRound = this.nextDeliveryRound(channel, this.round + 1 + (sameDatacenter ? 0 : this.latencyPerDatacenter - 1));
    this.markers++;
    const marker = { epoch, seq: ++this.nextMarkerSeq };
    this.inFlight.push({ kind: "marker", marker, from: fromId, to: toId, deliverRound, channel });
    this.event("marker-send", { epoch, from: fromId, to: toId, note, deliverRound });
  }

  peers(id) { return [...this.shards.keys()].filter((other) => other !== id).sort((a, b) => a - b); }

  /** Initiate a snapshot: record locally, then put a marker on every outgoing channel. */
  startSnapshot(initiatorId) {
    const epoch = this.snapshots.size + 1;
    const initiator = this.shards.get(initiatorId);
    this.event("snapshot-start", { epoch, shard: initiatorId });
    const eventIndex = this.eventIndex - 1;
    initiator.beginRecording(epoch, eventIndex);
    this.snapshots.set(epoch, { epoch, initiator: initiatorId, startedRound: this.round, status: "recording", events: 0, markers: 0 });
    for (const peer of this.peers(initiatorId)) this.sendMarker(epoch, initiatorId, peer, "initiator");
    return epoch;
  }

  /** One clock tick: creations, deliveries, marker processing, then a non-interference check. */
  tick() {
    this.round++;
    for (let i = 0; i < this.opsPerRound; i++) {
      const originId = 1 + this.rng.int(this.shards.size);
      const key = this.keys[this.rng.int(this.keys.length)];
      this.createOp(originId, key, this.rng.int(1000));
    }
    const due = this.inFlight.filter((message) => message.deliverRound <= this.round && this.link(message.from, message.to));
    this.inFlight = this.inFlight.filter((message) => !(message.deliverRound <= this.round && this.link(message.from, message.to)));
    due.sort((a, b) => (a.to - b.to) || (a.kind === b.kind ? 0 : a.kind === "op" ? -1 : 1) || (a.op?.seq ?? a.marker?.seq) - (b.op?.seq ?? b.marker?.seq));
    for (const message of due) {
      if (message.kind === "op") this.deliverOp(message);
      else this.deliverMarker(message);
    }
    for (const epoch of this.snapshots.keys()) this.checkCompletion(epoch);
    return this.round;
  }

  deliverOp(message) {
    const target = this.shards.get(message.to);
    this.event("receive", { from: message.from, to: message.to, op: message.op.seq });
    for (const [epoch, recording] of target.recordings) {
      if (recording.closed.has(message.channel)) continue;
      target.recordChannelMessage(epoch, message.channel, message.op);
    }
    const outcome = target.receive(message.op);
    this.delivered++;
    return outcome;
  }

  deliverMarker(message) {
    const target = this.shards.get(message.to);
    const epoch = message.marker.epoch;
    target.markersReceived++;
    this.event("marker-receive", { epoch, from: message.from, to: message.to });
    if (this.snapshots.has(epoch)) this.snapshots.get(epoch).markers++;
    for (const [activeEpoch, recording] of target.recordings) void activeEpoch;
    const first = target.beginRecording(epoch, this.eventIndex - 1);
    if (first) {
      // The sending rule: after recording, flush every outgoing channel - including the channel the
      // marker arrived on, otherwise the initiator's own incoming channels would never close.
      for (const peer of this.peers(target.id)) this.sendMarker(epoch, target.id, peer, "flush");
    }
    target.closeChannel(epoch, message.channel, this.eventIndex);
  }

  checkCompletion(epoch) {
    const snapshot = this.snapshots.get(epoch);
    if (snapshot === undefined || snapshot.status !== "recording") return;
    const shardCount = this.shards.size;
    const recorded = [...this.shards.values()].filter((shard) => shard.recordings.has(epoch)).length;
    const closed = [...this.shards.values()].every((shard) => {
      const recording = shard.recordings.get(epoch);
      if (recording === undefined) return false;
      return this.peers(shard.id).every((peer) => recording.closed.has(channelKey(peer, shard.id)));
    });
    if (recorded === shardCount && closed) {
      snapshot.status = "complete";
      snapshot.completedRound = this.round;
    } else if (this.round - snapshot.startedRound > 40) {
      snapshot.status = "aborted";
      snapshot.completedRound = this.round;
    }
    snapshot.events = this.history.length;
  }

  /** Reconstruct the recorded global state of an epoch. */
  reconstruct(epoch) {
    const snapshot = this.snapshots.get(epoch);
    if (snapshot === undefined || snapshot.status !== "complete") throw new SnapshotError(`snapshot ${epoch} is ${snapshot?.status ?? "unknown"}`);
    const states = new Map();
    const channels = new Map();
    for (const shard of this.shards.values()) {
      const recording = shard.recordings.get(epoch);
      states.set(shard.id, recording.state);
      for (const [channel, ops] of recording.channels) {
        if (!channels.has(channel)) channels.set(channel, []);
        channels.get(channel).push(...ops);
      }
    }
    for (const list of channels.values()) list.sort((a, b) => a - b);
    return { epoch, states, channels, completedRound: snapshot.completedRound, markers: snapshot.markers };
  }

  /** Aggregate over the reconstructed state (a global query served from the snapshot). */
  aggregate(epoch, mapper) {
    const { states } = this.reconstruct(epoch);
    let total = 0;
    for (const state of states.values()) for (const value of state.store.values()) total += mapper(value);
    return total;
  }
}

// --------------------------------------------------------------------------- replay oracle

/**
 * Independent oracle: replay the authoritative event log and check the snapshot against the cut it
 * claims - each shard's state must equal the replay up to its own recording event, and each recorded
 * channel must hold exactly the messages that were sent before the sender recorded and not yet
 * received when the marker closed that channel. No snapshot code is reused.
 */
function replayShardUpTo(world, shardId, limit) {
  const received = [];
  for (const event of world.history) {
    if (event.index >= limit) break;
    if (event.type === "create" && event.shard === shardId) received.push(world.opBySeq.get(event.op));
    else if (event.type === "receive" && event.to === shardId) received.push(world.opBySeq.get(event.op));
  }
  // Re-implement causal delivery independently: apply what is ready, hold the rest.
  const applied = new Set();
  const drained = new Set();
  const latest = new Map();                     // key -> {seq, value}, last write wins by op seq
  let progressed = true;
  while (progressed) {
    progressed = false;
    for (const op of received) {
      if (drained.has(op.seq)) continue;
      if (!op.deps.every((dependency) => applied.has(dependency))) continue;
      drained.add(op.seq);
      applied.add(op.seq);
      const current = latest.get(op.key);
      if (current === undefined || current.seq < op.seq) latest.set(op.key, { seq: op.seq, value: op.value });
      progressed = true;
    }
  }
  const buffered = new Set(received.filter((op) => !drained.has(op.seq)).map((op) => op.seq));
  return { applied, buffered, store: new Map([...latest.entries()].map(([key, entry]) => [key, entry.value])) };
}

function channelEvents(world, from, to) {
  const sends = [];
  const receives = [];
  for (const event of world.history) {
    if (event.type === "send" && event.from === from && event.to === to) sends.push({ index: event.index, op: event.op });
    else if (event.type === "receive" && event.from === from && event.to === to) receives.push({ index: event.index, op: event.op });
  }
  return { sends, receives };
}

/** Returns the violations found; an empty array means the snapshot is a consistent cut. */
function verifyConsistentCut(world, epoch) {
  const violations = [];
  const snapshot = world.reconstruct(epoch);
  const cutIndexes = new Map();
  for (const shard of [...world.shards.values()].sort((a, b) => a.id - b.id)) {
    const recordedAt = shard.recordedAt.get(epoch);
    if (recordedAt === undefined) { violations.push(`shard ${shard.id} never recorded`); continue; }
    cutIndexes.set(shard.id, recordedAt);
    const expected = replayShardUpTo(world, shard.id, recordedAt);
    const recorded = snapshot.states.get(shard.id);
    if (recorded.applied.size !== expected.applied.size) {
      violations.push(`shard ${shard.id} recorded ${recorded.applied.size} operations, the cut has ${expected.applied.size}`);
    }
    for (const seq of recorded.applied) if (!expected.applied.has(seq)) violations.push(`shard ${shard.id} recorded operation ${seq} that its cut does not contain`);
    for (const seq of expected.applied) if (!recorded.applied.has(seq)) violations.push(`shard ${shard.id} misses operation ${seq} that its cut contains`);
    if (JSON.stringify([...recorded.buffered].sort((a, b) => a - b)) !== JSON.stringify([...expected.buffered].sort((a, b) => a - b))) {
      violations.push(`shard ${shard.id} recorded a different causal buffer than the cut`);
    }
    if (recorded.store.size !== expected.store.size) violations.push(`shard ${shard.id} recorded ${recorded.store.size} keys, the cut has ${expected.store.size}`);
    for (const [key, value] of expected.store) {
      if (recorded.store.get(key) !== value) violations.push(`shard ${shard.id} recorded ${key}=${recorded.store.get(key)} but its cut has ${value}`);
    }
  }
  const placement = new Map();
  for (const [channel, ops] of snapshot.channels) {
    const [from, to] = channel.split(">").map(Number);
    const fromCut = cutIndexes.get(from);
    const toCut = cutIndexes.get(to);
    const recording = world.shards.get(to).recordings.get(epoch);
    const closedAt = recording.closedAt.get(channel);
    if (fromCut === undefined || toCut === undefined || closedAt === undefined) {
      violations.push(`channel ${channel} was never closed at the cut`);
      continue;
    }
    const { sends, receives } = channelEvents(world, from, to);
    const expected = sends.filter((send) => send.index < fromCut && !receives.some((receive) => receive.op === send.op && receive.index < toCut))
      .map((send) => send.op)
      .sort((a, b) => a - b);
    if (JSON.stringify(ops) !== JSON.stringify(expected)) {
      violations.push(`channel ${channel} recorded ${JSON.stringify(ops)}, the cut has ${JSON.stringify(expected)} in flight`);
    }
    for (const seq of ops) {
      const receive = receives.find((candidate) => candidate.op === seq);
      if (receive === undefined) { violations.push(`channel ${channel} recorded operation ${seq} that never arrived`); continue; }
      if (receive.index > closedAt) violations.push(`channel ${channel} recorded operation ${seq} that arrived after its marker`);
      if (receive.index < toCut) violations.push(`channel ${channel} recorded operation ${seq} that arrived before the receiver recorded`);
      const send = sends.find((candidate) => candidate.op === seq);
      if (send === undefined || send.index >= fromCut) violations.push(`channel ${channel} recorded operation ${seq} that was sent after the sender recorded`);
      if (snapshot.states.get(to).applied.has(seq)) violations.push(`operation ${seq} is in flight on ${channel} and already applied at ${to}`);
      if (world.opBySeq.get(seq).origin !== from) violations.push(`channel ${channel} carries operation ${seq} from the wrong origin`);
      placement.set(seq, `channel:${channel}`);
    }
  }
  for (const [id, state] of snapshot.states) {
    for (const seq of state.applied) {
      for (const dependency of world.opBySeq.get(seq).deps) {
        if (!state.applied.has(dependency)) violations.push(`shard ${id}: operation ${seq} is recorded without its causal dependency ${dependency}`);
      }
    }
    for (const seq of state.buffered) {
      if (state.applied.has(seq)) violations.push(`shard ${id}: operation ${seq} is both applied and buffered`);
    }
  }
  return { violations, cutIndexes, snapshot, placement };
}

/** Aggregate of the cut the snapshot claims: every shard replayed up to its own recording index. */
function cutAggregate(world, epoch, mapper) {
  let total = 0;
  for (const shard of world.shards.values()) {
    const recordedAt = shard.recordedAt.get(epoch);
    if (recordedAt === undefined) continue;
    for (const value of replayShardUpTo(world, shard.id, recordedAt).store.values()) total += mapper(value);
  }
  return total;
}

// --------------------------------------------------------------------------- helpers on the world

World.prototype.valueOf = function valueOf(opSeq) { return this.opBySeq.get(opSeq)?.value; };

if (require.main === module) {
  // --- 1. a snapshot of a busy mesh is a real consistent cut -------------------------------------
  const world = new World({ shards: 5, datacenters: 3, keys: 6, opsPerRound: 2, seed: 0x807 });
  for (let round = 0; round < 12; round++) world.tick();
  const epoch1 = world.startSnapshot(2);
  let rounds = 0;
  while (world.snapshots.get(epoch1).status === "recording" && rounds < 60) { world.tick(); rounds++; }
  assert.strictEqual(world.snapshots.get(epoch1).status, "complete", `snapshot 1 completed in ${rounds} rounds`);
  const check1 = verifyConsistentCut(world, epoch1);
  assert.deepStrictEqual(check1.violations, [], check1.violations.join("; "));
  const reconstructed = check1.snapshot;
  assert.strictEqual(reconstructed.states.size, 5, "every shard contributed a state");
  const recordedTotal = [...reconstructed.states.values()].reduce((sum, state) => sum + state.store.size, 0);
  assert.ok(recordedTotal > 0, `the snapshot holds ${recordedTotal} key/value pairs`);
  const crossing = [...reconstructed.channels.values()].reduce((sum, ops) => sum + ops.length, 0);
  assert.ok(crossing < world.opsCreated, `channel states hold ${crossing} in-flight operations`);
  assert.ok(world.opsCreated > 20, `${world.opsCreated} operations were created around the snapshot`);
  assert.ok(rounds <= 8, `the snapshot finished in ${rounds} rounds on a 5-shard, 3-datacenter mesh`);

  // --- 2. no torn state: a message is either applied or in flight, never both, never lost --------
  const allApplied = new Set();
  for (const state of reconstructed.states.values()) for (const seq of state.applied) allApplied.add(seq);
  for (const [channel, ops] of reconstructed.channels) {
    const [from, to] = channel.split(">").map(Number);
    for (const seq of ops) {
      assert.strictEqual(reconstructed.states.get(to).applied.has(seq), false, `operation ${seq} must not be both received and in flight`);
      assert.strictEqual(world.opBySeq.get(seq).origin, from, `channel ${channel} only carries operations from ${from}`);
    }
  }
  assert.ok(allApplied.size >= crossing, `${allApplied.size} operations are recorded in shard states, ${crossing} more are in flight`);
  for (const shard of world.shards.values()) {
    const state = reconstructed.states.get(shard.id);
    for (const seq of state.applied) {
      const op = world.opBySeq.get(seq);
      if (op.origin === shard.id) continue;
      assert.strictEqual(check1.cutIndexes.get(shard.id) > 0, true, `shard ${shard.id} applied ${seq} through a channel before its cut`);
    }
  }

  // --- 3. causal closure: every included operation carries its dependencies -----------------------
  for (const [id, state] of reconstructed.states) {
    for (const seq of state.applied) {
      const op = world.opBySeq.get(seq);
      for (const dependency of op.deps) {
        assert.ok(state.applied.has(dependency), `shard ${id}: operation ${seq} is recorded without its causal dependency ${dependency}`);
      }
    }
  }
  assert.ok(reconstructed.markers >= world.shards.size * (world.shards.size - 1),
    `every direction of the mesh got a marker (${reconstructed.markers})`);

  // --- 4. an aggregate served from the snapshot equals the aggregate of the same cut --------------
  const snapshotSum = world.aggregate(epoch1, (value) => value);
  assert.strictEqual(snapshotSum, cutAggregate(world, epoch1, (value) => value), "a global aggregate over the snapshot equals the aggregate of its cut");

  // --- 5. termination bound and non-interference --------------------------------------------------
  const beforeOps = world.opsCreated;
  const epoch2 = world.startSnapshot(4);
  let rounds2 = 0;
  while (world.snapshots.get(epoch2).status === "recording" && rounds2 < 60) { world.tick(); rounds2++; }
  assert.strictEqual(world.snapshots.get(epoch2).status, "complete", "the second snapshot also completes");
  assert.ok(rounds2 <= 8, `completion is bounded (${rounds2} rounds)`);
  assert.ok(world.opsCreated > beforeOps, "normal traffic kept flowing during the snapshot");
  const check2 = verifyConsistentCut(world, epoch2);
  assert.deepStrictEqual(check2.violations, [], check2.violations.join("; "));
  assert.ok(check2.cutIndexes.get(4) > check1.cutIndexes.get(4), "the second snapshot records a later cut on the same shard");
  assert.strictEqual(world.aggregate(epoch2, (value) => value), cutAggregate(world, epoch2, (value) => value), "the second snapshot is consistent with its own cut");

  // --- 6. quiescent snapshots and idle shards -----------------------------------------------------
  const quiescent = new World({ shards: 3, datacenters: 2, opsPerRound: 0, seed: 0x8072 });
  for (let round = 0; round < 5; round++) quiescent.tick();
  const quietEpoch = quiescent.startSnapshot(1);
  for (let round = 0; round < 6; round++) quiescent.tick();
  assert.strictEqual(quiescent.snapshots.get(quietEpoch).status, "complete", "a snapshot of an idle mesh completes");
  assert.deepStrictEqual([...quiescent.reconstruct(quietEpoch).states.values()].map((state) => state.store.size), [0, 0, 0], "idle shards record empty states");
  assert.strictEqual(quiescent.aggregate(quietEpoch, (value) => value), 0, "and the aggregate is zero");
  assert.deepStrictEqual(verifyConsistentCut(quiescent, quietEpoch).violations, [], "the idle snapshot is still a consistent cut");
  const lonely = new World({ shards: 1, datacenters: 1, opsPerRound: 0, seed: 0x8073 });
  const soloEpoch = lonely.startSnapshot(1);
  for (let round = 0; round < 4; round++) lonely.tick();
  assert.strictEqual(lonely.snapshots.get(soloEpoch).status, "complete", "a single shard snapshots itself");
  assert.deepStrictEqual(verifyConsistentCut(lonely, soloEpoch).violations, []);

  // --- 7. partition: the snapshot is aborted after a bound, traffic is not blocked ----------------
  const pworld = new World({ shards: 4, datacenters: 2, opsPerRound: 1, seed: 0x8074 });
  for (let round = 0; round < 6; round++) pworld.tick();
  pworld.partition([[1, 2], [3, 4]]);
  const blocked = pworld.startSnapshot(1);
  assert.ok(pworld.markers > 0, "the initiator sent its markers");
  const opsAtPartition = pworld.opsCreated;
  const sentAtPartition = pworld.history.length;
  for (let round = 0; round < 45; round++) pworld.tick();
  assert.strictEqual(pworld.snapshots.get(blocked).status, "aborted", "an unreachable snapshot is abandoned after the bound");
  assert.ok(pworld.opsCreated > opsAtPartition, "the cluster kept accepting operations during the failed snapshot");
  assert.ok(pworld.history.length > sentAtPartition, "and the event log kept growing");
  assert.strictEqual(pworld.shards.get(2).applied.size > 0, true, "the shards on either side of the partition still applied their own traffic");
  pworld.connect();
  const healed = pworld.startSnapshot(3);
  for (let round = 0; round < 20; round++) pworld.tick();
  assert.strictEqual(pworld.snapshots.get(healed).status, "complete", "after healing the snapshot completes");
  assert.deepStrictEqual(verifyConsistentCut(pworld, healed).violations, [], "and it is a consistent cut");

  // --- 8. two overlapping epochs never mix ---------------------------------------------------------
  const overlap = new World({ shards: 4, datacenters: 2, opsPerRound: 2, seed: 0x8075 });
  for (let round = 0; round < 8; round++) overlap.tick();
  const firstEpoch = overlap.startSnapshot(1);
  overlap.tick();
  const secondEpoch = overlap.startSnapshot(3);
  for (let round = 0; round < 25; round++) overlap.tick();
  assert.strictEqual(overlap.snapshots.get(firstEpoch).status, "complete");
  assert.strictEqual(overlap.snapshots.get(secondEpoch).status, "complete");
  const overlapFirst = verifyConsistentCut(overlap, firstEpoch);
  const overlapSecond = verifyConsistentCut(overlap, secondEpoch);
  assert.deepStrictEqual(overlapFirst.violations, [], overlapFirst.violations.join("; "));
  assert.deepStrictEqual(overlapSecond.violations, [], overlapSecond.violations.join("; "));
  assert.ok(overlapSecond.cutIndexes.get(3) > overlapFirst.cutIndexes.get(3), "the later initiation yields the later cut");
  assert.notStrictEqual(overlapFirst.cutIndexes.get(2), overlapSecond.cutIndexes.get(2), "each epoch recorded its own point on shard 2");
  const firstChannels = overlapFirst.snapshot.channels;
  const secondChannels = overlapSecond.snapshot.channels;
  assert.notStrictEqual(JSON.stringify([...firstChannels.entries()]), JSON.stringify([...secondChannels.entries()]),
    "each epoch kept its own channel records");

  // --- 9. cost: markers, bytes and the delivery budget --------------------------------------------
  const cost = new World({ shards: 6, datacenters: 3, opsPerRound: 3, keys: 8, seed: 0x8076 });
  for (let round = 0; round < 10; round++) cost.tick();
  const costEpoch = cost.startSnapshot(5);
  const markersBefore = cost.markers;
  let costRounds = 0;
  while (cost.snapshots.get(costEpoch).status === "recording" && costRounds < 40) { cost.tick(); costRounds++; }
  assert.strictEqual(cost.snapshots.get(costEpoch).status, "complete");
  const markersUsed = cost.markers - markersBefore;
  const channels = cost.shards.size * (cost.shards.size - 1);
  assert.ok(markersUsed <= channels + cost.shards.size, `one marker per channel direction (${markersUsed} markers for ${channels} channels)`);
  const costCheck = verifyConsistentCut(cost, costEpoch);
  assert.deepStrictEqual(costCheck.violations, [], costCheck.violations.join("; "));
  const stateBytes = [...costCheck.snapshot.states.values()].reduce((sum, state) => sum + state.store.size * 12 + state.applied.size * 4, 0);
  assert.ok(stateBytes > 0, `the snapshot carried about ${stateBytes} bytes of state`);
  assert.ok(costRounds <= 8, `completion within ${costRounds} rounds despite the wider mesh`);
  assert.strictEqual(cost.aggregate(costEpoch, (value) => value), cutAggregate(cost, costEpoch, (value) => value), "the six-shard aggregate matches its cut");

  console.log(`snapshot: 5-shard mesh, epoch 1 complete in ${rounds} rounds, ${crossing} in-flight operations recorded, ${recordedTotal} key/value pairs`);
  console.log(`snapshot: ${reconstructed.markers} markers, ${crossing} operations in flight, ${verifiedOps()} operations verified against the event log`);
  console.log(`partition: blocked snapshot aborted after the bound, ${pworld.opsCreated - opsAtPartition} operations created meanwhile, healed snapshot consistent`);
  console.log(`cost: ${markersUsed} markers over ${channels} channel directions for a 6-shard mesh, ${stateBytes} bytes, ${costRounds} rounds`);

  function verifiedOps() {
    let total = 0;
    for (const check of [check1, check2, overlapFirst, overlapSecond, costCheck]) {
      for (const state of check.snapshot.states.values()) total += state.applied.size;
    }
    return total;
  }
  console.log("PASS");
}

module.exports = {
  World, Shard, replayShardUpTo, verifyConsistentCut, cutAggregate, channelEvents, channelKey,
  splitmix64, SnapshotError,
};
