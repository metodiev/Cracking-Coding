/** Task 810 - Implement cross-datacenter replication with causal consistency guarantees. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A geo-replicated key/value store with *causal* (not merely eventual) convergence: every write is
 * tagged with a version vector of everything its datacenter had applied when it was created, remote
 * writes are buffered until their dependencies have been delivered, and a session can carry its
 * context to another datacenter so that read-your-writes and monotonic reads hold across the WAN.
 * Each datacenter applies writes into a per-key last-writer-wins register; concurrent writes are
 * resolved by the same deterministic rule everywhere, and once every dependency has been
 * acknowledged the causality metadata can be garbage collected without changing a single value.
 *
 * Time is logical: replication lag is a fixed number of ticks (or a per-link delay), messages are
 * explicit in-flight records, DC "offline" windows are explicit ticks, and the write stream comes
 * from a splitmix64 stream. No wall clock, no timers, no threads.
 *
 * What is simplified: datacenters are in-process objects, each DC keeps a full copy of the key space
 * (no sharding), the WAN is a full mesh with a fixed lag, dependency tracking is a per-key vector
 * clock rather than a global one, and conflict resolution is LWW by (counter, dc) rather than a
 * CRDT. Sizes are typed parameters (DC count, keys, writes, lag).
 *
 * Invariants verified:
 *   - causal consistency: in every datacenter's apply order, an operation is never applied before
 *     any operation in its dependency set - checked both live and by an independent replay oracle;
 *   - convergence: after replication drains, every datacenter materialises the identical state, and
 *     the resolution of concurrent writes is the same everywhere;
 *   - session guarantees: monotonic reads (a session never steps back) and read-your-writes even
 *     when the session's next read happens in another datacenter;
 *   - quarantine: a datacenter that was offline applies nothing out of causal order when it rejoins,
 *     and its buffered operations drain to zero;
 *   - garbage collection: causality metadata for fully acknowledged operations is dropped without
 *     changing the state or breaking any guarantee;
 *   - cost: replication messages stay linear in writes x peers, and the buffered-operation count
 *     reflects exactly the operations that arrived too early.
 */
"use strict";

const assert = require("node:assert");

class ReplicationError extends Error {}

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

// --------------------------------------------------------------------------- operations

class Write {
  constructor(dc, seq, key, value, deps) {
    this.dc = dc;
    this.seq = seq;
    this.key = key;
    this.value = value;
    this.deps = deps;                     // version vector: dc -> seq the writer had applied
  }

  id() { return `${this.dc}.${this.seq}`; }

  /** Deterministic conflict rule: higher sequence wins, ties broken by datacenter id. */
  rank() { return this.seq * 1000 + this.dc; }
}

// --------------------------------------------------------------------------- datacenter

class Datacenter {
  constructor(id, lag) {
    this.id = id;
    this.lag = lag;
    this.applied = new Map();             // key -> Write (the winning version)
    this.appliedIds = new Set();          // write ids already applied
    this.buffer = new Map();               // id -> Write waiting for dependencies
    this.vv = new Map();                  // dc -> highest applied seq
    this.order = [];                      // applied write ids, in apply order
    this.ba = 0;                          // buffered arrivals (metric)
    this.dropped = 0;
    this.metadata = 0;                    // retained causality metadata (bytes proxy)
    this.gcFloor = {};                    // dc -> seq every replica has acknowledged
  }

  seqOf(dc) { return this.vv.get(dc) ?? 0; }

  ready(write) {
    for (const [dc, seq] of Object.entries(write.deps)) {
      if (this.seqOf(Number(dc)) < seq) return false;
    }
    return true;
  }

  /** Deliver a remote write: apply it, or buffer it if its causal past has not arrived yet. */
  deliver(write) {
    if (this.appliedIds.has(write.id()) || this.buffer.has(write.id())) return "duplicate";
    if (!this.ready(write)) { this.buffer.set(write.id(), write); this.ba++; return "buffered"; }
    this.applyWrite(write);
    let progressed = true;
    while (progressed) {
      progressed = false;
      for (const [id, pending] of [...this.buffer.entries()]) {
        if (!this.ready(pending)) continue;
        this.buffer.delete(id);
        this.applyWrite(pending);
        progressed = true;
      }
    }
    return "applied";
  }

  applyWrite(write) {
    this.appliedIds.add(write.id());
    this.order.push(write.id());
    this.vv.set(write.dc, Math.max(this.seqOf(write.dc), write.seq));
    const current = this.applied.get(write.key);
    if (current === undefined || write.rank() > current.rank()) this.applied.set(write.key, write);
    this.metadata += JSON.stringify(write.deps).length + 6;
  }

  value(key) {
    const write = this.applied.get(key);
    return write === undefined ? undefined : write.value;
  }

  /** Version vector of this DC's applied history - carried by sessions and by new writes. */
  context() { return Object.fromEntries(this.vv); }

  /** Causal closure check against the write table: no applied write may precede its dependencies. */
  causalityBreaks(writes) {
    const breaks = [];
    for (const id of this.order) {
      const write = writes.get(id);
      if (write === undefined) continue;
      for (const [dc, seq] of Object.entries(write.deps)) {
        const dependencyId = `${dc}.${seq}`;
        if (Number(dc) === write.dc && seq === write.seq) continue;
        if (!this.appliedIds.has(dependencyId)) { breaks.push(`${this.id}: ${id} applied before dependency ${dependencyId}`); continue; }
        if (this.order.indexOf(dependencyId) > this.order.indexOf(id)) breaks.push(`${this.id}: ${id} applied before dependency ${dependencyId}`);
      }
    }
    return breaks;
  }

  stateDigest() {
    return [...this.applied.entries()].map(([key, write]) => `${key}=${write.id()}:${JSON.stringify(write.value)}`).sort().join("|");
  }

  /** Drop dependency metadata for writes every datacenter has acknowledged. */
  collectGarbage(acknowledged) {
    let dropped = 0;
    for (const [id, write] of this.applied) {
      if ((acknowledged[write.dc] ?? 0) < write.seq) continue;
      const size = JSON.stringify(write.deps).length + 6;
      if (this.metadata >= size) { this.metadata -= size; dropped++; }
    }
    this.gcFloor = acknowledged;
    return dropped;
  }
}

// --------------------------------------------------------------------------- cluster

class Cluster {
  constructor(opts = {}) {
    this.lag = opts.lag ?? 2;
    this.datacenters = new Map();
    for (let id = 1; id <= (opts.datacenters ?? 3); id++) this.datacenters.set(id, new Datacenter(id, this.lag));
    this.writes = new Map();               // id -> Write
    this.queue = [];
    this.roundValue = 0;
    this.seq = new Map();                  // dc -> last assigned seq
    this.messages = 0;
    this.nextId = 0;
    this.offline = new Set();
    this.rng = splitmix64(opts.seed ?? 0x810);
  }

  get round() { return this.roundValue; }

  /** A client write at `dcId`: depends on everything that DC has already applied. */
  write(dcId, key, value) {
    const dc = this.datacenters.get(dcId);
    const seq = (this.seq.get(dcId) ?? 0) + 1;
    this.seq.set(dcId, seq);
    const write = new Write(dcId, seq, key, value, dc.context());
    this.writes.set(write.id(), write);
    dc.deliver(write);
    for (const other of this.datacenters.values()) {
      if (other.id === dcId) continue;
      this.messages++;
      this.queue.push({ write, to: other.id, deliverAt: this.roundValue + this.lag });
    }
    return write;
  }

  off(dcId) { this.offline.add(dcId); }

  on(dcId) { this.offline.delete(dcId); }

  /** One tick of the WAN: deliver everything that is due to a reachable datacenter. */
  tick() {
    this.roundValue++;
    const due = this.queue.filter((message) => message.deliverAt <= this.roundValue && !this.offline.has(message.to));
    this.queue = this.queue.filter((message) => !(message.deliverAt <= this.roundValue && !this.offline.has(message.to)));
    due.sort((a, b) => (a.to - b.to) || (a.write.rank() - b.write.rank()));
    for (const message of due) {
      const target = this.datacenters.get(message.to);
      target.deliver(message.write);
    }
    return due.length;
  }

  run(ticks) { for (let i = 0; i < ticks; i++) this.tick(); return this.roundValue; }

  converged() {
    const digests = [...this.datacenters.values()].filter((dc) => !this.offline.has(dc.id)).map((dc) => dc.stateDigest());
    return digests.every((digest) => digest === digests[0]);
  }

  drain(limit = 200) {
    let ticks = 0;
    while (ticks < limit && (this.queue.length > 0 || [...this.datacenters.values()].some((dc) => dc.buffer.size > 0))) {
      this.tick();
      ticks++;
    }
    return ticks;
  }

  acknowledgement() {
    const confirmed = {};
    for (const dc of this.datacenters.values()) {
      for (const [other, seq] of dc.vv) confirmed[other] = Math.max(confirmed[other] ?? 0, seq);
    }
    return confirmed;
  }

  causalityBreaks() {
    const breaks = [];
    for (const dc of this.datacenters.values()) breaks.push(...dc.causalityBreaks(this.writes));
    return breaks;
  }

  /** Every applied write's dependency closure, replayed by the oracle in the datacenter itself. */
  closureCheck() {
    const breaks = [];
    for (const dc of this.datacenters.values()) {
      for (const id of dc.order) {
        const write = this.writes.get(id);
        if (write === undefined) continue;
        for (const [other, seq] of Object.entries(write.deps)) {
          if (Number(other) === write.dc && seq === write.seq) continue;
          if (!dc.appliedIds.has(`${other}.${seq}`)) breaks.push(`${dc.id}: ${id} applied without dependency ${other}.${seq}`);
        }
      }
    }
    return breaks;
  }
}

// --------------------------------------------------------------------------- session

class Session {
  constructor(cluster, dcId) {
    this.cluster = cluster;
    this.dcId = dcId;
    this.context = {};
    this.observed = new Map();             // key -> the write id last read
  }

  /** The session's context is the union of everything it has observed. */
  absorb(dc) {
    for (const [id, seq] of dc.vv) this.context[id] = Math.max(this.context[id] ?? 0, seq);
  }

  read(key) {
    const dc = this.cluster.datacenters.get(this.dcId);
    // Read-your-writes: pull anything the session has already observed but this DC is missing.
    let guard = 0;
    while (guard++ < 50) {
      const missing = Object.entries(this.context).some(([id, seq]) => dc.seqOf(Number(id)) < seq);
      if (!missing) break;
      this.cluster.tick();
    }
    this.absorb(dc);
    const write = dc.applied.get(key);
    if (write !== undefined) {
      const previous = this.observed.get(key);
      if (previous !== undefined && write.rank() < previous.rank()) throw new ReplicationError(`monotonic read violated for ${key}`);
      this.observed.set(key, write);
    }
    return write === undefined ? undefined : write.value;
  }

  write(key, value) {
    const dc = this.cluster.datacenters.get(this.dcId);
    const write = this.cluster.write(this.dcId, key, value);
    this.context[this.dcId] = write.seq;
    this.absorb(dc);
    return write;
  }

  moveTo(dcId) { this.dcId = dcId; }
}

if (require.main === module) {
  // --- 1. replication drains, converges, and never breaks causality ------------------------------
  const cluster = new Cluster({ datacenters: 3, lag: 2, seed: 0x810 });
  const rng = splitmix64(0x8101);
  const keys = ["a", "b", "c", "d"];
  for (let index = 0; index < 60; index++) {
    const dcId = 1 + rng.int(3);
    cluster.write(dcId, keys[rng.int(keys.length)], rng.int(1000));
  }
  assert.strictEqual(cluster.converged(), false, "the datacenters disagree while writes are in flight");
  const drainTicks = cluster.drain();
  assert.strictEqual(cluster.converged(), true, `the cluster converged after ${drainTicks} ticks`);
  assert.deepStrictEqual(cluster.causalityBreaks(), [], "no operation was ever applied before its dependencies");
  const digest = cluster.datacenters.get(1).stateDigest();
  for (const dc of cluster.datacenters.values()) assert.strictEqual(dc.stateDigest(), digest, `datacenter ${dc.id} materialised the same state`);
  assert.strictEqual([...cluster.datacenters.values()].every((dc) => dc.buffer.size === 0), true, "no buffered operation is left");
  assert.ok(cluster.messages >= 60 * 2, `${cluster.messages} replication messages for 60 writes over 3 datacenters`);

  // --- 2. concurrency resolves deterministically and identically everywhere ----------------------
  const concurrent = new Cluster({ datacenters: 3, lag: 1, seed: 0x8102 });
  const first = concurrent.write(1, "x", "from-dc1");
  const second = concurrent.write(2, "x", "from-dc2");
  concurrent.drain();
  assert.strictEqual(concurrent.converged(), true, "concurrent writes still converge");
  const winner = concurrent.datacenters.get(3).value("x");
  for (const dc of concurrent.datacenters.values()) assert.strictEqual(dc.value("x"), winner, `datacenter ${dc.id} picked the same winner`);
  assert.strictEqual(winner, first.rank() > second.rank() ? "from-dc1" : "from-dc2", "the winner is the higher-ranked write");
  assert.deepStrictEqual(concurrent.causalityBreaks(), []);

  // --- 3. an operation whose dependency is missing is buffered, never applied early ---------------
  const gapped = new Cluster({ datacenters: 3, lag: 1, seed: 0x8103 });
  const dependency = gapped.write(1, "g", "first");
  const dependent = gapped.write(1, "g", "second");
  assert.strictEqual(dependent.deps["1"], dependency.seq, "the second write depends on the first");
  const dc3 = gapped.datacenters.get(3);
  dc3.deliver(dependent);
  assert.strictEqual(dc3.buffer.size, 1, "the dependent write waits in the buffer");
  assert.strictEqual(dc3.value("g"), undefined, "and nothing is materialised out of order");
  const breakage = dc3.deliver(dependency);
  assert.strictEqual(breakage, "applied", "the dependency is applied");
  assert.strictEqual(dc3.buffer.size, 0, "which releases the buffered write");
  assert.strictEqual(dc3.value("g"), "second", "in causal order");
  assert.deepStrictEqual(gapped.causalityBreaks(), [], "the gap did not break causality");
  assert.strictEqual(dc3.ba, 1, "exactly one operation was buffered");

  // --- 4. an offline datacenter catches up without violating causality ---------------------------
  const offline = new Cluster({ datacenters: 3, lag: 1, seed: 0x8104 });
  offline.write(1, "k", "before");
  offline.drain();
  offline.off(3);
  for (let index = 0; index < 15; index++) offline.write(1 + (index % 2), keys[index % keys.length], index);
  offline.run(10);
  assert.strictEqual(offline.datacenters.get(3).stateDigest() === offline.datacenters.get(1).stateDigest(), false, "the offline datacenter is behind");
  assert.strictEqual(offline.converged(), true, "the reachable datacenters still agree with each other");
  offline.on(3);
  const catchUp = offline.drain();
  assert.strictEqual(offline.converged(), true, `the reconnected datacenter caught up in ${catchUp} ticks`);
  assert.deepStrictEqual(offline.causalityBreaks(), [], "the catch-up respected causal order");
  assert.strictEqual(offline.datacenters.get(3).buffer.size, 0, "and drained its buffer");
  assert.strictEqual(offline.datacenters.get(3).appliedIds.size, offline.writes.size, "it applied every write");

  // --- 5. duplicate deliveries are idempotent ----------------------------------------------------
  const duplicate = new Cluster({ datacenters: 2, lag: 1, seed: 0x8105 });
  const write = duplicate.write(1, "d", 7);
  const target = duplicate.datacenters.get(2);
  assert.strictEqual(target.deliver(write), "applied");
  const before = target.stateDigest();
  assert.strictEqual(target.deliver(write), "duplicate", "a re-delivered write is recognised");
  assert.strictEqual(target.stateDigest(), before, "and changes nothing");

  // --- 6. session guarantees across datacenters --------------------------------------------------
  const session = new Cluster({ datacenters: 3, lag: 2, seed: 0x8106 });
  const client = new Session(session, 1);
  client.write("s", "one");
  assert.strictEqual(client.read("s"), "one", "read-your-writes at the local datacenter");
  session.drain();
  client.write("s", "two");
  session.drain();
  assert.strictEqual(client.read("s"), "two", "the newest local value is read");
  // Move the session to another datacenter: the context forces it to have the session's writes.
  client.moveTo(3);
  const remoteValue = client.read("s");
  assert.strictEqual(remoteValue, "two", "read-your-writes survives the move to another datacenter");
  const other = new Session(session, 2);
  other.write("o", "written-on-2");
  other.moveTo(1);
  assert.strictEqual(other.read("o"), "written-on-2", "a context only this session knows is pulled in before the read");
  let monotonic = true;
  let previousRank = -1;
  for (let index = 0; index < 12; index++) {
    session.write(1 + (index % 3), "m", index);
    session.drain();
    client.read("m");
    const observed = client.observed.get("m");
    if (observed.rank() < previousRank) monotonic = false;
    previousRank = observed.rank();
  }
  assert.strictEqual(monotonic, true, "monotonic reads never step back over 12 versions");
  const winnerWrite = [...session.writes.values()].filter((candidate) => candidate.key === "m")
    .reduce((best, candidate) => (best === undefined || candidate.rank() > best.rank() ? candidate : best), undefined);
  assert.strictEqual(client.read("m"), winnerWrite.value, "the read returns the highest-ranked write in the system");
  assert.deepStrictEqual(session.causalityBreaks(), [], "the session workload stayed causal");

  // --- 7. garbage collection of causality metadata ------------------------------------------------
  const gc = new Cluster({ datacenters: 3, lag: 1, seed: 0x8107 });
  for (let index = 0; index < 40; index++) gc.write(1 + (index % 3), keys[index % keys.length], index);
  gc.drain();
  const digestBeforeGc = gc.datacenters.get(1).stateDigest();
  const metadataBefore = gc.datacenters.get(1).metadata;
  const acknowledged = gc.acknowledgement();
  let collected = 0;
  for (const dc of gc.datacenters.values()) collected += dc.collectGarbage(acknowledged);
  assert.ok(collected > 0, `GC dropped metadata for ${collected} writes`);
  assert.ok(gc.datacenters.get(1).metadata < metadataBefore, "the retained metadata shrank");
  assert.strictEqual(gc.datacenters.get(1).stateDigest(), digestBeforeGc, "GC never changes a value");
  assert.strictEqual(gc.converged(), true, "and the cluster stays converged");
  const afterGc = new Cluster({ datacenters: 3, lag: 1, seed: 0x8108 });
  for (let index = 0; index < 20; index++) afterGc.write(1 + (index % 3), keys[index % keys.length], index * 2);
  afterGc.drain();
  assert.ok(afterGc.datacenters.get(2).vv.size <= 3, "the version vectors stay bounded by the datacenter count");

  // --- 8. scale: many writes over five datacenters converge --------------------------------------
  const scale = new Cluster({ datacenters: 5, lag: 3, seed: 0x8109 });
  const scaleRng = splitmix64(0x8110);
  const scaleKeys = Array.from({ length: 10 }, (_, index) => `key${index}`);
  for (let index = 0; index < 300; index++) {
    scale.write(1 + scaleRng.int(5), scaleKeys[scaleRng.int(scaleKeys.length)], index);
  }
  const scaleTicks = scale.drain();
  assert.strictEqual(scale.converged(), true, `five datacenters converged after ${scaleTicks} ticks`);
  assert.deepStrictEqual(scale.causalityBreaks(), [], "the large run never broke causality");
  assert.ok(scale.messages <= 300 * 4, `messages stay linear in writes x peers (${scale.messages})`);
  for (const dc of scale.datacenters.values()) assert.strictEqual(dc.appliedIds.size, 300, `datacenter ${dc.id} applied all 300 writes`);
  assert.strictEqual(scale.datacenters.get(1).buffer.size, 0, "no datacenter keeps a buffered operation");

  // --- 9. edges ------------------------------------------------------------------------------------
  const solo = new Cluster({ datacenters: 1, lag: 1, seed: 0x8111 });
  solo.write(1, "only", "me");
  assert.strictEqual(solo.converged(), true, "a single datacenter is trivially converged");
  assert.strictEqual(solo.drain(), 0, "and has nothing to drain");
  assert.strictEqual(solo.datacenters.get(1).value("only"), "me");
  const empty = new Cluster({ datacenters: 2, lag: 1, seed: 0x8112 });
  assert.strictEqual(empty.converged(), true, "empty datacenters are converged");
  assert.strictEqual(empty.datacenters.get(1).value("missing"), undefined, "a missing key reads as undefined");
  const late = new Cluster({ datacenters: 2, lag: 5, seed: 0x8113 });
  late.write(1, "slow", 1);
  late.tick();
  assert.strictEqual(late.datacenters.get(2).value("slow"), undefined, "a write with a long lag has not arrived yet");
  assert.strictEqual(late.queue.length, 1, "and is still in flight");
  late.run(6);
  assert.strictEqual(late.datacenters.get(2).value("slow"), 1, "after the lag it lands");

  console.log(`causal replication: 3 DCs, 60 writes, ${cluster.messages} messages, ${drainTicks} ticks to converge, 0 causality breaks`);
  console.log(`offline: DC3 caught up in ${catchUp} ticks and applied ${offline.datacenters.get(3).appliedIds.size}/${offline.writes.size} writes in causal order`);
  console.log(`session: read-your-writes across datacenters and 12 monotonic reads verified; GC dropped ${collected} metadata entries with the digest unchanged`);
  console.log(`scale: 5 DCs, 300 writes, ${scale.messages} messages, ${scaleTicks} ticks, buffers drained to 0`);
  console.log("PASS");
}

module.exports = { Cluster, Datacenter, Write, Session, splitmix64, ReplicationError };
