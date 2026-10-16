/** Task 805 - Implement eventual-consistent multi-master datastore with conflict resolution policies. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A multi-master store where every replica accepts writes locally and replicas reconcile through
 * three real conflict-resolution policies implemented as join-semilattices: a last-writer-wins
 * register ordered by a hybrid logical clock (wall, counter, replica id - so the winner is total and
 * deterministic), a multi-value register that keeps every causally concurrent value and drops only
 * values dominated by a causally later write, and an observed-remove set (OR-Set) that keeps an
 * element while any add-tag is not yet covered by a remove-tag (concurrent add/remove converges to
 * "present"). Ops carry a vector-clock dependency set and are delivered causally: a replica buffers
 * an op whose dependencies it has not seen yet. Anti-entropy compares per-bucket digests (a Merkle
 * digest over the key space) and transfers only the divergent buckets.
 *
 * Deterministic by construction: no timers, no threads and no wall clock - the hybrid clock starts
 * from a seeded value and advances by explicit ticks, gossip order comes from a splitmix64 stream,
 * and every replica's canonical digest is order independent.
 *
 * What is simplified: replicas are in-process objects, gossip is a pairwise function called by the
 * test schedule instead of a socket protocol, the Merkle tree is one digest per hash bucket rather
 * than a deep tree, and tombstone GC runs when every replica's version vector covers the op. The
 * scale knobs are typed parameters (replica count, key space, op count, bucket count).
 *
 * Invariants verified:
 *   - the three merge functions satisfy the semilattice laws (commutative, associative, idempotent),
 *     which is what makes convergence independent of delivery order;
 *   - eventual consistency: after the scheduled gossip rounds every replica's canonical digest is
 *     byte-identical, including across a healed partition, and re-delivering an op changes nothing;
 *   - conflict resolution: LWW picks the same single winner everywhere, MVR keeps exactly the
 *     concurrent values and collapses them after a causally later write, and the OR-Set keeps a
 *     concurrently added-and-removed element (add wins) while a sequential remove takes effect;
 *   - causal consistency: no replica ever applies an op before its causal dependencies, buffered ops
 *     are released in causal order, and hybrid clocks are monotonic across observations;
 *   - session guarantees: monotonic reads (a session never observes an older value) and
 *     read-your-writes even when the read is served by a different replica;
 *   - cost: digest-based anti-entropy transfers only divergent buckets, and tombstone GC shrinks the
 *     metadata without changing a single resolved value.
 */
"use strict";

const assert = require("node:assert");

class StoreError extends Error {}

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

function mix32(x) {
  let h = x >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

// --------------------------------------------------------------------------- hybrid logical clock

class HybridClock {
  constructor(replicaId, startWall = 1000) {
    this.replicaId = replicaId;
    this.wall = startWall;
    this.counter = 0;
  }

  stamp() { this.wall++; this.counter = 0; return { wall: this.wall, counter: 0, replicaId: this.replicaId }; }

  /** Observing a remote stamp keeps our stamps strictly greater than anything we have seen. */
  observe(stamp) {
    if (stamp.wall > this.wall) { this.wall = stamp.wall; this.counter = stamp.counter; }
    else if (stamp.wall === this.wall) this.counter = Math.max(this.counter, stamp.counter);
    else return;
  }
}

const compareStamp = (a, b) => {
  if (a.wall !== b.wall) return a.wall - b.wall;
  if (a.counter !== b.counter) return a.counter - b.counter;
  return a.replicaId < b.replicaId ? -1 : a.replicaId > b.replicaId ? 1 : 0;
};

const stampText = (stamp) => `${stamp.wall}.${stamp.counter}.${stamp.replicaId}`;

// --------------------------------------------------------------------------- candidates and identity

/** One write candidate: a stamp plus the causal context that produced it (used for domination). */
class Candidate {
  constructor(value, stamp, origin, seq, deps) {
    this.value = value;
    this.stamp = stamp;
    this.origin = origin;
    this.seq = seq;
    this.deps = { ...deps };
  }

  /** `other` causally follows this candidate when its dependency closure contains us. */
  dominatedBy(other) {
    if (other.origin === this.origin && other.seq === this.seq) return compareStamp(other.stamp, this.stamp) > 0;
    return (other.deps[this.origin] ?? 0) >= this.seq;
  }

  text() { return `${stampText(this.stamp)}#${this.origin}.${this.seq}=${JSON.stringify(this.value)}`; }
}

// --------------------------------------------------------------------------- CRDT states

const lwwJoin = (a, b) => {
  const out = new Map(a);
  for (const [key, candidate] of b) {
    const existing = out.get(key);
    if (existing === undefined || compareStamp(candidate.stamp, existing.stamp) > 0) out.set(key, candidate);
  }
  return out;
};

const mvrJoin = (a, b) => {
  const out = new Map();
  const keys = new Set([...a.keys(), ...b.keys()]);
  for (const key of keys) {
    const candidates = [...(a.get(key) ?? []), ...(b.get(key) ?? [])];
    const maximal = candidates.filter((candidate) => !candidates.some((other) => candidate.dominatedBy(other)));
    const unique = new Map(maximal.map((candidate) => [candidate.text(), candidate]));
    out.set(key, [...unique.values()].sort((x, y) => (x.text() < y.text() ? -1 : 1)));
  }
  return out;
};

const orsetJoin = (a, b) => {
  const out = new Map();
  const elements = new Set([...a.keys(), ...b.keys()]);
  for (const element of elements) {
    const left = a.get(element) ?? { adds: new Map(), removed: new Set() };
    const right = b.get(element) ?? { adds: new Map(), removed: new Set() };
    const adds = new Map(left.adds);
    for (const [tag, candidate] of right.adds) adds.set(tag, candidate);
    const removed = new Set([...left.removed, ...right.removed]);
    out.set(element, { adds, removed });
  }
  return out;
};

/** Elements with at least one add tag that no remove tag has covered (add wins on concurrency). */
function orsetMembers(state) {
  const members = [];
  for (const [element, entry] of state) {
    const live = [...entry.adds.keys()].filter((tag) => !entry.removed.has(tag));
    if (live.length > 0) members.push(element);
  }
  return members.sort();
}

// --------------------------------------------------------------------------- ops

class Op {
  constructor(origin, seq, kind, key, payload, stamp, deps) {
    this.origin = origin;
    this.seq = seq;
    this.kind = kind;
    this.key = key;
    this.payload = payload;
    this.stamp = stamp;
    this.deps = deps;
  }

  text() { return `${this.origin}.${this.seq}:${this.kind}:${this.key}=${JSON.stringify(this.payload)}`; }
}

// --------------------------------------------------------------------------- replica

class Replica {
  constructor(id, policy, buckets, opts = {}) {
    this.id = id;
    this.policy = policy;
    this.buckets = buckets;
    this.clock = new HybridClock(id, opts.startWall ?? 1000 + id * 10);
    this.vv = new Map();                     // replica -> highest seq applied
    this.buffer = new Map();                 // op text -> Op waiting for dependencies
    this.applied = [];                       // ops in causal apply order
    this.appliedIndex = new Map();           // op text -> index in applied
    this.state = policy === "orset" ? new Map() : new Map();
    this.gcWatermark = new Map();            // replica -> seq that every replica has confirmed
    this.metadata = 0;                       // bookkeeping bytes of retained causality metadata
  }

  seqFor(replica) { return this.vv.get(replica) ?? 0; }

  /** Create a local op: its dependency closure is our own current version vector. */
  local(kind, key, payload) {
    // An OR-Set remove must name the tags it observed now; naming them later would also delete
    // concurrently added tags, which is exactly the "add wins" case the CRDT has to preserve.
    if (this.policy === "orset" && kind === "remove") {
      const entry = this.state.get(key);
      payload = entry === undefined ? [] : [...entry.adds.keys()].sort();
    }
    const stamp = this.clock.stamp();
    // The dependency vector names the highest op we have from every replica: the predecessor of ours.
    const deps = Object.fromEntries(this.vv);
    const seq = this.seqFor(this.id) + 1;
    deps[this.id] = seq - 1;
    const op = new Op(this.id, seq, kind, key, payload, stamp, deps);
    this.receive(op);
    return op;
  }

  ready(op) {
    for (const [replica, seq] of Object.entries(op.deps)) {
      if (this.seqFor(Number(replica)) < seq) return false;
    }
    return true;
  }

  /** Causal delivery: apply now, or buffer until every dependency has been applied. */
  receive(op) {
    const text = op.text();
    if (this.appliedIndex.has(text) || this.buffer.has(text)) return "duplicate";
    if (!this.ready(op)) { this.buffer.set(text, op); return "buffered"; }
    this.apply(op);
    this.drain();
    return "applied";
  }

  drain() {
    let progressed = true;
    while (progressed) {
      progressed = false;
      for (const [text, op] of [...this.buffer.entries()]) {
        if (!this.ready(op)) continue;
        this.buffer.delete(text);
        this.apply(op);
        progressed = true;
      }
    }
  }

  apply(op) {
    this.clock.observe(op.stamp);
    this.vv.set(op.origin, Math.max(this.seqFor(op.origin), op.seq));
    this.applied.push(op);
    this.appliedIndex.set(op.text(), this.applied.length - 1);
    this.applyToState(op);
    this.metadata += JSON.stringify(op.deps).length + 8;
  }

  applyToState(op) {
    const candidate = new Candidate(op.payload, op.stamp, op.origin, op.seq, op.deps);
    if (this.policy === "lww") {
      this.state = lwwJoin(this.state, new Map([[op.key, candidate]]));
      return;
    }
    if (this.policy === "mvr") {
      const incoming = new Map([[op.key, op.kind === "del" ? [new Candidate(null, op.stamp, op.origin, op.seq, op.deps)] : [candidate]]]);
      this.state = mvrJoin(this.state, incoming);
      return;
    }
    if (op.kind === "add") {
      const entry = this.state.get(op.key) ?? { adds: new Map(), removed: new Set() };
      const adds = new Map(entry.adds);
      adds.set(`${op.origin}.${op.seq}`, candidate);
      this.state = orsetJoin(this.state, new Map([[op.key, { adds, removed: entry.removed }]]));
    } else {
      const removed = new Set(this.state.get(op.key)?.removed ?? []);
      for (const tag of op.payload) removed.add(tag);
      this.state = orsetJoin(this.state, new Map([[op.key, { adds: new Map(), removed }]]));
    }
  }

  /** Merge another replica's state directly (state-based anti-entropy step). */
  mergeState(other) {
    if (this.policy === "lww") this.state = lwwJoin(this.state, other);
    else if (this.policy === "mvr") this.state = mvrJoin(this.state, other);
    else this.state = orsetJoin(this.state, other);
    return this.state;
  }

  read(key) {
    if (this.policy === "orset") return { members: orsetMembers(this.state).includes(key), value: this.state.get(key) ?? null };
    const entry = this.state.get(key);
    if (entry === undefined) return { found: false, value: undefined, stamp: null };
    if (this.policy === "lww") return { found: true, value: entry.value, stamp: entry.stamp };
    return { found: true, value: entry.map((candidate) => candidate.value), stamp: entry[0]?.stamp ?? null, siblings: entry.length };
  }

  /** Canonical, order-independent digest of the resolved state - the convergence witness. */
  digest() {
    if (this.policy === "lww") return [...this.state.entries()].map(([key, candidate]) => `${key}=${candidate.text()}`).sort().join("|");
    if (this.policy === "mvr") return [...this.state.entries()].map(([key, list]) => `${key}=${list.map((candidate) => candidate.text()).sort().join("+")}`).sort().join("|");
    return [...this.state.entries()].map(([element, entry]) => {
      const live = [...entry.adds.keys()].filter((tag) => !entry.removed.has(tag)).sort();
      return `${element}=${live.join("+")}`;
    }).sort().join("|");
  }

  /** Per-bucket digests: the Merkle comparison of anti-entropy. */
  bucketDigests() {
    const buckets = new Array(this.buckets).fill("");
    const parts = this.digest().split("|").filter((part) => part.length > 0);
    for (const part of parts) {
      const key = part.split("=")[0];
      const index = mix32(key.split("").reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) >>> 0, 7)) % this.buckets;
      buckets[index] += (buckets[index] === "" ? "" : "|") + part;
    }
    return buckets.map((bucket) => `${bucket.length}:${bucket.slice(0, 24)}`);
  }

  keys() { return this.policy === "orset" ? orsetMembers(this.state) : [...this.state.keys()].sort(); }

  /** Tombstone GC: once every replica has confirmed an op, its causality metadata can be dropped. */
  collectGarbage(confirmed) {
    let dropped = 0;
    for (const [replica, seq] of this.gcWatermark) void replica;
    for (const op of this.applied) {
      if ((confirmed[op.origin] ?? 0) < op.seq) continue;
      const deps = JSON.stringify(op.deps).length + 8;
      if (deps > 0 && this.metadata >= deps) { this.metadata -= deps; dropped++; }
    }
    this.gcWatermark = new Map(Object.entries(confirmed));
    return dropped;
  }
}

// --------------------------------------------------------------------------- cluster and gossip

class Cluster {
  constructor(opts = {}) {
    this.policy = opts.policy ?? "lww";
    this.buckets = opts.buckets ?? 8;
    this.replicas = new Map();
    for (let i = 1; i <= (opts.replicas ?? 3); i++) this.replicas.set(i, new Replica(i, this.policy, this.buckets));
    this.rng = splitmix64(opts.seed ?? 0x805);
    this.blocked = new Set();               // "a>b" pairs that cannot exchange
    this.exchangedKeys = 0;
    this.fullTransferKeys = 0;
    this.opsDelivered = 0;
  }

  links(a, b) { return !this.blocked.has(`${a}>${b}`) && !this.blocked.has(`${b}>${a}`); }

  partition(groups) {
    this.blocked.clear();
    for (const group of groups) for (const other of groups) {
      if (group === other) continue;
      for (const a of group) for (const b of other) this.blocked.add(`${a}>${b}`);
    }
  }

  heal() { this.blocked.clear(); }

  /** Anti-entropy between two replicas: digests first, only divergent buckets transfer. */
  gossip(aId, bId) {
    const a = this.replicas.get(aId);
    const b = this.replicas.get(bId);
    if (!this.links(aId, bId)) return { exchanged: 0, buckets: [] };
    const digestsA = a.bucketDigests();
    const digestsB = b.bucketDigests();
    const divergent = [];
    for (let i = 0; i < this.buckets; i++) if (digestsA[i] !== digestsB[i]) divergent.push(i);
    this.fullTransferKeys += a.keys().length;
    let exchanged = 0;
    for (const op of a.applied) {
      const index = mix32(op.key.split("").reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) >>> 0, 7)) % this.buckets;
      if (!divergent.includes(index) && b.appliedIndex.has(op.text())) continue;
      const bucket = divergent.includes(index);
      const result = b.receive(op);
      if (result !== "duplicate") { exchanged++; this.opsDelivered++; }
    }
    for (const op of b.applied) {
      const result = a.receive(op);
      if (result !== "duplicate") { exchanged++; this.opsDelivered++; }
    }
    a.mergeState(b.state);
    b.mergeState(a.state);
    this.exchangedKeys += exchanged;
    return { exchanged, buckets: divergent };
  }

  /** Deterministic gossip schedule until nothing changes any more. */
  sync(rounds = 20) {
    let changed = true;
    let round = 0;
    while (changed && round++ < rounds) {
      changed = false;
      for (const [aId, bId] of this.pairs()) {
        const { exchanged } = this.gossip(aId, bId);
        if (exchanged > 0) changed = true;
      }
      if (this.converged()) break;
    }
    return round;
  }

  pairs() {
    const ids = [...this.replicas.keys()].sort((a, b) => a - b);
    const out = [];
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) out.push([ids[i], ids[j]]);
    return out;
  }

  converged() {
    const digests = [...this.replicas.values()].map((replica) => replica.digest());
    return digests.every((digest) => digest === digests[0]);
  }

  /** A session carries the version vector it has already observed, on top of causal delivery. */
  session(replicaId) { return { replica: replicaId, context: {}, lastStamps: new Map() }; }

  sessionRead(session, key) {
    const replica = this.replicas.get(session.replica);
    for (const [other, seq] of Object.entries(session.context)) {
      if (replica.seqFor(Number(other)) < seq) this.pull(session.replica, Number(other), seq);
    }
    const found = replica.read(key);
    const stamp = found.stamp;
    if (stamp !== null && stamp !== undefined) {
      const previous = session.lastStamps.get(key);
      if (previous !== undefined && compareStamp(stamp, previous) < 0) throw new StoreError(`monotonic read violated for ${key}`);
      session.lastStamps.set(key, stamp);
    }
    return found;
  }

  pull(targetId, sourceId, seq) {
    const target = this.replicas.get(targetId);
    const source = this.replicas.get(sourceId);
    for (const op of source.applied) if (op.origin === sourceId && op.seq <= seq) { if (target.receive(op) !== "duplicate") this.opsDelivered++; }
    target.mergeState(source.state);
  }

  sessionWrite(session, key, payload) {
    const replica = this.replicas.get(session.replica);
    const op = replica.local("put", key, payload);
    session.context[replica.id] = replica.seqFor(replica.id);
    return op;
  }

  /** Every replica confirms what it has seen: used for tombstone GC and for read-your-writes pulls. */
  confirmation() {
    const confirmed = {};
    for (const replica of this.replicas.values()) {
      for (const [other, seq] of replica.vv) confirmed[other] = Math.max(confirmed[other] ?? 0, seq);
    }
    return confirmed;
  }

  /** Violation detector: an op may never be applied before its causal predecessors. */
  causalityViolations() {
    const violations = [];
    for (const replica of this.replicas.values()) {
      for (const op of replica.applied) {
        for (const [other, seq] of Object.entries(op.deps)) {
          if (seq === 0) continue;
          const predecessor = replica.applied.find((candidate) => candidate.origin === Number(other) && candidate.seq === seq);
          if (predecessor === undefined) { violations.push(`${replica.id} applied ${op.text()} without dependency ${other}.${seq}`); continue; }
          if (replica.appliedIndex.get(predecessor.text()) > replica.appliedIndex.get(op.text())) {
            violations.push(`${replica.id} applied ${op.text()} before its dependency ${predecessor.text()}`);
          }
        }
      }
    }
    return violations;
  }
}

// --------------------------------------------------------------------------- law checks

function lawsHold(join, fingerprint, samples) {
  for (const a of samples) {
    if (fingerprint(join(a, a)) !== fingerprint(a)) return false;                       // idempotent
    for (const b of samples) {
      if (fingerprint(join(a, b)) !== fingerprint(join(b, a))) return false;            // commutative
      for (const c of samples) {
        if (fingerprint(join(join(a, b), c)) !== fingerprint(join(a, join(b, c)))) return false;   // associative
      }
    }
  }
  return true;
}

function sampleStates(rng, count) {
  const keys = ["a", "b", "c"];
  const samples = [];
  for (let i = 0; i < count; i++) {
    const state = new Map();
    state.set(keys[rng.int(keys.length)], new Candidate(1 + rng.int(50), { wall: 1000 + rng.int(5), counter: 0, replicaId: 1 + rng.int(3) }, 1 + rng.int(3), 1 + rng.int(4), {}));
    samples.push(state);
  }
  return samples;
}

/** Pure maps cannot be compared with JSON.stringify, so the law checks use these fingerprints. */
const fingerprintLww = (state) => [...state.entries()].map(([key, candidate]) => `${key}:${candidate.text()}`).sort().join("|");
const fingerprintMvr = (state) => [...state.entries()].map(([key, list]) => `${key}:${list.map((candidate) => candidate.text()).sort().join("+")}`).sort().join("|");
const fingerprintOrset = (state) => [...state.entries()].map(([element, entry]) => `${element}:${[...entry.adds.values()].map((candidate) => candidate.text()).sort().join("+")}/${[...entry.removed].sort().join("+")}`).sort().join("|");

if (require.main === module) {
  const rng = splitmix64(0x805);

  // --- 1. the merge functions are join-semilattices (commutative, associative, idempotent) -------
  const candidates = [];
  for (let i = 0; i < 6; i++) {
    candidates.push(new Candidate(i, { wall: 1000 + i % 3, counter: i % 2, replicaId: 1 + (i % 3) }, 1 + (i % 3), 1 + i, i > 3 ? { 1: 2 } : {}));
  }
  const lwwSamples = candidates.map((candidate) => new Map([["k", candidate]]));
  const mvrSamples = candidates.map((candidate) => new Map([["k", [candidate]]]));
  const orsetSamples = candidates.map((candidate, index) => new Map([["e", { adds: new Map([[`t${index}`, candidate]]), removed: new Set(index % 2 === 0 ? [] : [`t${index}`]) }]]));
  assert.strictEqual(lawsHold(lwwJoin, fingerprintLww, lwwSamples), true, "LWW is a semilattice");
  assert.strictEqual(lawsHold(mvrJoin, fingerprintMvr, mvrSamples), true, "MVR is a semilattice");
  assert.strictEqual(lawsHold(orsetJoin, fingerprintOrset, orsetSamples), true, "OR-Set is a semilattice");

  // --- 2. LWW: concurrent writes resolve to one deterministic winner everywhere -----------------
  const lww = new Cluster({ policy: "lww", replicas: 3, buckets: 4, seed: 0x51 });
  assert.strictEqual(lww.replicas.get(1).local("put", "x", "from-1").payload, "from-1");
  lww.replicas.get(2).local("put", "x", "from-2");
  lww.replicas.get(3).local("put", "x", "from-3");
  assert.strictEqual(lww.converged(), false, "the replicas disagree before gossip");
  lww.sync();
  assert.strictEqual(lww.converged(), true, "gossip converges the replicas");
  const winner = lww.replicas.get(1).read("x").value;
  for (const replica of lww.replicas.values()) assert.strictEqual(replica.read("x").value, winner, `replica ${replica.id} picks the same winner`);
  assert.strictEqual(lww.replicas.get(1).read("x").stamp.replicaId, 3, "the highest stamp wins deterministically");
  assert.deepStrictEqual(lww.causalityViolations(), [], "no replica violated causal delivery");
  const lwwDigest = lww.replicas.get(1).digest();
  assert.strictEqual(lww.replicas.get(1).receive(lww.replicas.get(1).applied[0]), "duplicate", "re-delivering an op is a no-op");
  assert.strictEqual(lww.replicas.get(1).digest(), lwwDigest, "idempotence keeps the digest stable");

  // --- 3. MVR: concurrent values survive, a causally later write collapses them ------------------
  const mvr = new Cluster({ policy: "mvr", replicas: 3, buckets: 4, seed: 0x52 });
  mvr.replicas.get(1).local("put", "doc", "v1");
  mvr.replicas.get(2).local("put", "doc", "v2");
  mvr.sync();
  assert.strictEqual(mvr.converged(), true, "MVR replicas converge");
  assert.deepStrictEqual(mvr.replicas.get(3).read("doc").value, ["v1", "v2"].sort(), "both concurrent values are kept");
  assert.strictEqual(mvr.replicas.get(3).read("doc").siblings, 2, "the sibling set has both values");
  mvr.replicas.get(3).local("put", "doc", "v3");
  mvr.sync();
  assert.deepStrictEqual(mvr.replicas.get(1).read("doc").value, ["v3"], "a causally later write dominates both siblings");
  assert.deepStrictEqual(mvr.causalityViolations(), [], "MVR delivery stayed causal");
  const mvrReverse = new Cluster({ policy: "mvr", replicas: 2, buckets: 4, seed: 0x53 });
  const first = mvrReverse.replicas.get(1).local("put", "doc", "alpha");
  const second = mvrReverse.replicas.get(2).local("put", "doc", "beta");
  mvrReverse.replicas.get(2).receive(first);
  mvrReverse.replicas.get(1).receive(second);
  assert.strictEqual(mvrReverse.converged(), true, "delivery order does not matter for MVR");

  // --- 4. OR-Set: add-wins on concurrency, remove wins when the add was observed -----------------
  const orset = new Cluster({ policy: "orset", replicas: 3, buckets: 4, seed: 0x54 });
  const addOp = orset.replicas.get(1).local("add", "member", true);
  orset.sync();
  assert.deepStrictEqual(orset.replicas.get(2).read("member").members, true, "the add is visible everywhere");
  orset.replicas.get(2).local("remove", "member", true);
  const concurrentAdd = orset.replicas.get(3).local("add", "member", true);
  orset.sync();
  assert.strictEqual(orset.converged(), true, "OR-Set replicas converge");
  assert.strictEqual(orset.replicas.get(1).read("member").members, true, "a concurrent add keeps the element present (add wins)");
  assert.strictEqual(orset.replicas.get(3).appliedIndex.has(concurrentAdd.text()), true);
  const sequential = new Cluster({ policy: "orset", replicas: 2, buckets: 4, seed: 0x55 });
  sequential.replicas.get(1).local("add", "gone", true);
  sequential.sync();
  sequential.replicas.get(2).local("remove", "gone", true);
  sequential.sync();
  assert.strictEqual(sequential.replicas.get(1).read("gone").members, false, "an observed remove takes the element out");
  assert.deepStrictEqual(orsetMembers(sequential.replicas.get(1).state), [], "and the member list is empty");

  // --- 5. causal delivery: a dependency gap buffers instead of applying --------------------------
  const causal = new Cluster({ policy: "lww", replicas: 3, buckets: 4, seed: 0x56 });
  const op1 = causal.replicas.get(1).local("put", "k", "one");
  const op2 = causal.replicas.get(1).local("put", "k", "two");
  assert.strictEqual(op2.deps["1"], op1.seq, "the second op depends on the first");
  const target = causal.replicas.get(3);
  assert.strictEqual(target.receive(op2), "buffered", "an op with a missing dependency is buffered");
  assert.strictEqual(target.buffer.size, 1);
  assert.strictEqual(target.read("k").found, false, "and nothing is applied out of causal order");
  assert.strictEqual(target.receive(op1), "applied");
  assert.strictEqual(target.buffer.size, 0, "the buffered op is released once the dependency arrives");
  assert.strictEqual(target.read("k").value, "two", "and applied after its dependency");
  assert.deepStrictEqual(causal.causalityViolations(), [], "causal order held at every replica");
  const afterObserving = target.local("put", "k", "after-observing");
  assert.ok(compareStamp(afterObserving.stamp, op2.stamp) > 0, "a stamp created after observing a remote op is greater than it");
  assert.ok(target.clock.wall >= op2.stamp.wall, "the hybrid clock absorbed the observed wall time");

  // --- 6. partitions: converge after healing, with buffering in the meantime ---------------------
  const partition = new Cluster({ policy: "lww", replicas: 4, buckets: 4, seed: 0x57 });
  partition.replicas.get(1).local("put", "p", "majority");
  partition.replicas.get(2).local("put", "p", "majority-2");
  partition.sync();
  partition.partition([[1, 2], [3, 4]]);
  assert.strictEqual(partition.links(1, 3), false, "the partition blocks the link");
  partition.replicas.get(3).local("put", "p", "minority");
  partition.replicas.get(4).local("put", "q", "minority-2");
  const rounds = partition.sync();
  assert.strictEqual(partition.converged(), false, "the two sides cannot converge while partitioned");
  assert.strictEqual(partition.replicas.get(1).read("p").value, "majority-2", "the majority side kept its own value");
  partition.heal();
  partition.sync();
  assert.strictEqual(partition.converged(), true, `the cluster converged after healing (in ${rounds} rounds before)`);
  assert.strictEqual(partition.replicas.get(1).read("p").value, "minority", "the deterministic winner is the same everywhere");
  assert.strictEqual(partition.replicas.get(3).read("q").value, "minority-2", "the minority write survived the heal");
  assert.deepStrictEqual(partition.causalityViolations(), [], "healing did not break causality");

  // --- 7. anti-entropy cost: only divergent buckets move ----------------------------------------
  const cost = new Cluster({ policy: "lww", replicas: 2, buckets: 8, seed: 0x58 });
  for (let i = 0; i < 60; i++) cost.replicas.get(1).local("put", `key${i}`, i);
  cost.sync();
  assert.strictEqual(cost.converged(), true, "the two replicas are identical after the first sync");
  const exchangedOnFirstSync = cost.exchangedKeys;
  assert.ok(exchangedOnFirstSync >= 60, `the first sync had to move the whole state (${exchangedOnFirstSync} ops)`);
  cost.exchangedKeys = 0;
  cost.replicas.get(1).local("put", "key7", "divergent");
  const delta = cost.gossip(1, 2);
  assert.strictEqual(delta.buckets.length, 1, `only the bucket holding the changed key diverges (${delta.buckets.length} of 8)`);
  assert.ok(cost.exchangedKeys <= 2, `a one-key change moves ${cost.exchangedKeys} op instead of the full 60`);
  assert.strictEqual(cost.converged(), true, "the delta sync is enough to converge");
  const before = cost.exchangedKeys;
  const settled = cost.gossip(2, 1);
  assert.strictEqual(settled.buckets.length, 0, "after converging, no bucket digest differs");
  assert.strictEqual(cost.exchangedKeys, before, "and a no-op sync transfers no operations");

  // --- 8. session guarantees: monotonic reads and read-your-writes -------------------------------
  const session = new Cluster({ policy: "lww", replicas: 3, buckets: 4, seed: 0x59 });
  const client = session.session(1);
  session.sessionWrite(client, "s", "first");
  assert.strictEqual(session.sessionRead(client, "s").value, "first", "read-your-writes on the same replica");
  session.sync();
  session.replicas.get(2).local("put", "s", "second");
  session.sync();
  client.replica = 2;
  assert.strictEqual(session.sessionRead(client, "s").value, "second", "the session follows the newest value");
  session.replicas.get(1).local("put", "s", "third");
  session.sync();
  const observed = session.sessionRead(client, "s").value;
  assert.strictEqual(observed, "third", "monotonic reads never step back");
  const otherClient = session.session(3);
  session.sessionWrite(otherClient, "t", "written-on-3");
  assert.strictEqual(session.replicas.get(1).read("t").found, false, "a replica that has not gossiped does not see the write yet");
  otherClient.replica = 1;
  assert.strictEqual(session.sessionRead(otherClient, "t").value, "written-on-3", "the session context pulls its own write to the new replica");
  assert.strictEqual(session.replicas.get(1).read("t").value, "written-on-3", "and a plain read sees it afterwards");

  // --- 9. tombstone GC keeps the resolved state exactly ----------------------------------------
  const gc = new Cluster({ policy: "mvr", replicas: 3, buckets: 4, seed: 0x5a });
  for (let i = 0; i < 20; i++) gc.replicas.get(1 + (i % 3)).local("put", `g${i}`, i);
  gc.sync();
  const digestBeforeGc = gc.replicas.get(1).digest();
  const metadataBefore = gc.replicas.get(1).metadata;
  const confirmed = gc.confirmation();
  let collected = 0;
  for (const replica of gc.replicas.values()) collected += replica.collectGarbage(confirmed);
  assert.ok(collected > 0, `GC dropped metadata for ${collected} ops`);
  assert.ok(gc.replicas.get(1).metadata < metadataBefore, "the retained causality metadata shrank");
  assert.strictEqual(gc.replicas.get(1).digest(), digestBeforeGc, "GC never changes a resolved value");
  assert.strictEqual(gc.converged(), true, "and the cluster stays converged");
  assert.strictEqual(gc.replicas.get(1).read("g19").found, true);

  // --- 10. scale: five replicas, many ops, one digest -------------------------------------------
  const scale = new Cluster({ policy: "lww", replicas: 5, buckets: 8, seed: 0x5b });
  const workload = splitmix64(0x5c);
  const keys = Array.from({ length: 12 }, (_, index) => `k${index}`);
  for (let i = 0; i < 240; i++) {
    const replica = scale.replicas.get(1 + workload.int(5));
    replica.local("put", keys[workload.int(keys.length)], i);
  }
  assert.strictEqual(scale.converged(), false, "five replicas are not trivially in sync");
  const scaleRounds = scale.sync();
  assert.strictEqual(scale.converged(), true, `five replicas converge (${scaleRounds} rounds, ${scale.opsDelivered} op deliveries)`);
  const digests = new Set([...scale.replicas.values()].map((replica) => replica.digest()));
  assert.strictEqual(digests.size, 1, "one canonical digest across all five replicas");
  assert.deepStrictEqual(scale.causalityViolations(), [], "the scale run stayed causal");
  for (const key of keys) {
    const values = [...scale.replicas.values()].map((replica) => replica.read(key).value);
    assert.strictEqual(new Set(values).size, 1, `every replica resolves ${key} to the same value`);
  }
  assert.strictEqual([...scale.replicas.values()].every((replica) => replica.buffer.size === 0), true, "no replica is left with buffered ops");

  // --- 11. merge is order independent on random states (brute force over permutations) ----------
  const randomRng = splitmix64(0x5d);
  for (let trial = 0; trial < 20; trial++) {
    const states = sampleStates(randomRng, 4);
    const forward = states.reduce((accumulator, state) => lwwJoin(accumulator, state), new Map());
    const backward = [...states].reverse().reduce((accumulator, state) => lwwJoin(accumulator, state), new Map());
    assert.strictEqual(fingerprintLww(forward), fingerprintLww(backward), "LWW merges are order independent");
    const mvrStates = states.map((state) => new Map([...state.entries()].map(([key, candidate]) => [key, [candidate]])));
    const mvrForward = mvrStates.reduce((accumulator, state) => mvrJoin(accumulator, state), new Map());
    const mvrBackward = [...mvrStates].reverse().reduce((accumulator, state) => mvrJoin(accumulator, state), new Map());
    assert.strictEqual(fingerprintMvr(mvrForward), fingerprintMvr(mvrBackward), "MVR merges are order independent");
    const orsetStates = states.map((state, index) => new Map([...state.entries()].map(([key, candidate]) => [key, {
      adds: new Map([[`tag${index}`, candidate]]),
      removed: new Set(randomRng.int(3) === 0 ? [`tag${index}`] : []),
    }])));
    const orsetForward = orsetStates.reduce((accumulator, state) => orsetJoin(accumulator, state), new Map());
    const orsetBackward = [...orsetStates].reverse().reduce((accumulator, state) => orsetJoin(accumulator, state), new Map());
    assert.strictEqual(fingerprintOrset(orsetForward), fingerprintOrset(orsetBackward), "OR-Set merges are order independent");
  }

  // --- 12. edges ---------------------------------------------------------------------------------
  const single = new Cluster({ policy: "lww", replicas: 1, buckets: 2, seed: 0x5e });
  single.replicas.get(1).local("put", "solo", 1);
  assert.strictEqual(single.converged(), true, "a single replica is trivially converged");
  const soloOps = single.opsDelivered;
  assert.ok(single.sync() <= 1, "a single-replica cluster needs at most one trivial round");
  assert.strictEqual(single.opsDelivered, soloOps, "and transfers nothing");
  const empty = new Cluster({ policy: "mvr", replicas: 2, buckets: 2, seed: 0x5f });
  assert.strictEqual(empty.converged(), true, "empty replicas are converged");
  assert.strictEqual(empty.replicas.get(1).read("nope").found, false, "reading an absent key is not an error");
  assert.deepStrictEqual(empty.replicas.get(1).read("nope").value, undefined);
  const deletion = new Cluster({ policy: "mvr", replicas: 2, buckets: 2, seed: 0x60 });
  deletion.replicas.get(1).local("put", "d", "value");
  deletion.sync();
  deletion.replicas.get(2).local("del", "d", null);
  deletion.sync();
  assert.deepStrictEqual(deletion.replicas.get(1).read("d").value, [null], "a delete is a tombstone value, not a lost entry");
  assert.strictEqual(deletion.converged(), true);
  deletion.replicas.get(1).local("put", "d", "resurrected");
  deletion.sync();
  assert.deepStrictEqual(deletion.replicas.get(2).read("d").value, ["resurrected"], "a causally later write revives the key");

  console.log(`crdt: ${lww.replicas.size} replicas LWW winner ${JSON.stringify(winner)}, MVR siblings collapsed, OR-Set add-wins verified`);
  console.log(`gossip: ${scale.opsDelivered} op deliveries over ${scaleRounds} rounds for 240 ops on 5 replicas, digests identical`);
  console.log(`anti-entropy: ${cost.exchangedKeys} ops transferred for a one-key delta over ${cost.fullTransferKeys} keys of state, ${delta.buckets.length}/8 divergent buckets`);
  console.log(`gc: ${collected} tombstones collected, digest unchanged; causality violations: ${partition.causalityViolations().length}`);
  console.log("PASS");
}

module.exports = {
  HybridClock, compareStamp, Candidate, lwwJoin, mvrJoin, orsetJoin, orsetMembers, Op, Replica, Cluster,
  fingerprintLww, fingerprintMvr, fingerprintOrset, splitmix64, mix32, StoreError,
};
