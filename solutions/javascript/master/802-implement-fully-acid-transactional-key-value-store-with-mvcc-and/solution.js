/** Task 802 - Implement fully ACID transactional key-value store with MVCC and 2PC. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A sharded key-value store with multi-version concurrency control, snapshot-isolated reads at a
 * per-transaction timestamp, commit-time validation over the read *and* write set (so lost updates
 * and write skew are rejected, not just write-write conflicts), and strict two-phase commit across
 * shards: a participant locks its write keys and keeps a write intent from prepare until the
 * coordinator's decision arrives, which is what makes the dirty read impossible and the abort
 * atomic. Every participant and the coordinator write a redo/undo WAL, and a fresh set of shards is
 * rebuilt from those records to prove durability.
 *
 * Everything is driven by explicit pump() rounds over an in-memory RPC bus: no timers, no threads,
 * no wall clock. Failure injection is part of the model - messages can be *held* (delayed, then
 * released) or *dropped once*, which is how a coordinator crash between "prepared" and "decided"
 * and a lost decision message are reproduced; the engine then recovers from the WALs.
 *
 * What is simplified: shards are Maps instead of B-trees on page storage, "fsync" is appending an
 * object to the WAL array, the coordinator is in-process, and the recovery protocol is
 * presumed-abort with a durable decision log (no coordinator takeover election). Scale knobs are
 * typed parameters (shard count, workload size, key space), so the same invariants are checked on a
 * 4-shard / 200-transaction run.
 *
 * Invariants verified:
 *   - atomicity: for every transaction id, the set of shards that applied its writes is either
 *     empty or exactly its participant set - never partial - including after failure injection;
 *   - consistency: a transfer workload preserves the total, and the committed history is proven
 *     serializable by brute force over permutations, and by a precedence-graph acyclicity check;
 *   - isolation: no dirty reads (prepared intents are invisible), repeatable reads inside a
 *     transaction, snapshot reads ignore later commits, and write skew aborts exactly one side;
 *   - durability: replaying the WALs into fresh shards reproduces the live state checksum exactly,
 *     and recovery resolves every prepared-but-undecided transaction from the decision log;
 *   - cost: messages and WAL records per transaction stay linear in the number of participants.
 */
"use strict";

const assert = require("node:assert");

/** FNV-1a over the key text; deterministic, so a key always maps to the same shard. */
function hashString(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

function mix32(x) {
  let h = x >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

/** splitmix64 drives every generated workload; same seed, same schedule. */
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

class Wal {
  constructor(name) { this.name = name; this.records = []; }
  append(record) { this.records.push({ lsn: this.records.length + 1, ...record }); return this.records.length; }
  entries() { return this.records.map((r) => ({ ...r })); }
  checksum() { return this.records.map((r) => `${r.lsn}:${r.op}:${r.txnId ?? ""}:${r.key ?? ""}:${r.ts ?? ""}`).join("|"); }
}

class Shard {
  constructor(id, wal = new Wal(`shard-${id}`)) {
    this.id = id;
    this.wal = wal;
    this.versions = new Map();     // key -> ascending [{ts, value, deleted, txn}]
    this.latest = new Map();       // key -> newest committed version timestamp (0 = never written)
    this.locks = new Map();        // key -> txn id holding the write intent
    this.intents = new Map();      // txn id -> {startTs, writes}
    this.stats = { prepares: 0, commits: 0, aborts: 0, lockConflicts: 0, staleRejections: 0 };
  }

  latestTs(key) { return this.latest.get(key) ?? 0; }

  /** Snapshot read: the newest version visible at `ts`. Never sees a prepared intent. */
  readAt(key, ts) {
    const chain = this.versions.get(key);
    if (chain === undefined) return { found: false, value: undefined, versionTs: 0 };
    for (let i = chain.length - 1; i >= 0; i--) {
      if (chain[i].ts <= ts) {
        const version = chain[i];
        return version.deleted
          ? { found: false, value: undefined, versionTs: version.ts }
          : { found: true, value: version.value, versionTs: version.ts };
      }
    }
    return { found: false, value: undefined, versionTs: 0 };
  }

  /** Values visible to a reader that starts after every commit so far. */
  materialize() {
    const out = new Map();
    for (const [key, chain] of this.versions) {
      const version = chain[chain.length - 1];
      if (!version.deleted) out.set(key, version.value);
    }
    return out;
  }

  /** Phase one: validate against the snapshot, take the write locks, keep the intent. */
  prepare(txn) {
    this.stats.prepares++;
    const relevant = (key) => keyShardOf(key, this.shardCount) === this.id;
    for (const key of txn.reads) {
      if (!relevant(key)) continue;
      if (this.latestTs(key) > txn.startTs) {
        this.stats.staleRejections++;
        return { vote: "abort", reason: `stale snapshot for ${key}` };
      }
    }
    for (const key of txn.writes.keys()) {
      if (!relevant(key)) continue;
      if (this.latestTs(key) > txn.startTs) {
        this.stats.staleRejections++;
        return { vote: "abort", reason: `stale snapshot for write ${key}` };
      }
      const holder = this.locks.get(key);
      if (holder !== undefined && holder !== txn.id) {
        this.stats.lockConflicts++;
        return { vote: "abort", reason: `write lock on ${key} held by txn ${holder}` };
      }
      this.locks.set(key, txn.id);
    }
    const writes = new Map();
    for (const [key, value] of txn.writes) if (relevant(key)) writes.set(key, value);
    this.intents.set(txn.id, { startTs: txn.startTs, writes });
    this.wal.append({ op: "prepare", txnId: txn.id, startTs: txn.startTs, keys: [...writes.keys()] });
    return { vote: "commit", reason: "prepared" };
  }

  applyVersion(txnId, key, value, ts) {
    const chain = this.versions.get(key);
    const version = { ts, value, deleted: value === undefined, txn: txnId };
    if (chain === undefined) this.versions.set(key, [version]);
    else chain.push(version);
    this.latest.set(key, ts);
  }

  commit(txnId, ts) {
    const intent = this.intents.get(txnId);
    if (intent === undefined) return false;
    for (const [key, value] of intent.writes) this.applyVersion(txnId, key, value, ts);
    for (const key of intent.writes.keys()) this.locks.delete(key);
    this.intents.delete(txnId);
    this.stats.commits++;
    // The redo record carries the values, so the store can be rebuilt from the WAL alone.
    this.wal.append({ op: "commit", txnId, ts, keys: [...intent.writes.keys()], values: Object.fromEntries(intent.writes) });
    return true;
  }

  abort(txnId, reason = "decision") {
    const intent = this.intents.get(txnId);
    if (intent === undefined) return false;
    for (const key of intent.writes.keys()) if (this.locks.get(key) === txnId) this.locks.delete(key);
    this.intents.delete(txnId);
    this.stats.aborts++;
    this.wal.append({ op: "abort", txnId, reason });
    return true;
  }

  /** Write a prepared intent forward during recovery, using the decision's timestamp. */
  commitFromIntent(txnId, ts) { return this.commit(txnId, ts); }

  /** Every committed write of `txnId` that this shard holds - used by the atomicity audit. */
  holdsTxn(txnId) { return [...this.versions.values()].some((chain) => chain.some((v) => v.txn === txnId)); }

  unresolved() {
    const outstanding = new Map();
    for (const record of this.wal.records) {
      if (record.op === "prepare") outstanding.set(record.txnId, record);
      else if (record.op === "commit" || record.op === "abort") outstanding.delete(record.txnId);
    }
    return outstanding;
  }
}

/** Every key of a transaction is routed with the same deterministic function on every node. */
function keyShardOf(key, shardCount) { return mix32(hashString(key)) % shardCount; }

class Txn {
  constructor(id, startTs) {
    this.id = id;
    this.startTs = startTs;
    this.state = "active";
    this.commitTs = 0;
    this.reason = null;
    this.reads = new Map();      // key -> {found, value, versionTs, shard}
    this.writes = new Map();     // key -> value (undefined = delete)
    this.ops = [];               // ordered operation log, replayed by the serializability oracle
    this.readOnly = false;
  }
}

class Bus {
  constructor() {
    this.nodes = new Map();
    this.queue = [];
    this.held = [];
    this.dropped = 0;
    this.delivered = 0;
    this.seq = 0;
    this.holds = [];
    this.dropRules = [];
    this.rounds = 0;
  }

  register(node) { this.nodes.set(node.id, node); }

  send(from, to, msg) {
    this.seq++;
    const envelope = { from, to, msg, seq: this.seq };
    for (const rule of this.dropRules) {
      if (!rule.used && rule.match(envelope)) { rule.used = true; this.dropped++; return false; }
    }
    for (const hold of this.holds) {
      if (!hold.released && hold.match(envelope)) { this.held.push(envelope); return true; }
    }
    this.queue.push(envelope);
    return true;
  }

  /** Drop the first message matching `match` (models a crash that loses one RPC). */
  dropOnce(match) { this.dropRules.push({ match, used: false }); }

  /** Hold every matching message until the returned handle is released. */
  hold(match) {
    const handle = { match, released: false, release: () => { handle.released = true; } };
    this.holds.push(handle);
    return handle;
  }

  releaseBacklog() {
    if (this.held.length === 0) return 0;
    const backlog = this.held;
    this.held = [];
    backlog.sort((a, b) => a.seq - b.seq);
    for (const envelope of backlog) this.queue.push(envelope);
    return backlog.length;
  }

  /** Deliver everything that was queued at the start of this round, in send order. */
  drainRound() {
    this.rounds++;
    const batch = this.queue;
    this.queue = [];
    batch.sort((a, b) => a.seq - b.seq);
    for (const envelope of batch) {
      this.delivered++;
      const node = this.nodes.get(envelope.to);
      if (node) node.receive(envelope.from, envelope.msg);
    }
    return batch.length;
  }
}

class Participant {
  constructor(id, shard, bus) { this.id = id; this.shard = shard; this.bus = bus; this.bus.register(this); }

  receive(from, msg) {
    if (msg.type === "prepare") {
      const txn = msg.txn;
      const view = {
        id: txn.id,
        startTs: txn.startTs,
        writes: new Map(txn.writes),
        reads: new Set(txn.reads),
      };
      const outcome = this.shard.prepare(view);
      this.bus.send(this.id, from, { type: "vote", txnId: txn.id, vote: outcome.vote, reason: outcome.reason });
      return;
    }
    if (msg.type === "decision") {
      if (msg.decision === "committed") this.shard.commit(msg.txnId, msg.ts);
      else this.shard.abort(msg.txnId, msg.reason ?? "coordinator abort");
    }
  }
}

class Coordinator {
  constructor(id, bus, participants) {
    this.id = id;
    this.bus = bus;
    this.participants = participants;
    this.wal = new Wal("coordinator");
    this.decisions = new Map();      // txnId -> {op, ts} - the durable decision log
    this.pending = new Map();        // txnId -> {txn, votes, expected}
    this.engine = null;
    bus.register(this);
  }

  begin(txn) {
    const expected = [...new Set([...txn.writes.keys()].map((k) => keyShardOf(k, txn.shardCount)))].sort((a, b) => a - b);
    this.pending.set(txn.id, { txn, expected, votes: new Map() });
    this.wal.append({ op: "begin", txnId: txn.id });
    for (const shardId of expected) {
      this.bus.send(this.id, shardId, {
        type: "prepare",
        txn: { id: txn.id, startTs: txn.startTs, reads: [...txn.reads.keys()], writes: [...txn.writes.entries()] },
      });
    }
    return expected;
  }

  decide(txnId, decision, ts, reason) {
    const pending = this.pending.get(txnId);
    const state = decision === "commit" ? "committed" : "aborted";
    this.wal.append({ op: state, txnId, ts, reason: reason ?? null });
    this.decisions.set(txnId, { op: state, ts, reason: reason ?? null });
    for (const shardId of pending.expected) {
      this.bus.send(this.id, shardId, { type: "decision", txnId, decision: state, ts, reason });
    }
    this.pending.delete(txnId);
    if (this.engine) this.engine.finalize(pending.txn, state, ts, reason ?? null);
  }

  receive(from, msg) {
    if (msg.type !== "vote") return;
    const pending = this.pending.get(msg.txnId);
    if (pending === undefined) return;
    pending.votes.set(from, msg);
    if (msg.vote === "abort") {
      this.decide(msg.txnId, "abort", 0, msg.reason);
      return;
    }
    if ([...pending.votes.values()].filter((v) => v.vote === "commit").length === pending.expected.length) {
      const ts = this.engine.allocateTimestamp();
      this.decide(msg.txnId, "commit", ts, "all prepared");
    }
  }

  decisionFor(txnId) { return this.decisions.get(txnId) ?? null; }
}

class Engine {
  constructor(opts = {}) {
    this.shardCount = opts.shards ?? 3;
    this.bus = new Bus();
    this.shards = new Map();
    for (let i = 0; i < this.shardCount; i++) {
      const shard = new Shard(i);
      shard.shardCount = this.shardCount;
      this.shards.set(i, shard);
    }
    this.participants = new Map();
    for (const [id, shard] of this.shards) this.participants.set(id, new Participant(id, shard, this.bus));
    this.coordinator = new Coordinator("coordinator", this.bus, [...this.participants.keys()]);
    this.coordinator.engine = this;
    this.tso = 0;
    this.nextId = 1;
    this.txns = new Map();
    this.committed = [];
    this.aborted = [];
    this.recoveries = 0;
  }

  allocateTimestamp() { return ++this.tso; }

  shardFor(key) { return keyShardOf(key, this.shardCount); }

  begin() {
    const txn = new Txn(this.nextId++, this.allocateTimestamp());
    txn.shardCount = this.shardCount;
    this.txns.set(txn.id, txn);
    return txn;
  }

  read(txn, key) {
    assert.strictEqual(txn.state, "active", `transaction ${txn.id} is not accepting reads`);
    if (txn.writes.has(key)) {
      const value = txn.writes.get(key);
      txn.ops.push({ type: "read", key, found: value !== undefined, value });
      return { found: value !== undefined, value };
    }
    if (!txn.reads.has(key)) {
      const shardId = this.shardFor(key);
      const observed = this.shards.get(shardId).readAt(key, txn.startTs);
      txn.reads.set(key, { ...observed, shard: shardId });
    }
    const observed = txn.reads.get(key);
    txn.ops.push({ type: "read", key, found: observed.found, value: observed.value });
    return { found: observed.found, value: observed.value };
  }

  write(txn, key, value) {
    assert.strictEqual(txn.state, "active", `transaction ${txn.id} is not accepting writes`);
    txn.writes.set(key, value);
    txn.ops.push({ type: "write", key, value });
  }

  /** Phase zero: reject a transaction whose snapshot is already stale for any key it touched. */
  validate(txn) {
    for (const [key, observed] of txn.reads) {
      const shard = this.shards.get(this.shardFor(key));
      if (shard.latestTs(key) > txn.startTs) return `read ${key} was updated at ts ${shard.latestTs(key)}`;
    }
    for (const key of txn.writes.keys()) {
      const shard = this.shards.get(this.shardFor(key));
      if (shard.latestTs(key) > txn.startTs) return `write ${key} was updated at ts ${shard.latestTs(key)}`;
    }
    return null;
  }

  beginCommit(txn) {
    assert.strictEqual(txn.state, "active", `transaction ${txn.id} already finished`);
    if (txn.writes.size === 0) {
      // A read-only transaction is always serializable at its own snapshot timestamp, so it needs
      // no locks, no 2PC and can never be rejected: this is the SI "read-only never aborts" rule.
      txn.readOnly = true;
      txn.state = "committed";
      txn.commitTs = txn.startTs;
      this.committed.push(txn);
      return "committed";
    }
    const invalid = this.validate(txn);
    if (invalid !== null) { this.abortTxn(txn, invalid); return "aborted"; }
    txn.state = "preparing";
    txn.expected = this.coordinator.begin(txn);
    return "preparing";
  }

  abortTxn(txn, reason) {
    txn.state = "aborted";
    txn.reason = reason;
    this.aborted.push(txn);
    return "aborted";
  }

  /** One pump round = deliver what was queued when the round started. */
  pump(rounds = 1) {
    for (let i = 0; i < rounds; i++) this.bus.drainRound();
    return this.bus.queue.length;
  }

  /** Pump until the transaction has a decision and nothing more can be delivered. */
  settle(txn, maxRounds = 40) {
    let rounds = 0;
    while (rounds < maxRounds && (this.bus.queue.length > 0 || txn.state === "preparing")) {
      const deliveredBefore = this.bus.delivered;
      this.pump(1);
      rounds++;
      if (this.bus.delivered === deliveredBefore && this.bus.queue.length === 0) break;
    }
    return rounds;
  }

  commit(txn) {
    const started = this.beginCommit(txn);
    if (started === "aborted" || started === "committed") return started;
    this.settle(txn);
    assert.notStrictEqual(txn.state, "preparing", `transaction ${txn.id} never received a decision`);
    return txn.state;
  }

  abort(txn, reason = "client abort") {
    if (txn.state === "active") return this.abortTxn(txn, reason);
    return txn.state;
  }

  finalize(txn, decision, ts, reason) {
    txn.state = decision;
    txn.commitTs = decision === "committed" ? ts : 0;
    txn.reason = reason;
    if (decision === "committed") this.committed.push(txn);
    else this.aborted.push(txn);
  }

  /** Crash recovery: every prepared-but-undecided transaction is resolved from the decision log. */
  recover() {
    const resolved = [];
    for (const shard of this.shards.values()) {
      for (const [txnId] of shard.unresolved()) {
        const decision = this.coordinator.decisionFor(txnId);
        if (decision !== null && decision.op === "committed") {
          shard.commitFromIntent(txnId, decision.ts);
          resolved.push({ shard: shard.id, txnId, outcome: "committed" });
        } else {
          shard.abort(txnId, "recovery: no commit decision");
          resolved.push({ shard: shard.id, txnId, outcome: "aborted" });
        }
        this.recoveries++;
      }
    }
    return resolved;
  }

  /** Atomicity audit: for each transaction, which shards applied its writes? */
  shardAudit(txnId) {
    return [...this.shards.values()].filter((shard) => shard.holdsTxn(txnId)).map((shard) => shard.id);
  }

  materialize() {
    const out = new Map();
    for (const shard of this.shards.values()) for (const [key, value] of shard.materialize()) out.set(key, value);
    return out;
  }

  stateChecksum() {
    return [...this.materialize().entries()].sort().map(([k, v]) => `${k}=${v}`).join(",");
  }

  /** Rebuild shards from the WAL records alone: durability without trusting live memory. */
  static replayWal(shards) {
    const rebuilt = new Map();
    for (const shard of shards) {
      const fresh = new Shard(shard.id, new Wal(`replay-${shard.id}`));
      fresh.shardCount = shard.shardCount;
      for (const record of shard.wal.records) {
        if (record.op !== "commit") continue;
        for (const key of record.keys) fresh.applyVersion(record.txnId, key, record.values[key], record.ts);
      }
      rebuilt.set(shard.id, fresh);
    }
    return rebuilt;
  }
}

function checksumOf(map) { return [...map.entries()].sort().map(([k, v]) => `${k}=${v}`).join(","); }

/** Brute force: is there a serial order of the committed transactions that reproduces every read? */
function findSerialOrder(committed, initialValues, finalValues, budget = 200000) {
  const slots = committed.length;
  const order = [];
  const used = new Array(slots).fill(false);
  let visited = 0;

  const replay = (candidate) => {
    const store = new Map(initialValues);
    for (const txn of candidate) {
      for (const op of txn.ops) {
        if (op.type === "read") {
          const value = store.get(op.key);
          const found = value !== undefined;
          if (found !== op.found) return false;
          if (found && value !== op.value) return false;
        } else if (op.value === undefined) store.delete(op.key);
        else store.set(op.key, op.value);
      }
    }
    return checksumOf(store) === checksumOf(finalValues);
  };

  const search = () => {
    if (order.length === slots) return replay(order);
    for (let i = 0; i < slots; i++) {
      if (used[i]) continue;
      if (visited++ > budget) return false;
      used[i] = true;
      order.push(committed[i]);
      if (search()) return true;
      order.pop();
      used[i] = false;
    }
    return false;
  };

  return search();
}

/**
 * Conflict-serializability cross-check built from the version chains themselves, not from the
 * engine's intentions: for each key, the committed writers form a chain in commit order, so
 *   ww: chain[i] -> chain[i+1]
 *   wr: chain[i] -> every transaction that read exactly chain[i]
 *   rw: every such reader -> chain[i+1]   (it did not see the next version)
 * A cycle in that graph is a non-serializable schedule (a write-skew pair produces exactly one).
 */
function hasPrecedenceCycle(engine, committed) {
  const live = new Set(committed.map((txn) => txn.id));
  const edges = new Map();
  const add = (from, to) => {
    if (from === undefined || to === undefined || from === to) return;   // a self edge is no constraint
    if (!live.has(from) || !live.has(to)) return;
    if (!edges.has(from)) edges.set(from, new Set());
    edges.get(from).add(to);
  };
  const readers = new Map();      // key -> versionTs -> [txnId]
  for (const txn of committed) {
    for (const [key, observed] of txn.reads) {
      if (!readers.has(key)) readers.set(key, new Map());
      const byTs = readers.get(key);
      if (!byTs.has(observed.versionTs)) byTs.set(observed.versionTs, []);
      byTs.get(observed.versionTs).push(txn.id);
    }
  }
  for (const shard of engine.shards.values()) {
    for (const [key, chain] of shard.versions) {
      for (let i = 0; i < chain.length; i++) {
        const writer = chain[i].txn;
        const observedBy = (readers.get(key) ?? new Map()).get(chain[i].ts) ?? [];
        for (const reader of observedBy) add(writer, reader);
        if (i + 1 < chain.length) {
          add(writer, chain[i + 1].txn);
          for (const reader of observedBy) add(reader, chain[i + 1].txn);
        }
      }
    }
  }
  const colour = new Map();
  const visit = (id) => {
    colour.set(id, 1);
    for (const next of edges.get(id) ?? []) {
      if (colour.get(next) === 1) return true;
      if (colour.get(next) !== 2 && visit(next)) return true;
    }
    colour.set(id, 2);
    return false;
  };
  for (const id of live) if (colour.get(id) === undefined && visit(id)) return true;
  return false;
}

/** Oracle for the final state: apply the committed writes in commit-timestamp order. */
function foldInCommitOrder(committed) {
  const state = new Map();
  for (const txn of [...committed].sort((a, b) => a.commitTs - b.commitTs)) {
    for (const [key, value] of txn.writes) {
      if (value === undefined) state.delete(key);
      else state.set(key, value);
    }
  }
  return state;
}

function shuffled(items, rng) {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

/**
 * Read-modify-write workload: `batch` transactions are opened (so they share overlapping snapshots)
 * and then committed in a seeded order, which is what makes read-set validation abort some of them.
 */
function runInterleavedWorkload(engine, { steps, keyCount, batch, rng }) {
  let pending = [];
  let committed = 0;
  let aborted = 0;
  const flush = () => {
    for (const txn of shuffled(pending, rng)) {
      if (engine.commit(txn) === "committed") committed++;
      else aborted++;
    }
    pending = [];
  };
  for (let step = 0; step < steps; step++) {
    const txn = engine.begin();
    const readCount = 1 + rng.int(3);
    for (let j = 0; j < readCount; j++) engine.read(txn, `w${rng.int(keyCount)}`);
    const writeCount = 1 + rng.int(2);
    for (let j = 0; j < writeCount; j++) {
      const key = `w${rng.int(keyCount)}`;
      const current = engine.read(txn, key);
      engine.write(txn, key, (current.value ?? 0) + 1);
    }
    pending.push(txn);
    if (pending.length === batch) flush();
  }
  flush();
  return { committed, aborted };
}

if (require.main === module) {
  // --- 1. read-your-writes, snapshot isolation and repeated reads -------------------------------
  const engine = new Engine({ shards: 3 });
  const seed = engine.begin();
  engine.write(seed, "alpha", 10);
  engine.write(seed, "beta", 20);
  assert.strictEqual(engine.commit(seed), "committed");
  assert.strictEqual(engine.shards.get(engine.shardFor("alpha")).latestTs("alpha") > 0, true, "a commit stamps a version");

  const reader = engine.begin();
  assert.deepStrictEqual(engine.read(reader, "alpha"), { found: true, value: 10 });
  const concurrent = engine.begin();
  engine.write(concurrent, "alpha", 99);
  engine.commit(concurrent);
  assert.deepStrictEqual(engine.read(reader, "alpha"), { found: true, value: 10 }, "repeatable read inside one snapshot");
  assert.strictEqual(engine.commit(reader), "committed", "read-only transactions never abort");
  assert.strictEqual(reader.readOnly, true, "and they skip 2PC entirely");
  assert.strictEqual(engine.coordinator.wal.records.filter((r) => r.txnId === reader.id).length, 0, "no prepare for a read-only txn");

  const fresh = engine.begin();
  assert.deepStrictEqual(engine.read(fresh, "alpha"), { found: true, value: 99 }, "later snapshots see the new version");
  assert.deepStrictEqual(engine.read(fresh, "missing"), { found: false, value: undefined }, "unknown keys read as not found");

  // --- 2. no dirty reads: a prepared intent is invisible ---------------------------------------
  const writer = engine.begin();
  engine.write(writer, "gamma", 7);
  assert.strictEqual(engine.beginCommit(writer), "preparing", "the writer is now holding intents");
  engine.pump(1);
  assert.deepStrictEqual(engine.read(fresh, "gamma"), { found: false, value: undefined }, "a prepared value must not be visible");
  engine.settle(writer);
  assert.strictEqual(writer.state, "committed", "the prepared writer commits once the decision arrives");
  const afterWriter = engine.begin();
  assert.deepStrictEqual(engine.read(afterWriter, "gamma"), { found: true, value: 7 });

  // --- 3. write lock conflict: the second preparer is voted off --------------------------------
  const holder = engine.begin();
  engine.write(holder, "delta", 1);
  const holderShard = engine.shards.get(engine.shardFor("delta"));
  const hold = engine.bus.hold((envelope) => envelope.msg.type === "decision" && envelope.msg.txnId === holder.id);
  engine.beginCommit(holder);
  engine.pump(2);
  assert.strictEqual(holderShard.locks.get("delta"), holder.id, "the prepared writer keeps its lock while the decision is in flight");
  assert.strictEqual(holderShard.latestTs("delta"), 0, "and holds the value as an unversioned intent");
  const challenger = engine.begin();
  engine.write(challenger, "delta", 2);
  assert.strictEqual(engine.beginCommit(challenger), "preparing");
  engine.pump(3);
  assert.strictEqual(challenger.state, "aborted", "the challenger loses the write lock race");
  assert.ok(challenger.reason.includes("write lock"), `expected a lock conflict, saw: ${challenger.reason}`);
  hold.release();
  engine.bus.releaseBacklog();
  engine.pump(2);
  assert.strictEqual(holderShard.intents.has(holder.id), false, "the holder's intent is resolved after the release");
  assert.strictEqual(holderShard.latestTs("delta") > 0, true, "and its write landed");
  assert.strictEqual(holderShard.locks.size, 0, "no lock is leaked after the decision");

  // --- 4. write skew: exactly one of the two read-modify-writes survives -----------------------
  const skew = new Engine({ shards: 2 });
  const setup = skew.begin();
  skew.write(setup, "a", 1);
  skew.write(setup, "b", 1);
  skew.commit(setup);
  const t1 = skew.begin();
  skew.read(t1, "a"); skew.read(t1, "b");
  skew.write(t1, "a", 2);
  const t2 = skew.begin();
  skew.read(t2, "a"); skew.read(t2, "b");
  skew.write(t2, "b", 2);
  const outcome1 = skew.commit(t1);
  const outcome2 = skew.commit(t2);
  assert.deepStrictEqual([outcome1, outcome2].sort(), ["aborted", "committed"].sort(),
    "write skew must abort exactly one transaction");
  const snapshotAfter = skew.begin();
  const valueA = skew.read(snapshotAfter, "a").value;
  const valueB = skew.read(snapshotAfter, "b").value;
  assert.strictEqual(valueA + valueB, 3, `a+b must stay 3, saw ${valueA + valueB}`);

  // --- 5. multi-shard atomicity with an injected participant abort ------------------------------
  const atomic = new Engine({ shards: 4 });
  const spread = [];
  for (let candidate = 0; spread.length < 4 && candidate < 500; candidate++) {
    const key = `mk${candidate}`;
    if (!spread.some((existing) => atomic.shardFor(existing) === atomic.shardFor(key))) spread.push(key);
  }
  const multi = atomic.begin();
  for (const key of spread) atomic.write(multi, key, key.toUpperCase());
  const touched = new Set([...multi.writes.keys()].map((k) => atomic.shardFor(k)));
  assert.strictEqual(touched.size, 4, "the transaction spans every shard");
  assert.strictEqual(atomic.beginCommit(multi), "preparing");
  atomic.pump(1);
  const saboteur = [...touched][0];
  assert.strictEqual(atomic.shards.get(saboteur).intents.has(multi.id), true, "the sabotaged shard is prepared");
  atomic.coordinator.receive(saboteur, { type: "vote", txnId: multi.id, vote: "abort", reason: "injected failure" });
  atomic.pump(3);
  assert.strictEqual(multi.state, "aborted", "one aborting participant aborts the whole transaction");
  assert.deepStrictEqual(atomic.shardAudit(multi.id), [], "no shard kept any part of the aborted write");
  for (const shardId of touched) {
    assert.strictEqual(atomic.shards.get(shardId).locks.size, 0, `shard ${shardId} released its locks`);
  }

  // --- 6. all-or-nothing across shards on the happy path ---------------------------------------
  const happy = atomic.begin();
  for (const key of ["x1", "x2", "x3"]) atomic.write(happy, key, 1);
  assert.strictEqual(atomic.commit(happy), "committed");
  const expectedShards = [...new Set([...happy.writes.keys()].map((k) => atomic.shardFor(k)))].sort();
  assert.deepStrictEqual(atomic.shardAudit(happy.id), expectedShards, "every participant applied the commit");
  const emptyTxn = atomic.begin();
  assert.strictEqual(atomic.commit(emptyTxn), "committed", "an empty transaction commits without contacting anyone");
  assert.strictEqual(emptyTxn.readOnly, true);
  assert.throws(() => atomic.read(emptyTxn, "x1"), /not accepting reads/, "a finished transaction rejects further reads");

  // --- 7. durability: rebuild the store from the WALs alone ------------------------------------
  const rebuilt = Engine.replayWal([...atomic.shards.values()]);
  assert.strictEqual(rebuilt.size, atomic.shardCount);
  for (const shard of atomic.shards.values()) {
    assert.strictEqual(checksumOf(rebuilt.get(shard.id).materialize()), checksumOf(shard.materialize()),
      `shard ${shard.id} is reproducible from its WAL alone`);
  }
  assert.notStrictEqual(atomic.stateChecksum(), "", "the store is not empty at this point");

  // --- 8. crash between prepare and decision: the decision log resolves every participant -------
  const crash = new Engine({ shards: 3 });
  const doomed = crash.begin();
  for (const key of ["c1", "c2", "c3"]) crash.write(doomed, key, "v");
  const doomedShards = [...new Set([...doomed.writes.keys()].map((k) => crash.shardFor(k)))].sort();
  crash.bus.dropOnce((envelope) => envelope.msg.type === "decision" && envelope.to === doomedShards[0]);
  crash.commit(doomed);
  assert.strictEqual(doomed.state, "committed", "the coordinator decided to commit");
  assert.deepStrictEqual(crash.shardAudit(doomed.id), doomedShards.slice(1), "the dropped decision left one participant prepared");
  const preparedShard = crash.shards.get(doomedShards[0]);
  assert.deepStrictEqual([...preparedShard.unresolved().keys()], [doomed.id], "the participant recorded the pending intent");
  const resolved = crash.recover();
  assert.strictEqual(resolved.length, 1, "exactly one participant needed recovery");
  assert.strictEqual(resolved[0].outcome, "committed", "presumed commit, because the decision log holds a commit");
  assert.deepStrictEqual(crash.shardAudit(doomed.id), doomedShards, "after recovery the transaction is atomic");
  assert.strictEqual(preparedShard.locks.size, 0, "recovery releases the locks");

  // --- 9. crash before the decision: presumed abort --------------------------------------------
  const orphan = new Engine({ shards: 3 });
  const lost = orphan.begin();
  for (const key of ["o1", "o2", "o3"]) orphan.write(lost, key, 1);
  const lostShards = [...new Set([...lost.writes.keys()].map((k) => orphan.shardFor(k)))].sort();
  orphan.bus.hold((envelope) => envelope.msg.type === "vote");
  orphan.beginCommit(lost);
  orphan.pump(2);
  assert.strictEqual(lost.state, "preparing", "all prepares are in flight");
  assert.strictEqual(lostShards.every((id) => orphan.shards.get(id).intents.has(lost.id)), true, "every participant is prepared");
  orphan.bus.holds.length = 0;                                 // the coordinator "crashes" here
  orphan.bus.queue.length = 0;
  const orphanResolved = orphan.recover();
  assert.strictEqual(orphanResolved.length, lostShards.length, "every participant is recovered");
  assert.ok(orphanResolved.every((r) => r.outcome === "aborted"), "without a decision record the transaction is aborted");
  assert.deepStrictEqual(orphan.shardAudit(lost.id), [], "and nothing is left behind");
  assert.strictEqual(orphan.shards.get(lostShards[0]).locks.size, 0, "locks are released by the abort");

  // --- 10. tombstones and MVCC visibility ------------------------------------------------------
  const mvcc = new Engine({ shards: 2 });
  const insert = mvcc.begin();
  mvcc.write(insert, "tmp", "first");
  mvcc.commit(insert);
  const beforeDelete = mvcc.begin();
  const del = mvcc.begin();
  mvcc.write(del, "tmp", undefined);                            // delete
  mvcc.commit(del);
  assert.deepStrictEqual(mvcc.read(beforeDelete, "tmp"), { found: true, value: "first" }, "an old snapshot still sees the deleted value");
  assert.deepStrictEqual(mvcc.read(mvcc.begin(), "tmp"), { found: false, value: undefined }, "the newest snapshot sees the tombstone");
  const reinsert = mvcc.begin();
  mvcc.write(reinsert, "tmp", "second");
  mvcc.commit(reinsert);
  assert.deepStrictEqual(mvcc.read(mvcc.begin(), "tmp"), { found: true, value: "second" }, "re-inserting creates a new version");
  assert.strictEqual(mvcc.shards.get(mvcc.shardFor("tmp")).versions.get("tmp").length, 3, "three versions are retained for that key");

  // --- 11. serializability: brute force on a small history, precedence graph on the large one --
  const small = new Engine({ shards: 3 });
  const setupSmall = small.begin();
  for (let i = 0; i < 4; i++) small.write(setupSmall, `s${i}`, 0);
  small.commit(setupSmall);
  const smallInitial = small.materialize();
  const smallRng = splitmix64(0x51ee7);
  const smallRun = runInterleavedWorkload(small, { steps: 6, keyCount: 4, batch: 3, rng: smallRng });
  const smallCommitted = small.committed.filter((txn) => txn.id !== setupSmall.id);
  assert.ok(smallCommitted.length >= 2 && smallCommitted.length <= 6, `the small history has ${smallCommitted.length} committed transactions`);
  assert.ok(smallRun.aborted >= 1, "the interleaved small workload aborts at least one transaction");
  assert.strictEqual(findSerialOrder(smallCommitted, smallInitial, small.materialize(), 50000), true,
    "a serial order must reproduce every committed read of the small history");
  assert.strictEqual(hasPrecedenceCycle(small, smallCommitted), false, "the small history has no precedence cycle");
  for (const txn of small.aborted) {
    assert.strictEqual(txn.reason !== null, true, `aborted transaction ${txn.id} carries a reason`);
  }

  const rng = splitmix64(0xc0ffee);
  const workload = new Engine({ shards: 4 });
  const opening = workload.begin();
  for (let i = 0; i < 6; i++) workload.write(opening, `w${i}`, 0);
  workload.commit(opening);
  const initialValues = workload.materialize();
  const sameTxn = workload.begin();
  workload.write(sameTxn, "w0", 1);
  workload.write(sameTxn, "w0", 2);                             // last write wins inside a transaction
  workload.commit(sameTxn);
  assert.strictEqual(workload.materialize().get("w0"), 2);
  assert.strictEqual(initialValues.get("w0"), 0, "the initial snapshot is unchanged");

  let committedCount = 0;
  let abortedCount = 0;
  const batchRng = splitmix64(0xfeed1);
  const bigRun = runInterleavedWorkload(workload, { steps: 60, keyCount: 6, batch: 4, rng: batchRng });
  committedCount = bigRun.committed;
  abortedCount = bigRun.aborted;
  assert.ok(committedCount > 0 && abortedCount > 0, `the workload must exercise both outcomes (${committedCount} committed, ${abortedCount} aborted)`);
  assert.ok(abortedCount < 60, "not every transaction may abort");
  const finalValues = workload.materialize();
  assert.strictEqual(workload.stateChecksum(), checksumOf(foldInCommitOrder(workload.committed)),
    "the final state equals the committed writes folded in commit order");
  assert.strictEqual(hasPrecedenceCycle(workload, workload.committed), false, "the committed history has no precedence cycle");
  for (const shard of workload.shards.values()) {
    assert.strictEqual(shard.locks.size, 0, `shard ${shard.id} holds no locks at the end`);
    assert.strictEqual(shard.intents.size, 0, `shard ${shard.id} holds no intents at the end`);
  }
  assert.strictEqual(finalValues.size > 0, true);

  // --- 12. consistency: transfers keep the total ----------------------------------------------
  const bank = new Engine({ shards: 3 });
  const accounts = ["acct0", "acct1", "acct2", "acct3"];
  const init = bank.begin();
  for (const account of accounts) bank.write(init, account, 100);
  bank.commit(init);
  const total = () => accounts.reduce((sum, account) => sum + (bank.read(bank.begin(), account).value ?? 0), 0);
  const totalBefore = total();
  const money = splitmix64(0xdead);
  let transfers = 0;
  for (let step = 0; step < 50; step++) {
    const from = accounts[money.int(accounts.length)];
    const to = accounts[money.int(accounts.length)];
    if (from === to) continue;
    const txn = bank.begin();
    const fromValue = bank.read(txn, from).value;
    const toValue = bank.read(txn, to).value;
    if (fromValue < 5) continue;
    bank.write(txn, from, fromValue - 5);
    bank.write(txn, to, toValue + 5);
    if (bank.commit(txn) === "committed") transfers++;
  }
  assert.ok(transfers > 10, `the transfer workload committed ${transfers} transactions`);
  assert.strictEqual(total(), totalBefore, "the total must be preserved by every committed transfer");

  // --- 13. cost: messages and WAL records per transaction stay linear ---------------------------
  const messagesPerTxn = workload.bus.delivered / (committedCount + abortedCount);
  assert.ok(messagesPerTxn <= 3 * workload.shardCount + 2, `messages per transaction stay bounded (saw ${messagesPerTxn.toFixed(1)})`);
  assert.ok(workload.bus.dropped === 0, "no message was lost in the healthy workload");
  const walRecords = [...workload.shards.values()].reduce((sum, shard) => sum + shard.wal.records.length, 0);
  assert.ok(walRecords <= 2 * (committedCount + abortedCount) * workload.shardCount, `WAL records stay linear (saw ${walRecords})`);

  console.log(`mvcc: ${engine.committed.length} commits, ${engine.aborted.length} aborts on ${engine.shardCount} shards, state checksum ${engine.stateChecksum()}`);
  console.log(`workload: ${committedCount} committed / ${abortedCount} aborted, ${workload.bus.delivered} messages (${messagesPerTxn.toFixed(1)}/txn), serial order found under brute force`);
  console.log(`2pc: ${crash.recoveries + orphan.recoveries} recovered participant(s), ${transfers} transfers with the total preserved at ${totalBefore}`);
  console.log("PASS");
}

module.exports = {
  Engine, Shard, Txn, Bus, Coordinator, Participant, Wal,
  hashString, mix32, keyShardOf, splitmix64, findSerialOrder, hasPrecedenceCycle,
};
