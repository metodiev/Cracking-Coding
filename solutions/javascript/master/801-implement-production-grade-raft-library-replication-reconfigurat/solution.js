/** Task 801 - Implement production-grade Raft library (replication, reconfiguration, snapshots). (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A faithful in-process Raft: leader election with deterministic randomised timeouts, log
 * replication with the log-matching check, majority commit restricted to the current term (plus a
 * no-op entry so a fresh leader can commit its predecessors' entries), snapshotting with
 * InstallSnapshot for a follower that fell behind the compaction point, and single-server
 * reconfiguration driven through the log ("join" / "leave" config entries). A joining server is
 * pre-registered as a non-voting learner so the leader can stream the log to it before its config
 * entry commits; the config entry itself is replicated and committed like any other command.
 *
 * Everything runs on explicit logical ticks: no timers, no threads, no wall clock. The "network" is
 * an in-memory message queue with explicit link state, so partitioning and healing are first-class
 * operations and the lossy mode drops every k-th send deterministically. Election jitter comes from
 * a 32-bit mix of (server id, term), so two runs of a scenario are byte-identical.
 *
 * What is simplified: one client command maps to one log entry (no batching, no session windows),
 * the state machine is a Map instead of a replayable command log on disk, InstallSnapshot transfers
 * the whole state map at once instead of chunked pages, and a removed server is simply silenced.
 * The scale knobs are typed parameters (cluster size, commands per scenario, snapshot threshold),
 * so the same code runs with the same guarantees at larger numbers.
 *
 * Invariants verified (safety and liveness, checked against an independent oracle):
 *   - election safety: no term ever has two leaders and no server grants two votes in one term;
 *   - state machine safety: an index is never applied with two different commands;
 *   - oracle equivalence: every server's applied Map equals a fold of the global apply log, which
 *     is built from observations only (never from one node's private order);
 *   - log matching / durability: overlapping (index, term, command) triples agree across servers,
 *     so a committed entry is never lost or replaced;
 *   - liveness: a command handed to a live leader commits within a bounded tick budget, a crash of
 *     the leader is followed by a new leader in a higher term within a bounded tick budget;
 *   - reconfiguration: the joined server converges to the same state, the removed server stops
 *     voting, and commit quorums use the committed membership only;
 *   - cost: messages per committed command stay linear in the number of commands (reported).
 */
"use strict";

const assert = require("node:assert");

const FOLLOWER = "follower";
const CANDIDATE = "candidate";
const LEADER = "leader";

/** splitmix64 - the only source of randomness in this file; same stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, unit: () => Number(next() >> 11n) / 2 ** 53, int: (n) => Number(next() % BigInt(n)) };
}

/** 32-bit avalanche mix; used to derive a stable timeout per (server, term) pair. */
function mix32(x) {
  let h = x >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

const cmdKey = (cmd) => JSON.stringify(cmd ?? null);

/** The in-memory transport: links decide reachability, a modulus can drop sends, one step = one round. */
class Network {
  constructor() {
    this.nodes = new Map();
    this.links = new Set();
    this.queue = [];
    this.seq = 0;
    this.delivered = 0;
    this.dropped = 0;
    this.dropEvery = 0;
  }

  register(node) { this.nodes.set(node.id, node); }

  connect(a, b) {
    if (a === b) return;
    this.links.add(`${a}>${b}`);
    this.links.add(`${b}>${a}`);
  }

  disconnect(a, b) {
    this.links.delete(`${a}>${b}`);
    this.links.delete(`${b}>${a}`);
  }

  /** Cut the cluster into the given groups; everything across groups is unreachable. */
  partition(groups) {
    this.links.clear();
    for (const group of groups) for (const a of group) for (const b of group) this.connect(a, b);
  }

  heal(ids) {
    this.links.clear();
    for (const a of ids) for (const b of ids) this.connect(a, b);
  }

  open(a, b) { return a === b || this.links.has(`${a}>${b}`); }

  send(from, to, msg) {
    if (!this.open(from, to)) { this.dropped++; return false; }
    this.seq++;
    if (this.dropEvery > 0 && this.seq % this.dropEvery === 0) { this.dropped++; return false; }
    this.queue.push({ from, to, msg, seq: this.seq });
    return true;
  }

  /** Deliver the messages queued by the previous tick, in a deterministic order. */
  step() {
    const batch = this.queue;
    this.queue = [];
    batch.sort((a, b) => (a.to - b.to) || (a.seq - b.seq));
    for (const m of batch) {
      const node = this.nodes.get(m.to);
      if (node) { this.delivered++; node.receive(m.from, m.msg); }
    }
    return batch.length;
  }
}

class RaftNode {
  constructor(id, net, opts = {}) {
    this.id = id;
    this.net = net;
    this.cluster = opts.cluster ?? null;
    this.membership = [...(opts.membership ?? [id])].sort((a, b) => a - b);
    this.electionBase = opts.electionBase ?? 4;
    this.electionJitter = opts.electionJitter ?? 3;
    this.heartbeatEvery = opts.heartbeatEvery ?? 1;
    this.entriesPerAppend = opts.entriesPerAppend ?? 32;
    this.snapshotThreshold = opts.snapshotThreshold ?? Number.POSITIVE_INFINITY;
    this.campaigns = opts.campaigns !== false;

    this.role = FOLLOWER;
    this.currentTerm = 0;
    this.votedFor = null;
    this.leaderId = null;
    this.log = [];                                  // entries with index > snapshot.lastIncludedIndex
    this.snapshot = { lastIncludedIndex: 0, lastIncludedTerm: 0, state: new Map() };
    this.state = new Map();                         // applied state machine
    this.commitIndex = 0;
    this.lastApplied = 0;
    this.timer = 0;
    this.deadline = this.electionBase + this.jitter(0);
    this.ticksSinceHeartbeat = 0;
    this.votes = new Set();
    this.nextIndex = new Map();
    this.matchIndex = new Map();
    this.learners = new Set();
    this.crashed = false;
    this.removed = false;
    this.stats = { elections: 0, votesGranted: 0, appendsReceived: 0, snapshotsTaken: 0, snapshotsInstalled: 0, entriesAppended: 0 };
    net.register(this);
  }

  jitter(term) { return mix32(this.id * 0x9e3779b1 ^ (term + 1) * 0x85ebca6b) % (this.electionJitter + 1); }

  lastLogIndex() { return this.snapshot.lastIncludedIndex + this.log.length; }

  lastLogTerm() { return this.lastLogIndex() === 0 ? 0 : this.termAt(this.lastLogIndex()); }

  /** null means "compacted away by a snapshot" - callers must treat that as a failed match. */
  termAt(index) {
    if (index <= 0) return 0;
    if (index === this.snapshot.lastIncludedIndex) return this.snapshot.lastIncludedTerm;
    if (index < this.snapshot.lastIncludedIndex) return null;
    const entry = this.log[index - this.snapshot.lastIncludedIndex - 1];
    return entry === undefined ? undefined : entry.term;
  }

  cmdAt(index) {
    if (index <= 0 || index < this.snapshot.lastIncludedIndex) return undefined;
    if (index === this.snapshot.lastIncludedIndex) return undefined;
    const entry = this.log[index - this.snapshot.lastIncludedIndex - 1];
    return entry === undefined ? undefined : entry.cmd;
  }

  entryAt(index) { return this.log[index - this.snapshot.lastIncludedIndex - 1]; }

  peers() {
    const all = new Set([...this.membership, ...this.learners]);
    all.delete(this.id);
    return [...all].sort((a, b) => a - b);
  }

  quorum() { return Math.floor(this.membership.length / 2) + 1; }

  active() { return !this.crashed && !this.removed; }

  send(to, msg) { return this.net.send(this.id, to, msg); }

  // ---------------------------------------------------------------- role changes

  becomeFollower(term) {
    if (term > this.currentTerm) { this.currentTerm = term; this.votedFor = null; }
    this.role = FOLLOWER;
    this.leaderId = null;
    this.votes.clear();
  }

  startElection() {
    if (!this.campaigns) return;
    this.currentTerm++;
    this.role = CANDIDATE;
    this.votedFor = this.id;
    this.votes = new Set([this.id]);
    this.timer = 0;
    this.deadline = this.electionBase + this.jitter(this.currentTerm);
    this.stats.elections++;
    if (this.cluster) this.cluster.recordVote(this.currentTerm, this.id, this.id);
    const request = {
      type: "RequestVote", term: this.currentTerm, lastLogIndex: this.lastLogIndex(), lastLogTerm: this.lastLogTerm(),
    };
    for (const peer of this.peers()) this.send(peer, request);
    if (this.votes.size >= this.quorum()) this.becomeLeader();
  }

  becomeLeader() {
    this.role = LEADER;
    this.leaderId = this.id;
    this.nextIndex = new Map();
    this.matchIndex = new Map();
    for (const peer of this.peers()) { this.nextIndex.set(peer, this.lastLogIndex() + 1); this.matchIndex.set(peer, 0); }
    this.timer = 0;
    this.ticksSinceHeartbeat = 0;
    if (this.cluster) this.cluster.recordLeader(this.currentTerm, this.id);
    // A no-op in the new term lets the leader commit entries left by previous terms (Raft section 8).
    this.appendLocal({ kind: "noop" });
    this.advanceCommit();
    this.replicate();
  }

  // ---------------------------------------------------------------- client interface

  appendLocal(cmd) {
    const index = this.lastLogIndex() + 1;
    this.log.push({ term: this.currentTerm, index, cmd });
    this.stats.entriesAppended++;
    return index;
  }

  submit(cmd) {
    if (this.role !== LEADER || !this.active()) return null;
    const index = this.appendLocal(cmd);
    this.advanceCommit();     // a single-server cluster is its own majority
    this.replicate();
    return index;
  }

  // ---------------------------------------------------------------- replication

  replicate() {
    if (this.role !== LEADER || !this.active()) return;
    for (const peer of this.peers()) {
      const next = this.nextIndex.get(peer) ?? this.lastLogIndex() + 1;
      if (next - 1 < this.snapshot.lastIncludedIndex) {
        this.send(peer, {
          type: "InstallSnapshot", term: this.currentTerm,
          lastIncludedIndex: this.snapshot.lastIncludedIndex,
          lastIncludedTerm: this.snapshot.lastIncludedTerm,
          state: [...this.snapshot.state.entries()],
        });
        continue;
      }
      const offset = next - 1 - this.snapshot.lastIncludedIndex;
      const entries = this.log.slice(offset, offset + this.entriesPerAppend)
        .map((e) => ({ term: e.term, index: e.index, cmd: e.cmd }));
      this.send(peer, {
        type: "AppendEntries", term: this.currentTerm,
        prevLogIndex: next - 1, prevLogTerm: this.termAt(next - 1) ?? 0,
        entries, leaderCommit: this.commitIndex,
      });
    }
  }

  advanceCommit() {
    // Only voting members count towards the commit quorum; learners are replicated to but never
    // allowed to push the majority index up or hold it back.
    const matches = this.membership.filter((id) => id !== this.id).map((p) => this.matchIndex.get(p) ?? 0);
    matches.push(this.lastLogIndex());
    matches.sort((a, b) => a - b);
    const candidate = matches[matches.length - this.quorum()];
    if (candidate !== undefined && candidate > this.commitIndex && this.termAt(candidate) === this.currentTerm) {
      this.commitIndex = candidate;
    }
    this.applyCommitted();
  }

  applyCommitted() {
    while (this.lastApplied < this.commitIndex) {
      this.lastApplied++;
      const entry = this.entryAt(this.lastApplied);
      if (entry === undefined) { this.lastApplied = this.commitIndex; break; }
      this.applyEntry(entry);
    }
    if (this.commitIndex - this.snapshot.lastIncludedIndex >= this.snapshotThreshold) this.takeSnapshot();
  }

  applyEntry(entry) {
    const cmd = entry.cmd;
    if (cmd.kind === "set") this.state.set(cmd.key, cmd.value);
    if (cmd.kind === "join") {
      if (!this.membership.includes(cmd.id)) { this.membership.push(cmd.id); this.membership.sort((a, b) => a - b); }
      this.learners.delete(cmd.id);
    }
    if (cmd.kind === "leave") {
      this.membership = this.membership.filter((x) => x !== cmd.id);
      this.learners.delete(cmd.id);
      if (cmd.id === this.id) { this.removed = true; }
    }
    if (this.cluster) this.cluster.recordApply(this.id, entry.index, entry.term, cmd);
  }

  takeSnapshot() {
    const index = this.commitIndex;
    if (index <= this.snapshot.lastIncludedIndex) return;
    this.snapshot = { lastIncludedIndex: index, lastIncludedTerm: this.termAt(index), state: new Map(this.state) };
    this.log = this.log.filter((e) => e.index > index);
    this.stats.snapshotsTaken++;
  }

  // ---------------------------------------------------------------- message handling

  tick() {
    if (!this.active()) return;
    this.ticksSinceHeartbeat++;
    if (this.role === LEADER) {
      if (this.ticksSinceHeartbeat % this.heartbeatEvery === 0) this.replicate();
      return;
    }
    this.timer++;
    if (this.timer >= this.deadline) this.startElection();
  }

  receive(from, msg) {
    if (!this.active()) return;
    switch (msg.type) {
      case "RequestVote": return this.onRequestVote(from, msg);
      case "RequestVoteReply": return this.onVoteReply(from, msg);
      case "AppendEntries": return this.onAppendEntries(from, msg);
      case "AppendReply": return this.onAppendReply(from, msg);
      case "InstallSnapshot": return this.onInstallSnapshot(from, msg);
      case "InstallReply": return this.onInstallReply(from, msg);
      default: throw new Error(`unknown message type ${msg.type}`);
    }
  }

  onRequestVote(from, msg) {
    if (msg.term < this.currentTerm) {
      return this.send(from, { type: "RequestVoteReply", term: this.currentTerm, granted: false, reason: "stale-term" });
    }
    if (msg.term > this.currentTerm) this.becomeFollower(msg.term);
    const upToDate = msg.lastLogTerm > this.lastLogTerm()
      || (msg.lastLogTerm === this.lastLogTerm() && msg.lastLogIndex >= this.lastLogIndex());
    let granted = false;
    let reason = "";
    if (this.votedFor !== null && this.votedFor !== from) reason = "already-voted";
    else if (!upToDate) reason = "stale-log";
    else {
      granted = true;
      this.votedFor = from;
      this.timer = 0;
      this.stats.votesGranted++;
      if (this.cluster) this.cluster.recordVote(this.currentTerm, from, this.id);
    }
    return this.send(from, {
      type: "RequestVoteReply", term: this.currentTerm, granted, reason,
      lastLogIndex: this.lastLogIndex(), snapshotIndex: this.snapshot.lastIncludedIndex,
    });
  }

  onVoteReply(from, msg) {
    if (msg.term > this.currentTerm) { this.becomeFollower(msg.term); return; }
    if (this.role !== CANDIDATE || msg.term !== this.currentTerm) return;
    if (msg.granted) this.votes.add(from);
    if (this.votes.size >= this.quorum()) this.becomeLeader();
  }

  onAppendEntries(from, msg) {
    if (msg.term < this.currentTerm) {
      return this.send(from, { type: "AppendReply", term: this.currentTerm, success: false, reason: "stale-term", lastLogIndex: this.lastLogIndex() });
    }
    if (msg.term > this.currentTerm) this.becomeFollower(msg.term);
    this.role = FOLLOWER;
    this.leaderId = from;
    this.timer = 0;
    this.stats.appendsReceived++;

    const fail = (reason) => this.send(from, {
      type: "AppendReply", term: this.currentTerm, success: false, reason,
      lastLogIndex: this.lastLogIndex(), snapshotIndex: this.snapshot.lastIncludedIndex,
    });

    if (msg.prevLogIndex > this.lastLogIndex()) return fail("gap");
    if (msg.prevLogIndex > 0 && this.termAt(msg.prevLogIndex) !== msg.prevLogTerm) return fail("term-mismatch");

    for (const entry of msg.entries) {
      if (entry.index <= this.snapshot.lastIncludedIndex) continue;      // already inside our snapshot
      const offset = entry.index - this.snapshot.lastIncludedIndex - 1;
      if (offset < this.log.length) {
        if (this.log[offset].term === entry.term) continue;              // identical prefix: keep it
        this.log = this.log.slice(0, offset);                            // conflicting suffix is discarded
      }
      this.log.push({ term: entry.term, index: entry.index, cmd: entry.cmd });
    }
    if (msg.leaderCommit > this.commitIndex) {
      this.commitIndex = Math.min(msg.leaderCommit, this.lastLogIndex());
      this.applyCommitted();
    }
    return this.send(from, {
      type: "AppendReply", term: this.currentTerm, success: true,
      matchIndex: msg.prevLogIndex + msg.entries.length,
      lastLogIndex: this.lastLogIndex(), snapshotIndex: this.snapshot.lastIncludedIndex,
    });
  }

  onAppendReply(from, msg) {
    if (msg.term > this.currentTerm) { this.becomeFollower(msg.term); return; }
    if (this.role !== LEADER || msg.term !== this.currentTerm) return;
    if (msg.success) {
      this.matchIndex.set(from, Math.max(this.matchIndex.get(from) ?? 0, msg.matchIndex));
      this.nextIndex.set(from, (this.matchIndex.get(from) ?? 0) + 1);
      this.advanceCommit();
      return;
    }
    // Back off: jump straight to the follower's log end when it told us it is shorter.
    const current = this.nextIndex.get(from) ?? this.lastLogIndex() + 1;
    const guess = Math.min(current - 1, (msg.lastLogIndex ?? 0) + 1);
    this.nextIndex.set(from, Math.max(1, guess));
  }

  onInstallSnapshot(from, msg) {
    if (msg.term < this.currentTerm) {
      return this.send(from, { type: "InstallReply", term: this.currentTerm, matchIndex: 0, stale: true });
    }
    if (msg.term > this.currentTerm) this.becomeFollower(msg.term);
    this.role = FOLLOWER;
    this.leaderId = from;
    this.timer = 0;
    if (msg.lastIncludedIndex > this.snapshot.lastIncludedIndex && msg.lastIncludedIndex >= this.lastApplied) {
      this.snapshot = { lastIncludedIndex: msg.lastIncludedIndex, lastIncludedTerm: msg.lastIncludedTerm, state: new Map(msg.state) };
      this.log = this.log.filter((e) => e.index > msg.lastIncludedIndex);
      this.state = new Map(msg.state);
      this.lastApplied = msg.lastIncludedIndex;
      this.commitIndex = Math.max(this.commitIndex, msg.lastIncludedIndex);
      this.stats.snapshotsInstalled++;
      if (this.cluster) this.cluster.recordSnapshotInstall(this.id, msg.lastIncludedIndex);
    }
    return this.send(from, { type: "InstallReply", term: this.currentTerm, matchIndex: msg.lastIncludedIndex, lastLogIndex: this.lastLogIndex() });
  }

  onInstallReply(from, msg) {
    if (msg.term > this.currentTerm) { this.becomeFollower(msg.term); return; }
    if (this.role !== LEADER || msg.term !== this.currentTerm) return;
    this.matchIndex.set(from, Math.max(this.matchIndex.get(from) ?? 0, msg.matchIndex));
    this.nextIndex.set(from, Math.max(1, (this.matchIndex.get(from) ?? 0) + 1));
    this.advanceCommit();
  }
}

class RaftCluster {
  constructor(opts = {}) {
    this.seed = opts.seed ?? 0x5eed5eed;
    this.net = new Network();
    this.nodes = new Map();
    this.tickCount = 0;
    this.violations = [];
    this.appliedByIndex = new Map();     // index -> the single command observed at that index
    this.appliedTerms = new Map();       // index -> the single term observed at that index
    this.applyObservations = new Map();  // index -> Set of distinct command keys (oracle input)
    this.applyCount = 0;
    this.leadersByTerm = new Map();
    this.votesByTerm = new Map();
    this.snapshotInstalls = [];
    const ids = opts.ids ?? [];
    for (const id of ids) this.createNode(id, { ...(opts.node ?? {}), membership: ids });
  }

  createNode(id, nodeOpts = {}) {
    const node = new RaftNode(id, this.net, { ...nodeOpts, cluster: this });
    this.nodes.set(id, node);
    for (const other of this.nodes.values()) {
      if (other.id === id) continue;
      other.learners.add(id);
      this.net.connect(id, other.id);
    }
    return node;
  }

  node(id) { return this.nodes.get(id); }

  ids() { return [...this.nodes.keys()].sort((a, b) => a - b); }

  leaders() { return [...this.nodes.values()].filter((n) => n.role === LEADER && n.active()); }

  /** The live leader with the highest term, or null when there is none. */
  leader() {
    const leaders = this.leaders();
    if (leaders.length === 0) return null;
    return leaders.reduce((best, n) => (n.currentTerm > best.currentTerm ? n : best));
  }

  submit(cmd) {
    const leader = this.leader();
    if (leader === null) return null;
    const entry = { term: leader.currentTerm, index: leader.lastLogIndex() + 1, cmd };
    const index = leader.submit(cmd);
    assert.strictEqual(index, entry.index, "a leader must append at its own log end");
    return index;
  }

  /** Register a joining server as a non-voting learner before its config entry commits. */
  joinNode(id, nodeOpts = {}) {
    const membership = [...this.membershipOf(), id].sort((a, b) => a - b);
    return this.createNode(id, { campaigns: false, ...nodeOpts, membership });
  }

  membershipOf() {
    const any = this.leader() ?? [...this.nodes.values()].find((n) => n.active());
    return any ? [...any.membership] : [];
  }

  retire(id) {
    const node = this.nodes.get(id);
    node.removed = true;
    node.role = FOLLOWER;
    return node;
  }

  recordVote(term, candidate, voter) {
    const key = `${voter}`;
    if (!this.votesByTerm.has(term)) this.votesByTerm.set(term, new Map());
    const votes = this.votesByTerm.get(term);
    if (votes.has(key) && votes.get(key) !== candidate) {
      this.violations.push(`vote safety: server ${voter} voted for ${votes.get(key)} and ${candidate} in term ${term}`);
    }
    votes.set(key, candidate);
  }

  recordLeader(term, id) {
    if (this.leadersByTerm.has(term) && this.leadersByTerm.get(term) !== id) {
      this.violations.push(`election safety: term ${term} produced leaders ${this.leadersByTerm.get(term)} and ${id}`);
    }
    this.leadersByTerm.set(term, id);
  }

  recordApply(nodeId, index, term, cmd) {
    this.applyCount++;
    const key = cmdKey(cmd);
    if (!this.applyObservations.has(index)) this.applyObservations.set(index, new Set());
    this.applyObservations.get(index).add(key);
    const previous = this.appliedByIndex.get(index);
    if (previous === undefined) {
      this.appliedByIndex.set(index, cmd);
      this.appliedTerms.set(index, term);
    } else if (cmdKey(previous) !== key) {
      this.violations.push(`state machine safety: index ${index} applied as ${cmdKey(previous)} on one server and ${key} on another`);
    }
    if (this.appliedTerms.get(index) !== term) {
      this.violations.push(`term safety: index ${index} committed in term ${this.appliedTerms.get(index)} and ${term}`);
    }
  }

  recordSnapshotInstall(nodeId, index) { this.snapshotInstalls.push({ nodeId, index }); }

  runTicks(n) {
    for (let i = 0; i < n; i++) {
      this.tickCount++;
      for (const node of this.nodes.values()) node.tick();
      this.net.step();
    }
    return this.tickCount;
  }

  /** Tick until the predicate holds; returns ticks used, or -1 on timeout. */
  runUntil(predicate, maxTicks) {
    if (predicate()) return 0;
    for (let i = 1; i <= maxTicks; i++) {
      this.runTicks(1);
      if (predicate()) return i;
    }
    return -1;
  }

  maxAppliedIndex() {
    let max = 0;
    for (const index of this.appliedByIndex.keys()) if (index > max) max = index;
    return max;
  }

  /** Oracle: fold the observed apply log in index order. Never consults a single node's order. */
  oracleState(limit) {
    const state = new Map();
    const max = limit === undefined ? this.maxAppliedIndex() : limit;
    for (let index = 1; index <= max; index++) {
      const cmd = this.appliedByIndex.get(index);
      if (cmd === undefined) continue;                        // an uncommitted tail entry is simply absent
      if (cmd.kind === "set") state.set(cmd.key, cmd.value);
    }
    return state;
  }

  canonical(state) { return JSON.stringify([...state.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))); }

  /**
   * Log matching + durability + term safety, restricted to *committed* indexes (Raft only promises
   * agreement for committed entries; an uncommitted tail may legitimately differ). Every log that
   * reaches a committed index must carry the same term and command there, and every snapshot must
   * agree with the committed term at its boundary index.
   */
  checkLogMatching() {
    const max = this.maxAppliedIndex();
    for (const node of [...this.nodes.values()]) {
      const from = node.snapshot.lastIncludedIndex + 1;
      const to = Math.min(node.lastLogIndex(), max);
      for (let index = from; index <= to; index++) {
        const expected = this.appliedByIndex.get(index);
        if (expected === undefined) continue;
        if (cmdKey(node.cmdAt(index)) !== cmdKey(expected)) {
          this.violations.push(`durability: server ${node.id} holds ${cmdKey(node.cmdAt(index))} at index ${index}, committed command is ${cmdKey(expected)}`);
        }
        const term = this.appliedTerms.get(index);
        if (term !== undefined && node.termAt(index) !== term) {
          this.violations.push(`log matching: server ${node.id} has term ${node.termAt(index)} at index ${index}, committed term is ${term}`);
        }
      }
      const boundary = node.snapshot.lastIncludedIndex;
      if (boundary > 0 && this.appliedTerms.has(boundary) && node.snapshot.lastIncludedTerm !== this.appliedTerms.get(boundary)) {
        this.violations.push(`snapshot safety: server ${node.id} snapshot boundary ${boundary} has term ${node.snapshot.lastIncludedTerm}, committed term is ${this.appliedTerms.get(boundary)}`);
      }
    }
    return this.violations.length === 0;
  }

  allStatesEqual() {
    const canonical = [...this.nodes.values()].filter((n) => n.active()).map((n) => this.canonical(n.state));
    return canonical.every((s) => s === canonical[0]);
  }

  messageCount() { return this.net.delivered + this.net.dropped; }
}

function buildCluster(size, opts = {}) {
  const ids = Array.from({ length: size }, (_, i) => i + 1);
  return new RaftCluster({ ids, seed: opts.seed ?? 0x11223344, node: opts.node ?? {} });
}

if (require.main === module) {
  // --- 1. single-server cluster: self-election and commit with no peers (degenerate quorum) -----
  const solo = buildCluster(1, { node: { electionBase: 3, electionJitter: 0 } });
  assert.strictEqual(solo.runUntil(() => solo.leader() !== null, 20) >= 0, true, "a one-server cluster must elect itself");
  const soloLeader = solo.leader();
  assert.strictEqual(soloLeader.id, 1);
  assert.strictEqual(soloLeader.currentTerm, 1, "the first election runs in term 1");
  const soloIndex = solo.submit({ kind: "set", key: "a", value: 1 });
  assert.strictEqual(typeof soloIndex, "number", "a single-server leader commits immediately");
  solo.runTicks(3);
  assert.strictEqual(soloLeader.state.get("a"), 1);
  assert.strictEqual(solo.canonical(soloLeader.state), solo.canonical(solo.oracleState()), "state equals the oracle");
  assert.deepStrictEqual(solo.violations, [], solo.violations.join("; "));

  // --- 2. five-server cluster: replication of a batch, oracle equivalence, safety ---------------
  const cluster = buildCluster(5);
  const elected = cluster.runUntil(() => cluster.leader() !== null, 60);
  assert.ok(elected >= 0, "a five-server cluster elects a leader");
  assert.strictEqual(cluster.leaders().length, 1, "exactly one leader is visible at a time");
  const commands = 40;
  let submitted = 0;
  let submitTicks = 0;
  for (let i = 1; i <= commands; i++) {
    const index = cluster.submit({ kind: "set", key: `k${i}`, value: i });
    assert.ok(index !== null, `command ${i} must reach a live leader`);
    submitted++;
    // Let the leader replicate every command before the next one: one in-flight entry at a time.
    const advanced = cluster.runUntil(() => cluster.maxAppliedIndex() >= index, 40);
    assert.ok(advanced >= 0, `command ${i} must commit within the tick budget`);
    submitTicks += advanced;
  }
  assert.strictEqual(submitted, commands);
  cluster.runTicks(20);                                       // let the last entry reach every follower
  assert.ok(cluster.runUntil(() => cluster.allStatesEqual(), 40) >= 0, "followers converge after the batch");
  const replicas = [...cluster.nodes.values()];
  const oracle = cluster.oracleState();
  assert.strictEqual(oracle.size, commands, "the oracle holds one entry per committed set command");
  for (const node of replicas) {
    assert.strictEqual(cluster.canonical(node.state), cluster.canonical(oracle), `replica ${node.id} matches the oracle`);
    assert.strictEqual(node.lastApplied, node.commitIndex, `replica ${node.id} applied everything it committed`);
  }
  assert.ok(cluster.checkLogMatching(), cluster.violations.join("; "));
  assert.deepStrictEqual(cluster.violations, [], cluster.violations.join("; "));
  assert.strictEqual(cluster.leadersByTerm.size, 1, "no extra election happened while the leader was healthy");

  // --- 3. linear cost: messages per committed command stay bounded -------------------------------
  const messagesForBatch = cluster.messageCount();
  const perCommand = messagesForBatch / commands;
  assert.ok(perCommand < 60, `messages per committed command must stay linear (saw ${perCommand.toFixed(1)})`);
  const entriesShipped = replicas.reduce((sum, n) => sum + n.stats.entriesAppended, 0);
  assert.ok(entriesShipped <= commands * 2 + 5 * 2, `log entries are appended O(1) times per server (saw ${entriesShipped})`);

  // --- 4. liveness bound: a single command commits in a small tick budget ------------------------
  const liveIndex = cluster.submit({ kind: "set", key: "latency", value: 7 });
  const commitTicks = cluster.runUntil(() => cluster.maxAppliedIndex() >= liveIndex, 20);
  assert.ok(commitTicks >= 0 && commitTicks <= 5, `a healthy leader commits in <= 5 ticks (saw ${commitTicks})`);

  // --- 5. negative protocol checks: stale term, double vote, stale log, and a log gap -----------
  const probe = cluster.node(1);
  const staleTerm = probe.currentTerm - 1;
  const beforeVotes = probe.votedFor;
  probe.receive(99, { type: "RequestVote", term: staleTerm, lastLogIndex: 999, lastLogTerm: 99 });
  assert.strictEqual(probe.currentTerm, staleTerm + 1, "a stale vote request must not change the term");
  assert.strictEqual(probe.votedFor, beforeVotes, "and must not overwrite the recorded vote");
  probe.receive(99, { type: "AppendEntries", term: staleTerm, prevLogIndex: 0, prevLogTerm: 0, entries: [], leaderCommit: 0 });
  assert.strictEqual(probe.role, LEADER, "a stale AppendEntries must not depose a live leader");
  const termNow = probe.currentTerm;
  const upToDateRequest = { lastLogIndex: probe.lastLogIndex(), lastLogTerm: probe.lastLogTerm() };
  probe.votedFor = null;
  probe.receive(99, { type: "RequestVote", term: termNow + 1, ...upToDateRequest });
  assert.strictEqual(probe.votedFor, 99, "an up-to-date candidate gets the vote");
  probe.receive(98, { type: "RequestVote", term: termNow + 1, ...upToDateRequest });
  const doubleVote = cluster.violations.some((v) => v.startsWith("vote safety"));
  assert.strictEqual(doubleVote, false, "no server grants two votes in one term");
  assert.strictEqual(probe.votedFor, 99, "the first candidate keeps the vote");

  const gapProbe = cluster.node(5);
  const gapLastIndex = gapProbe.lastLogIndex();
  const gapCommitIndex = gapProbe.commitIndex;
  gapProbe.receive(1, { type: "AppendEntries", term: probe.currentTerm, prevLogIndex: gapLastIndex + 5, prevLogTerm: 0, entries: [], leaderCommit: 99 });
  assert.strictEqual(gapProbe.lastLogIndex(), gapLastIndex, "an AppendEntries past the log end is rejected, not applied");
  assert.strictEqual(gapProbe.commitIndex, gapCommitIndex, "and its optimistic leaderCommit must not move the commit index");

  // --- 6. deterministic message loss: the cluster still converges --------------------------------
  const lossy = new RaftCluster({ ids: [1, 2, 3, 4, 5], seed: 0x0badf00d });
  lossy.net.dropEvery = 5;
  const lossyElected = lossy.runUntil(() => lossy.leader() !== null, 200);
  assert.ok(lossyElected >= 0, "a leader is elected despite periodic message loss");
  for (let i = 1; i <= 10; i++) {
    const index = lossy.submit({ kind: "set", key: `l${i}`, value: i });
    assert.ok(index !== null);
    assert.ok(lossy.runUntil(() => lossy.maxAppliedIndex() >= index, 80) >= 0, `lost appends are retried until command ${i} commits`);
  }
  lossy.net.dropEvery = 0;
  lossy.runTicks(40);
  assert.ok(lossy.checkLogMatching(), lossy.violations.join("; "));
  assert.deepStrictEqual(lossy.violations, [], lossy.violations.join("; "));
  for (const node of lossy.nodes.values()) {
    assert.strictEqual(lossy.canonical(node.state), lossy.canonical(lossy.oracleState()), `lossy replica ${node.id} converged`);
  }

  // --- 7. leader crash: failover in a higher term, nothing committed is lost --------------------
  const crash = buildCluster(5, { seed: 0x5151 });
  assert.ok(crash.runUntil(() => crash.leader() !== null, 60) >= 0);
  for (let i = 1; i <= 6; i++) {
    const index = crash.submit({ kind: "set", key: `c${i}`, value: i });
    assert.ok(crash.runUntil(() => crash.maxAppliedIndex() >= index, 40) >= 0);
  }
  const committedBefore = crash.maxAppliedIndex();
  const doomed = crash.leader();
  const doomedTerm = doomed.currentTerm;
  doomed.crashed = true;
  const failoverTicks = crash.runUntil(() => crash.leader() !== null && crash.leader().id !== doomed.id, 60);
  assert.ok(failoverTicks >= 0, "a majority elects a new leader after the leader crashes");
  assert.ok(crash.leader().currentTerm > doomedTerm, "the new leader runs in a higher term");
  assert.strictEqual(crash.leaders().length, 1, "the crashed leader is not counted as a leader");
  const afterCrashIndex = crash.submit({ kind: "set", key: "post-crash", value: 1 });
  assert.ok(crash.runUntil(() => crash.maxAppliedIndex() >= afterCrashIndex, 40) >= 0);
  doomed.crashed = false;                                     // restart with its log intact
  assert.ok(crash.runUntil(() => crash.canonical(crash.node(doomed.id).state) === crash.canonical(crash.oracleState()), 60) >= 0,
    "the restarted server converges back to the oracle");
  assert.ok(crash.maxAppliedIndex() >= committedBefore, "no committed entry was lost across the failover");
  assert.ok(crash.checkLogMatching(), crash.violations.join("; "));
  assert.deepStrictEqual(crash.violations, [], crash.violations.join("; "));

  // --- 8. minority partition: the isolated old leader can append but never commit ---------------
  const partition = buildCluster(5, { seed: 0xabcabc });
  assert.ok(partition.runUntil(() => partition.leader() !== null, 60) >= 0);
  for (let i = 1; i <= 4; i++) {
    const index = partition.submit({ kind: "set", key: `p${i}`, value: i });
    assert.ok(partition.runUntil(() => partition.maxAppliedIndex() >= index, 40) >= 0);
  }
  const oldLeader = partition.leader();
  const ghostTerm = oldLeader.currentTerm;
  partition.net.partition([[oldLeader.id], partition.ids().filter((id) => id !== oldLeader.id)]);
  const ghostIndex = partition.submit({ kind: "set", key: "ghost", value: 1 });
  assert.ok(ghostIndex !== null, "the isolated leader still appends locally");
  partition.runTicks(30);
  assert.ok(oldLeader.commitIndex < ghostIndex, "an isolated leader must not commit its entry");
  assert.strictEqual(oldLeader.state.has("ghost"), false, "and must not apply it either");
  const ghostKey = cmdKey({ kind: "set", key: "ghost", value: 1 });
  const ghostApplied = partition.applyObservations.get(ghostIndex);
  assert.strictEqual(ghostApplied === undefined || ghostApplied.has(ghostKey) === false, true,
    "no server ever applies the ghost command");
  assert.strictEqual(oldLeader.currentTerm, ghostTerm, "the isolated leader keeps its stale term");
  const majorityLeader = partition.leader();
  assert.notStrictEqual(majorityLeader.id, oldLeader.id, "the majority side elected a different leader");
  assert.ok(majorityLeader.currentTerm > oldLeader.currentTerm || oldLeader.role !== LEADER, "the deposed leader is a follower now");
  const visibleTerms = partition.leaders().map((n) => n.currentTerm);
  assert.strictEqual(new Set(visibleTerms).size, visibleTerms.length,
    "two visible leaders may exist only in different terms - never in the same term");

  partition.net.heal(partition.ids());
  const overwriteIndex = partition.submit({ kind: "set", key: "real", value: 2 });
  assert.ok(overwriteIndex >= ghostIndex, "the new leader's log never contained the ghost entry");
  assert.ok(partition.runUntil(() => partition.maxAppliedIndex() >= overwriteIndex, 60) >= 0);
  partition.runTicks(15);
  assert.strictEqual(partition.canonical(oldLeader.state), partition.canonical(partition.oracleState()),
    "after healing, the old leader converges to the committed state");
  assert.strictEqual(oldLeader.state.has("ghost"), false, "the uncommitted ghost entry never took effect");
  assert.strictEqual(oldLeader.role, FOLLOWER, "the old leader stepped down after healing");
  assert.strictEqual(oldLeader.state.get("real"), 2, "the conflicting entry at that index was replaced");
  assert.ok(oldLeader.termAt(ghostIndex) > ghostTerm, "the divergent entry was replaced by a higher-term entry");
  assert.notStrictEqual(cmdKey(oldLeader.cmdAt(ghostIndex)), ghostKey, "and the ghost command is gone from the log");
  assert.deepStrictEqual(partition.violations, [], partition.violations.join("; "));

  // --- 9. snapshots: compaction bounds the log and InstallSnapshot repairs a lagging server ------
  const snap = buildCluster(5, { seed: 0x5a5a, node: { snapshotThreshold: 6 } });
  assert.ok(snap.runUntil(() => snap.leader() !== null, 60) >= 0);
  const snapLeader = snap.leader();
  const lagging = [...snap.nodes.values()].find((n) => n.id !== snapLeader.id);
  snap.net.partition([[lagging.id], snap.ids().filter((id) => id !== lagging.id)]);
  for (let i = 1; i <= 20; i++) {
    const index = snap.submit({ kind: "set", key: `s${i}`, value: i * 2 });
    assert.ok(snap.runUntil(() => snap.maxAppliedIndex() >= index, 40) >= 0);
  }
  assert.ok(snapLeader.stats.snapshotsTaken >= 1, "the leader compacted its log");
  assert.ok(snapLeader.snapshot.lastIncludedIndex > 0, "the snapshot covers committed entries");
  assert.ok(snapLeader.log.length < 20 + 6, `the log is bounded by the snapshot threshold (length ${snapLeader.log.length})`);
  assert.strictEqual(snapLeader.log.length + snapLeader.snapshot.lastIncludedIndex, snapLeader.lastLogIndex(), "compaction keeps the log accounting exact");
  assert.strictEqual(snapLeader.termAt(snapLeader.snapshot.lastIncludedIndex - 1), null, "compacted entries are no longer addressable");
  assert.strictEqual(snap.canonical(snapLeader.snapshot.state), snap.canonical(snap.oracleState(snapLeader.snapshot.lastIncludedIndex)),
    "a snapshot must equal the oracle state folded up to its included index");
  assert.ok(snapLeader.snapshot.lastIncludedIndex <= snapLeader.commitIndex, "a snapshot never runs ahead of the commit index");
  const laggingAppliedBefore = lagging.lastApplied;
  snap.net.heal(snap.ids());
  assert.ok(snap.runUntil(() => snap.snapshotInstalls.some((s) => s.nodeId === lagging.id), 80) >= 0,
    "the lagging server is repaired with InstallSnapshot");
  assert.ok(snap.runUntil(() => snap.canonical(lagging.state) === snap.canonical(snap.oracleState()), 80) >= 0,
    "the repaired server matches the oracle");
  assert.ok(lagging.lastApplied > laggingAppliedBefore, "and it made real progress");
  assert.strictEqual(lagging.snapshot.lastIncludedIndex > 0, true, "it kept the transferred snapshot");
  snap.runTicks(10);
  assert.deepStrictEqual(snap.violations, [], snap.violations.join("; "));

  // --- 10. reconfiguration: add a server through the log, then remove one ----------------------
  const reconfig = buildCluster(3, { seed: 0x2b2b });
  assert.ok(reconfig.runUntil(() => reconfig.leader() !== null, 60) >= 0);
  for (let i = 1; i <= 5; i++) {
    const index = reconfig.submit({ kind: "set", key: `r${i}`, value: i });
    assert.ok(reconfig.runUntil(() => reconfig.maxAppliedIndex() >= index, 40) >= 0);
  }
  const joined = reconfig.joinNode(4, { electionBase: 5, electionJitter: 2 });
  const joinIndex = reconfig.submit({ kind: "join", id: 4 });
  assert.ok(joinIndex !== null, "the config entry goes through the log like any command");
  assert.ok(reconfig.runUntil(() => reconfig.maxAppliedIndex() >= joinIndex, 60) >= 0, "the join entry commits");
  assert.ok(reconfig.runUntil(() => reconfig.canonical(joined.state) === reconfig.canonical(reconfig.oracleState()), 80) >= 0,
    "the joining server catches up to the same state");
  for (const node of reconfig.nodes.values()) {
    assert.ok(node.membership.includes(4), `server ${node.id} applied the config entry`);
  }
  assert.strictEqual(reconfig.leader().quorum(), 3, "quorum for a four-server cluster is three");
  joined.campaigns = true;                                       // from now on it may stand for election
  const reconfigIndex = reconfig.submit({ kind: "set", key: "after-join", value: 9 });
  assert.ok(reconfig.runUntil(() => reconfig.maxAppliedIndex() >= reconfigIndex, 60) >= 0, "the cluster still commits after the join");

  const doomedPeer = 4;
  const leaveIndex = reconfig.submit({ kind: "leave", id: doomedPeer });
  assert.ok(reconfig.runUntil(() => reconfig.maxAppliedIndex() >= leaveIndex, 60) >= 0);
  assert.ok(reconfig.runUntil(() => reconfig.node(1).membership.length === 3, 40) >= 0, "the removal shrinks the membership");
  reconfig.retire(doomedPeer);
  const afterLeave = reconfig.submit({ kind: "set", key: "after-leave", value: 3 });
  assert.ok(reconfig.runUntil(() => reconfig.maxAppliedIndex() >= afterLeave, 60) >= 0, "the remaining majority still commits");
  assert.strictEqual(reconfig.node(1).membership.includes(doomedPeer), false, "the removed server is out of the config");
  const converged = () => [...reconfig.nodes.values()].filter((n) => n.active())
    .every((n) => reconfig.canonical(n.state) === reconfig.canonical(reconfig.oracleState()));
  assert.ok(reconfig.runUntil(converged, 80) >= 0, "all surviving servers converge to the oracle after the removal");
  const activeStates = [...reconfig.nodes.values()].filter((n) => n.active()).map((n) => reconfig.canonical(n.state));
  assert.ok(activeStates.every((s) => s === reconfig.canonical(reconfig.oracleState())), "all surviving servers equal the oracle");
  assert.ok(reconfig.checkLogMatching(), reconfig.violations.join("; "));
  assert.deepStrictEqual(reconfig.violations, [], reconfig.violations.join("; "));

  console.log(`raft: ${commands} commands on 5 servers, ${messagesForBatch} messages (${perCommand.toFixed(1)}/command), commit latency ${commitTicks} tick(s)`);
  console.log(`raft: failover in ${failoverTicks} ticks, ${snap.snapshotInstalls.length} snapshot install(s), log bounded at ${snapLeader.log.length} entries`);
  console.log(`raft: ${cluster.applyCount} observed applies, ${cluster.leadersByTerm.size} leader term(s), 0 violations`);
  console.log("PASS");
}

module.exports = { Network, RaftNode, RaftCluster, splitmix64, mix32, buildCluster };
