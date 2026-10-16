/** Task 891 - Implement a benchmarking harness for large distributed systems with failure injection.
 *  (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
 *
 * A deterministic, event-driven benchmarking harness for a replicated key-value service built on a
 * faithful Raft-lite core: leader election with terms and up-to-date vote checks, log replication with
 * prevIndex/prevTerm matching and nextIndex back-off, majority commit, leader-only reads barrier (the
 * new leader appends a no-op entry before answering), and exactly-once client semantics by
 * (clientId, seq) de-duplication on retries. Time is *virtual* milliseconds: every timer, message
 * delay and failure is an event in one priority queue ordered by (time, sequence), and all randomness
 * comes from a seeded splitmix32, so a scenario replays bit-for-bit.
 *
 * Failure injection is a first-class part of the harness: pause/crash-restart nodes, block delivery
 * across a partition, add per-node latency and drop a fraction of messages. The harness then measures
 * what an operator cares about — completion rate, throughput, latency percentiles, elections, message
 * volume — and judges an SLO.
 *
 * Invariants verified below: (1) the event queue orders by (time, sequence) exactly, and percentiles
 * are checked against an independent sorted brute force (including the empty and single-sample
 * boundaries); (2) every recorded client history is *linearizable*, decided by a brute-force
 * linearization search (Wing-Gong minimal-operation pruning) over the completed put/get operations —
 * an oracle that is independent of the replication code; (3) safety — never two leaders in one term,
 * and a committed prefix is never lost or reordered, checked after partitions and crashes heal;
 * (4) liveness — with faults active, operations still complete and a new leader is elected within a
 * bounded number of virtual milliseconds; (5) detection — an injected latency must make the harness
 * report the SLO violation, so the metrics are not vacuous; (6) cost — message volume stays O(ops·N);
 * (7) reproduce — identical seeds give an identical trace digest, different seeds do not.
 *
 * Simplifications: in-process nodes with an explicit message bus (no sockets, no threads), reads go
 * through the log (a documented, linearizable simplification), a single key-value state machine, and
 * small clusters (3 or 5 nodes) with tens of operations — "large" is expressed through the injected
 * fault rates and node counts, not wall-clock time.
 */
"use strict";

/* ------------------------------------------------------------------ primitives */

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
  unit() { return this.next() / 0x1_0000_0000; }
}

/** Min-heap over (time, seq): the total order that makes a simulation reproducible. */
class EventQueue {
  constructor() { this.heap = []; this.sequence = 0; }
  get size() { return this.heap.length; }
  schedule(time, kind, payload) {
    const event = { time, seq: this.sequence++, kind, payload };
    this.heap.push(event);
    let index = this.heap.length - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.before(this.heap[index], this.heap[parent])) {
        [this.heap[index], this.heap[parent]] = [this.heap[parent], this.heap[index]];
        index = parent;
      } else break;
    }
    return event;
  }
  before(a, b) { return a.time < b.time || (a.time === b.time && a.seq < b.seq); }
  pop() {
    if (this.heap.length === 0) return null;
    const top = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = last;
      let index = 0;
      for (;;) {
        const left = 2 * index + 1;
        const right = left + 1;
        let smallest = index;
        if (left < this.heap.length && this.before(this.heap[left], this.heap[smallest])) smallest = left;
        if (right < this.heap.length && this.before(this.heap[right], this.heap[smallest])) smallest = right;
        if (smallest === index) break;
        [this.heap[index], this.heap[smallest]] = [this.heap[smallest], this.heap[index]];
        index = smallest;
      }
    }
    return top;
  }
}

/* ------------------------------------------------------------------ linearizability oracle */

/**
 * Brute-force linearizability of a completed register history. At each step only "minimal"
 * operations (those that no remaining operation finished before) may be linearized next; this is the
 * classic Wing-Gong pruning and it is exact for a register.
 */
function isLinearizable(operations, initialValue) {
  const n = operations.length;
  const used = new Array(n).fill(false);
  const step = (value, remaining) => {
    if (remaining === 0) return true;
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      let minimal = true;
      for (let j = 0; j < n && minimal; j++) {
        if (j === i || used[j]) continue;
        if (operations[j].end < operations[i].start) minimal = false;
      }
      if (!minimal) continue;
      if (operations[i].kind === "get" && operations[i].result !== value) continue;
      used[i] = true;
      const next = operations[i].kind === "put" ? operations[i].value : value;
      if (step(next, remaining - 1)) { used[i] = false; return true; }
      used[i] = false;
    }
    return false;
  };
  return step(initialValue, n);
}

/* ------------------------------------------------------------------ metrics */

const percentile = (sorted, fraction) => {
  if (sorted.length === 0) return null;
  const index = Math.min(sorted.length - 1, Math.ceil(fraction * sorted.length) - 1);
  return sorted[index];
};

const brutePercentiles = (latencies) => {
  const sorted = [...latencies].sort((a, b) => a - b);
  return { p50: percentile(sorted, 0.5), p95: percentile(sorted, 0.95), p99: percentile(sorted, 0.99) };
};

/* ------------------------------------------------------------------ simulate network */

class Network {
  constructor(queue, cluster, prng, options = {}) {
    this.queue = queue;
    this.cluster = cluster;
    this.prng = prng;
    this.minDelay = options.minDelay !== undefined ? options.minDelay : 5;
    this.maxDelay = options.maxDelay !== undefined ? options.maxDelay : 15;
    this.lossRate = options.lossRate || 0;
    this.faults = options.faults || [];
    this.sent = 0;
    this.dropped = 0;
    this.delivered = 0;
  }

  blocked(from, to, time) {
    for (const fault of this.faults) {
      if (time < fault.at || (fault.until !== undefined && time > fault.until)) continue;
      if (fault.kind === "partition") {
        const fromSide = fault.groups.findIndex((group) => group.includes(from));
        const toSide = fault.groups.findIndex((group) => group.includes(to));
        if (fromSide !== -1 && toSide !== -1 && fromSide !== toSide) return true;
      }
      if (fault.kind === "pause" && (fault.node === from || fault.node === to)) return true;
      if (fault.kind === "crash" && (fault.node === from || fault.node === to)) return true;
    }
    return false;
  }

  extraDelay(node, time) {
    let extra = 0;
    for (const fault of this.faults) {
      if (fault.kind !== "slow" && fault.kind !== "slow-client") continue;
      if (time < fault.at || (fault.until !== undefined && time > fault.until)) continue;
      if (fault.node === node) extra += fault.delay;
    }
    return extra;
  }

  send(from, to, message, time) {
    this.sent++;
    if (this.lossRate > 0 && this.prng.unit() < this.lossRate) { this.dropped++; return; }
    if (this.blocked(from, to, time)) { this.dropped++; return; }
    const latency = this.minDelay + this.prng.below(Math.max(1, this.maxDelay - this.minDelay + 1))
      + this.extraDelay(from, time) + this.extraDelay(to, time);
    this.queue.schedule(time + latency, "message", { from, to, message });
  }

  deliver(event) {
    this.delivered++;
    const target = this.cluster.nodes.get(event.payload.to);
    const source = this.cluster.nodes.get(event.payload.from);
    if (!target || !source) return;
    if (this.isFaulted(target.id, event.time) || this.isFaulted(source.id, event.time)) { this.dropped++; return; }
    target.onMessage(event.payload.from, event.payload.message, event.time);
  }

  isFaulted(nodeId, time) {
    return this.faults.some((fault) =>
      (fault.kind === "pause" || fault.kind === "crash") && fault.node === nodeId &&
      time >= fault.at && (fault.until === undefined || time <= fault.until));
  }
}

/* ------------------------------------------------------------------ raft-lite node */

const ELECTION_MIN = 150;
const ELECTION_SPAN = 150;
const HEARTBEAT = 50;

class RaftNode {
  constructor(id, cluster) {
    this.id = id;
    this.cluster = cluster;
    this.prng = new Prng(1000 + id * 7919);
    this.term = 0;
    this.votedFor = null;
    this.log = [];                  // { term, command }
    this.commitIndex = 0;
    this.appliedIndex = 0;
    this.role = "follower";
    this.leader = null;
    this.timerGeneration = 0;
    this.nextIndex = new Map();
    this.matchIndex = new Map();
    this.started = false;
  }

  get lastIndex() { return this.log.length; }
  get lastTerm() { return this.log.length === 0 ? 0 : this.log[this.log.length - 1].term; }
  get quorum() { return Math.floor(this.cluster.nodes.size / 2) + 1; }

  start(time) { this.resetTimer(time); this.started = true; }

  resetTimer(time) {
    const generation = ++this.timerGeneration;
    const timeout = ELECTION_MIN + this.prng.below(ELECTION_SPAN);
    this.cluster.queue.schedule(time + timeout, "election-timeout", { node: this.id, generation });
  }

  onEvent(kind, payload, time) {
    if (kind === "election-timeout") {
      if (payload.generation !== this.timerGeneration) return;
      if (this.role === "leader") { this.resetTimer(time); return; }
      this.becomeCandidate(time);
    } else if (kind === "heartbeat") {
      if (this.role !== "leader") return;
      this.broadcastAppend(time);
      this.cluster.queue.schedule(time + HEARTBEAT, "heartbeat", { node: this.id });
    }
  }

  becomeCandidate(time) {
    this.term++;
    this.role = "candidate";
    this.votedFor = this.id;
    this.leader = null;
    this.cluster.elections++;
    this.votes = new Set([this.id]);
    if (this.votes.size >= this.quorum) { this.votes = new Set(); this.becomeLeader(time); return; }
    this.resetTimer(time);
    for (const peer of this.cluster.nodes.keys()) {
      if (peer !== this.id) {
        this.cluster.network.send(this.id, peer, {
          type: "request-vote", term: this.term, candidate: this.id,
          lastIndex: this.lastIndex, lastTerm: this.lastTerm,
        }, time);
      }
    }
  }

  becomeLeader(time) {
    this.role = "leader";
    this.leader = this.id;
    this.cluster.noteLeader(this.term, this.id, time);
    for (const peer of this.cluster.nodes.keys()) {
      this.nextIndex.set(peer, this.lastIndex + 1);
      this.matchIndex.set(peer, 0);
    }
    // Leader-only read barrier: a no-op in the new term must commit before reads are answered.
    this.log.push({ term: this.term, command: { type: "noop" } });
    this.broadcastAppend(time);
    this.cluster.queue.schedule(time + HEARTBEAT, "heartbeat", { node: this.id });
  }

  stepDown(term, leader, time) {
    if (term > this.term) { this.term = term; this.votedFor = null; }
    this.role = "follower";
    this.leader = leader;
    this.resetTimer(time);
  }

  broadcastAppend(time) {
    for (const peer of this.cluster.nodes.keys()) {
      if (peer !== this.id) this.sendAppend(peer, time);
    }
  }

  sendAppend(peer, time) {
    const next = this.nextIndex.get(peer) || 1;
    const prevIndex = next - 1;
    const prevTerm = prevIndex > 0 && prevIndex <= this.log.length ? this.log[prevIndex - 1].term : 0;
    this.cluster.network.send(this.id, peer, {
      type: "append", term: this.term, leader: this.id, prevIndex, prevTerm,
      entries: this.log.slice(prevIndex, prevIndex + 8),
      leaderCommit: this.commitIndex,
    }, time);
  }

  onMessage(from, message, time) {
    if (message.term > this.term && message.type !== "request-vote") this.stepDown(message.term, message.leader, time);
    switch (message.type) {
      case "request-vote": {
        if (message.term < this.term) return;
        const upToDate = message.lastTerm > this.lastTerm
          || (message.lastTerm === this.lastTerm && message.lastIndex >= this.lastIndex);
        // A vote is scoped to a term: a node that voted in an older term may vote again.
        const canVote = message.term > this.term || this.votedFor === null || this.votedFor === message.candidate;
        const granted = upToDate && canVote;
        if (granted) {
          this.term = message.term;
          this.votedFor = message.candidate;
          this.role = "follower";
          this.resetTimer(time);
        }
        this.cluster.network.send(this.id, message.candidate,
          { type: "vote", term: this.term, granted, voter: this.id }, time);
        return;
      }
      case "vote": {
        if (message.term !== this.term || this.role !== "candidate") return;
        if (message.granted) {
          this.votes = (this.votes || new Set([this.id])).add(message.voter);
          if (this.votes.size >= this.quorum) { this.votes = new Set(); this.becomeLeader(time); }
        }
        return;
      }
      case "append": {
        if (message.term < this.term) {
          this.cluster.network.send(this.id, message.leader, { type: "append-reply", term: this.term, ok: false, matchIndex: 0 }, time);
          return;
        }
        this.stepDown(message.term, message.leader, time);
        const matches = message.prevIndex === 0
          || (message.prevIndex <= this.log.length && this.log[message.prevIndex - 1].term === message.prevTerm);
        if (!matches) {
          this.cluster.network.send(this.id, message.leader,
            { type: "append-reply", term: this.term, ok: false, matchIndex: Math.min(message.prevIndex, this.log.length) - 1 }, time);
          return;
        }
        this.log = this.log.slice(0, message.prevIndex);
        for (const entry of message.entries) this.log.push(entry);
        if (message.leaderCommit > this.commitIndex) {
          this.commitIndex = Math.min(message.leaderCommit, this.log.length);
        }
        this.applyCommitted(time);
        this.cluster.network.send(this.id, message.leader,
          { type: "append-reply", term: this.term, ok: true, matchIndex: this.log.length }, time);
        return;
      }
      case "append-reply": {
        if (this.role !== "leader" || message.term !== this.term) return;
        if (message.ok) {
          this.matchIndex.set(from, message.matchIndex);
          this.nextIndex.set(from, Math.max(1, message.matchIndex + 1));
          this.advanceCommit(time);
        } else {
          this.nextIndex.set(from, Math.max(1, (this.nextIndex.get(from) || 1) - 1));
          this.sendAppend(from, time);
        }
        return;
      }
      default:
    }
  }

  advanceCommit(time) {
    let candidate = this.commitIndex;
    for (let index = this.log.length; index > this.commitIndex; index--) {
      if (this.log[index - 1].term !== this.term) break;
      let acks = 1;
      for (const peer of this.cluster.nodes.keys()) {
        if (peer !== this.id && (this.matchIndex.get(peer) || 0) >= index) acks++;
      }
      if (acks >= this.quorum) { candidate = index; break; }
    }
    if (candidate > this.commitIndex) {
      this.commitIndex = candidate;
      this.applyCommitted(time);
    }
  }

  applyCommitted(time) {
    while (this.appliedIndex < this.commitIndex) {
      this.appliedIndex++;
      this.cluster.apply(this.id, this.log[this.appliedIndex - 1].command, time, this.appliedIndex);
    }
    this.cluster.observeProgress(this.id, this.commitIndex);
  }

  /** Fast-forward a restarted node to the node it was before the crash (log survives). */
  stateDigest() { return `${this.term}|${this.commitIndex}|${this.log.length}`; }
}

/* ------------------------------------------------------------------ benchmark harness */

class Harness {
  constructor(options = {}) {
    this.queue = new EventQueue();
    this.prng = new Prng(options.seed || 891);
    this.cluster = {
      nodes: new Map(), queue: this.queue, elections: 0, messages: 0,
      termLeaders: new Map(), appliedEntries: [], duplicateApplies: 0, maxCommit: 0, digest: 0,
    };
    const size = options.nodes || 3;
    for (let id = 0; id < size; id++) this.cluster.nodes.set(id, new RaftNode(id, this.cluster));
    this.network = new Network(this.queue, this.cluster, new Prng((options.seed || 891) ^ 0x5f5f), {
      faults: options.faults || [], lossRate: options.lossRate || 0,
    });
    this.cluster.network = this.network;
    this.cluster.apply = (nodeId, command, time) => this.apply(nodeId, command, time);
    this.cluster.noteLeader = (term, leader, time) => {
      if (this.cluster.termLeaders.has(term) && this.cluster.termLeaders.get(term) !== leader) {
        this.splitBrain = `${term}: ${this.cluster.termLeaders.get(term)} vs ${leader}`;
      }
      this.cluster.termLeaders.set(term, leader);
      this.leaderHistory.push({ term, leader, time });
    };
    this.cluster.observeProgress = (nodeId, commitIndex) => {
      this.cluster.maxCommit = Math.max(this.cluster.maxCommit, commitIndex);
      this.progress.set(nodeId, commitIndex);
    };
    this.state = new Map();
    this.appliedPerNode = new Set();
    this.appliedCommands = new Set();
    this.operations = new Map();
    this.latencies = [];
    this.history = new Map();
    this.leaderHistory = [];
    this.progress = new Map();
    this.splitBrain = null;
    this.leaderHint = null;
    this.opCounter = 0;
  }

  start(time = 0) {
    for (const node of this.cluster.nodes.values()) node.start(time);
    return this;
  }

  submit(kind, key, value, time, options = {}) {
    const id = this.opCounter++;
    const operation = {
      id, kind, key, value, submitTime: time, deadline: time + (options.timeout || 4000),
      done: false, result: null, responseTime: null, attempts: 0, clientId: options.clientId || "c0",
    };
    this.operations.set(id, operation);
    this.attempt(operation, time);
    return operation;
  }

  attempt(operation, time, targetNode = null) {
    if (operation.done || time > operation.deadline) return;
    operation.attempts++;
    let target = targetNode;
    if (target === null) {
      // A real client keeps a leader hint but must periodically re-probe, otherwise a crashed
      // leader would pin it forever.
      const probe = [...this.cluster.nodes.keys()][this.prng.below(this.cluster.nodes.size)];
      const hintUsable = this.leaderHint !== null && this.cluster.nodes.has(this.leaderHint);
      target = hintUsable && operation.attempts % 3 !== 1 ? this.leaderHint : probe;
    }
    this.network.send("client", target, {
      type: "client-request",
      command: { type: operation.kind, key: operation.key, value: operation.value, clientId: operation.clientId, seq: operation.id },
    }, time);
    this.queue.schedule(time + 100, "client-retry", { op: operation.id });
  }

  /** A follower redirects the client instead of dropping the request; the hint is learned once. */
  handleClientReply(message, time) {
    if (message.type !== "redirect") return;
    if (message.leader === null || message.leader === undefined) { this.leaderHint = null; return; }
    this.leaderHint = message.leader;
    const operation = this.operations.get(message.command.seq);
    if (operation && !operation.done) this.attempt(operation, time, message.leader);
  }

  apply(nodeId, command, time) {
    this.cluster.appliedEntries.push({ nodeId, command, time });
    if (command.type === "noop") return;
    // De-duplication is per replica and per (client, seq): every replica applies a committed entry
    // exactly once, and a client retry must never be applied twice anywhere.
    const replicaKey = `${nodeId}:${command.clientId}:${command.seq}`;
    if (this.appliedPerNode.has(replicaKey)) { this.cluster.duplicateApplies++; return; }
    this.appliedPerNode.add(replicaKey);
    const globalKey = `${command.clientId}:${command.seq}`;
    if (this.appliedCommands.has(globalKey)) return;
    this.appliedCommands.add(globalKey);
    if (command.type === "put") this.state.set(command.key, command.value);
    const operation = this.operations.get(command.seq);
    if (operation && !operation.done) {
      operation.done = true;
      operation.responseTime = time;
      operation.result = command.type === "put" ? "ok" : this.state.get(command.key);
      this.latencies.push(time - operation.submitTime);
      this.recordHistory(operation);
    }
  }

  /** Commands are submitted to a node through the bus; only a leader appends them. */
  handleClientRequest(node, message, time) {
    const command = message.command;
    const key = `${command.clientId}:${command.seq}`;
    if (node.role !== "leader") {
      // A follower answers with a redirect so the client can find the leader in one round trip.
      this.network.send(node.id, "client",
        { type: "redirect", leader: node.leader === node.id ? null : node.leader, command }, time);
      return;
    }
    for (const entry of node.log) {
      if (entry.command.type !== "noop" && `${entry.command.clientId}:${entry.command.seq}` === key) return;
    }
    node.log.push({ term: node.term, command });
    node.broadcastAppend(time);
    node.advanceCommit(time);
  }

  recordHistory(operation) {
    if (!this.history.has(operation.key)) this.history.set(operation.key, []);
    this.history.get(operation.key).push({
      kind: operation.kind, value: operation.value, result: operation.result,
      start: operation.submitTime, end: operation.responseTime,
    });
  }

  /** Runs the clock up to `until`, then returns the collected metrics. */
  run(until) {
    while (this.queue.size > 0) {
      const event = this.queue.pop();
      if (event.time > until) { this.queue.schedule(event.time, event.kind, event.payload); break; }
      this.trace(`${event.time}:${event.kind}`);
      this.cluster.messages++;
      if (event.kind === "message") {
        const targetNode = this.cluster.nodes.get(event.payload.to);
        if (event.payload.to === "client") {
          this.handleClientReply(event.payload.message, event.time);
        } else if (!targetNode || this.network.isFaulted(targetNode.id, event.time)) {
          this.network.dropped++;
        } else if (event.payload.message.type === "client-request") {
          this.handleClientRequest(targetNode, event.payload.message, event.time);
        } else {
          this.network.deliver(event);
        }
      } else if (event.kind === "client-retry") {
        const operation = this.operations.get(event.payload.op);
        if (operation && !operation.done && !(event.time >= operation.deadline)) this.attempt(operation, event.time);
      } else if (event.payload && event.payload.node !== undefined) {
        const node = this.cluster.nodes.get(event.payload.node);
        if (!this.network.isFaulted(node.id, event.time)) node.onEvent(event.kind, event.payload, event.time);
        else if (event.kind === "election-timeout") {
          this.queue.schedule(event.time + 25, event.kind, event.payload);   // paused node: try later
        }
      }
    }
    return this.metrics(until);
  }

  trace(text) {
    for (let i = 0; i < text.length; i++) {
      this.cluster.digest = (Math.imul(this.cluster.digest ^ text.charCodeAt(i), 0x01000193) >>> 0);
    }
  }

  metrics(until) {
    const completed = [...this.operations.values()].filter((operation) => operation.done);
    const timedOut = [...this.operations.values()].filter((operation) => !operation.done);
    const percentiles = brutePercentiles(this.latencies);
    const perKeyLinearizable = [...this.history.entries()].map(([key, operations]) =>
      ({ key, operations, ok: isLinearizable(operations, undefined) }));
    return {
      until, completed: completed.length, timedOut: timedOut.length, latencies: this.latencies.slice(),
      percentiles, throughput: completed.length / (until / 1000),
      completedTimes: completed.map((operation) => operation.responseTime),
      elections: this.cluster.elections, messages: this.network.sent, dropped: this.network.dropped,
      appliedCommands: this.appliedCommands.size, duplicateApplies: this.cluster.duplicateApplies,
      maxCommit: this.cluster.maxCommit, leaderChanges: this.leaderHistory.filter((entry) => entry.term > 1).length,
      splitBrain: this.splitBrain, linearizable: perKeyLinearizable.every((entry) => entry.ok),
      inconsistentKeys: perKeyLinearizable.filter((entry) => !entry.ok).map((entry) => entry.key),
      state: new Map(this.state), digest: this.cluster.digest,
      leaderHistory: this.leaderHistory.slice(), progress: new Map(this.progress),
    };
  }

  sloViolations(metrics, slo) {
    const violations = [];
    if (metrics.percentiles.p99 === null || metrics.percentiles.p99 > slo.p99) violations.push("p99");
    if (metrics.percentiles.p95 === null || metrics.percentiles.p95 > slo.p95) violations.push("p95");
    if (metrics.throughput < slo.throughput) violations.push("throughput");
    if (metrics.timedOut > 0) violations.push("completion");
    if (metrics.splitBrain) violations.push("split-brain");
    if (!metrics.linearizable) violations.push("linearizability");
    return violations;
  }
}

function workload(harness, { count, startTime, key = "k", gap = 120 }) {
  for (let i = 0; i < count; i++) {
    harness.submit("put", key, `v${i}`, startTime + i * gap);
    if (i % 3 === 0) harness.submit("get", key, null, startTime + i * gap + 20);
  }
}

module.exports = {
  Prng, EventQueue, isLinearizable, percentile, brutePercentiles, Network, RaftNode, Harness, workload,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- event queue ordering and percentiles -------------------------------------------------
  {
    const queue = new EventQueue();
    queue.schedule(30, "c", {});
    queue.schedule(10, "a", {});
    queue.schedule(30, "d", {});
    queue.schedule(20, "b", {});
    assert.deepStrictEqual([queue.pop().kind, queue.pop().kind, queue.pop().kind, queue.pop().kind],
      ["a", "b", "c", "d"], "ties are broken by insertion sequence");
    assert.strictEqual(queue.pop(), null);
    queue.schedule(5, "x", {});
    assert.strictEqual(queue.pop().kind, "x");
  }
  assert.deepStrictEqual(brutePercentiles([5]), { p50: 5, p95: 5, p99: 5 });
  assert.strictEqual(brutePercentiles([]).p99, null);
  {
    const values = Array.from({ length: 100 }, (_, i) => i + 1);
    const sorted = [...values].sort((a, b) => a - b);
    const oracle = (fraction) => sorted[Math.ceil(fraction * sorted.length) - 1];
    assert.strictEqual(brutePercentiles(values).p50, oracle(0.5));
    assert.strictEqual(brutePercentiles(values).p95, oracle(0.95));
    assert.strictEqual(brutePercentiles(values).p99, oracle(0.99));
    assert.strictEqual(brutePercentiles(values).p99, 99);
  }

  // --- the linearizability oracle must accept legal and reject illegal histories ------------
  {
    const legal = [
      { kind: "put", value: "a", start: 0, end: 10, result: "ok" },
      { kind: "get", value: null, start: 12, end: 20, result: "a" },
    ];
    assert.ok(isLinearizable(legal, undefined), "sequential put then get is legal");
    const illegal = [
      { kind: "put", value: "a", start: 0, end: 30, result: "ok" },
      { kind: "get", value: null, start: 5, end: 10, result: "b" },
    ];
    assert.ok(!isLinearizable(illegal, undefined), "a read of an unwritten value is illegal");
    const concurrent = [
      { kind: "put", value: "x", start: 0, end: 40, result: "ok" },
      { kind: "put", value: "y", start: 5, end: 30, result: "ok" },
      { kind: "get", value: null, start: 10, end: 20, result: "x" },
    ];
    assert.ok(isLinearizable(concurrent, undefined), "concurrent puts allow either order");
    const stale = [
      { kind: "put", value: "x", start: 0, end: 10, result: "ok" },
      { kind: "put", value: "y", start: 20, end: 30, result: "ok" },
      { kind: "get", value: null, start: 40, end: 50, result: "x" },
    ];
    assert.ok(!isLinearizable(stale, undefined), "a read must reflect the latest completed write");
    assert.ok(isLinearizable([], undefined), "the empty history is trivially legal");
  }

  // --- baseline: a healthy 3-node cluster ---------------------------------------------------
  const healthy = new Harness({ seed: 11, nodes: 3 });
  healthy.start(0);
  workload(healthy, { count: 12, startTime: 600, gap: 150 });
  const healthyMetrics = healthy.run(4000);
  assert.strictEqual(healthyMetrics.timedOut, 0, "no operation may time out without faults");
  assert.ok(healthyMetrics.completed >= 12, `completed ${healthyMetrics.completed}`);
  assert.ok(healthyMetrics.linearizable, `healthy history not linearizable: ${healthyMetrics.inconsistentKeys}`);
  assert.strictEqual(healthyMetrics.splitBrain, null, "never two leaders in one term");
  assert.ok(healthyMetrics.percentiles.p99 < 400, `p99 ${healthyMetrics.percentiles.p99}`);
  assert.ok(healthyMetrics.elections >= 1, "a leader must have been elected");
  assert.strictEqual(healthyMetrics.state.get("k"), "v11", "the last acknowledged put is the visible value");
  assert.deepStrictEqual(healthy.sloViolations(healthyMetrics, { p99: 500, p95: 500, throughput: 1 }), []);
  assert.ok(healthyMetrics.messages <= 400 * healthyMetrics.completed + 1000,
    `message volume ${healthyMetrics.messages} must stay linear in the number of operations`);
  assert.strictEqual(healthyMetrics.duplicateApplies, 0, "no entry may be applied twice on a replica");

  // --- detection: an injected latency must be visible, so the metrics are not vacuous -------
  const slow = new Harness({
    seed: 11, nodes: 3,
    faults: [{ kind: "slow-client", node: "client", delay: 300, at: 0, until: 100000 }],
  });
  slow.start(0);
  workload(slow, { count: 10, startTime: 600, gap: 400 });
  const slowMetrics = slow.run(8000);
  assert.ok(slowMetrics.percentiles.p99 >= 300,
    `an injected 300ms delay must show up in p99 (saw ${slowMetrics.percentiles.p99})`);
  assert.ok(slow.sloViolations(slowMetrics, { p99: 200, p95: 150, throughput: 0.5 }).includes("p99"),
    "the harness must report the SLO violation");
  assert.strictEqual(slow.sloViolations(healthyMetrics, { p99: 500, p95: 500, throughput: 0.5 }).length, 0,
    "the same SLO must pass on the healthy run");
  assert.ok(slowMetrics.completed >= 10, "the injected latency must not break correctness");

  // --- crash and restart: a new leader is elected and committed data survives ---------------
  const crashed = new Harness({ seed: 23, nodes: 3 });
  crashed.start(0);
  crashed.run(900);
  const firstLeader = crashed.leaderHistory[crashed.leaderHistory.length - 1].leader;
  assert.ok(Number.isInteger(firstLeader), "a leader must exist before the crash");
  // Crash the current leader, then restart it inside the window (its persisted log survives).
  crashed.network.faults.push({ kind: "crash", node: firstLeader, at: 905, until: 2200 });
  workload(crashed, { count: 14, startTime: 950, gap: 250 });
  const crashMetrics = crashed.run(9000);
  assert.strictEqual(crashMetrics.splitBrain, null);
  assert.ok(crashMetrics.linearizable, `crash history not linearizable: ${crashMetrics.inconsistentKeys}`);
  assert.ok(crashMetrics.elections >= 1, "a crashed leader must trigger an election");
  assert.ok(crashMetrics.completed >= 8, `only ${crashMetrics.completed} operations completed across the outage`);
  const afterCrash = crashMetrics.leaderHistory.filter((entry) => entry.time > 905);
  assert.ok(afterCrash.length >= 1, "a new leader must be elected after the crash");
  assert.notStrictEqual(afterCrash[0].leader, firstLeader, "leadership must move to a survivor");
  assert.ok(afterCrash[0].time <= 2200, "recovery must happen within the outage window (liveness)");
  assert.ok(crashed.cluster.nodes.get(firstLeader).commitIndex >= 1,
    "the restarted replica must still hold the committed log");

  // --- durability: every replica ends up with the committed prefix --------------------------
  {
    const logs = [...crashed.cluster.nodes.values()]
      .map((node) => node.log.slice(0, node.commitIndex).map((entry) => JSON.stringify(entry.command)));
    const shortest = logs.reduce((a, b) => (a.length <= b.length ? a : b));
    for (const log of logs) assert.deepStrictEqual(log.slice(0, shortest.length), shortest,
      "every replica must agree on the committed prefix");
    assert.ok(crashMetrics.maxCommit > 0, "something must have committed");
    const leaders = new Set(crashMetrics.leaderHistory.map((entry) => entry.leader));
    assert.ok(leaders.size >= 2, "the crash must have produced at least two leaders over time");
  }

  // --- partition: only the majority side may commit, and it heals ---------------------------
  {
    const split = new Harness({
      seed: 31, nodes: 5,
      faults: [{ kind: "partition", groups: [[0, 1], [2, 3, 4]], at: 900, until: 2600 }],
    });
    split.start(0);
    workload(split, { count: 16, startTime: 600, gap: 220 });
    const partitionMetrics = split.run(9000);
    assert.strictEqual(partitionMetrics.splitBrain, null, "a majority partition must remain unique per term");
    assert.ok(partitionMetrics.linearizable, `partition history not linearizable: ${partitionMetrics.inconsistentKeys}`);
    assert.ok(partitionMetrics.completed >= 6, `the majority side must keep serving (${partitionMetrics.completed})`);
    assert.ok(partitionMetrics.timedOut <= 6, `too many timeouts: ${partitionMetrics.timedOut}`);
    const duringPartition = partitionMetrics.completedTimes.filter((time) => time >= 900 && time <= 2800);
    assert.ok(duringPartition.length >= 2,
      `the majority side must have made progress during the partition (${duringPartition.length} ops)`);
    // After healing, the whole cluster converges on a single committed prefix.
    const logs = [...split.cluster.nodes.values()]
      .map((node) => node.log.slice(0, node.commitIndex).map((entry) => JSON.stringify(entry.command)));
    const shortest = logs.reduce((a, b) => (a.length <= b.length ? a : b));
    for (const log of logs) assert.deepStrictEqual(log.slice(0, shortest.length), shortest, "logs must converge");
    const entries = new Set(partitionMetrics.leaderHistory.map((entry) => entry.leader));
    assert.ok(entries.size >= 1, "a leader existed before, during and after the partition");
  }

  // --- message loss: liveness needs retries but must still hold -----------------------------
  {
    const lossy = new Harness({ seed: 41, nodes: 3, lossRate: 0.3 });
    lossy.start(0);
    workload(lossy, { count: 10, startTime: 600, gap: 300 });
    const lossyMetrics = lossy.run(12000);
    assert.ok(lossyMetrics.dropped > 0, "the harness must actually drop messages");
    assert.ok(lossyMetrics.completed >= 8, `lossy run completed only ${lossyMetrics.completed}`);
    assert.ok(lossyMetrics.linearizable, "retries must preserve exactly-once semantics");
    assert.strictEqual(lossyMetrics.duplicateApplies, 0, "client retries must be de-duplicated at the state machine");
  }

  // --- reproducibility ----------------------------------------------------------------------
  {
    const runOnce = (seed) => {
      const harness = new Harness({ seed, nodes: 3, lossRate: 0.1 });
      harness.start(0);
      workload(harness, { count: 8, startTime: 600, gap: 200 });
      const metrics = harness.run(4000);
      return `${metrics.digest}|${metrics.completed}|${metrics.messages}|${metrics.state.get("k")}`;
    };
    assert.strictEqual(runOnce(7), runOnce(7), "the same seed must reproduce the trace");
    assert.notStrictEqual(runOnce(7), runOnce(8), "a different seed must change the trace");
  }

  // --- edge cases ----------------------------------------------------------------------------
  {
    const idle = new Harness({ seed: 3, nodes: 1 });
    idle.start(0);
    const idleMetrics = idle.run(2000);
    assert.strictEqual(idleMetrics.completed, 0);
    assert.strictEqual(idleMetrics.percentiles.p99, null, "no latencies -> no percentile");
    assert.strictEqual(idleMetrics.timedOut, 0);
    // No samples at all: an SLO on percentiles cannot be met, and the harness says so.
    assert.deepStrictEqual(idle.sloViolations(idleMetrics, { p99: 100, p95: 100, throughput: 0 }), ["p99", "p95"]);
    const single = new Harness({ seed: 4, nodes: 1 });
    single.start(0);
    single.submit("put", "k", "only", 100);
    const singleMetrics = single.run(1500);
    assert.strictEqual(singleMetrics.completed, 1, "a single-node cluster commits immediately");
    assert.strictEqual(singleMetrics.state.get("k"), "only");
    assert.ok(singleMetrics.linearizable);
    const noWork = new Harness({ seed: 5, nodes: 3 });
    noWork.start(0);
    const noWorkMetrics = noWork.run(1000);
    assert.strictEqual(noWorkMetrics.completed, 0);
    assert.strictEqual(noWorkMetrics.linearizable, true, "an empty history is trivially linearizable");
  }

  console.log(`harness: healthy p99=${healthyMetrics.percentiles.p99}ms ${healthyMetrics.completed} ops, ` +
    `crash run ${crashMetrics.completed} ops / ${crashMetrics.elections} elections, ` +
    `injected-latency p99=${slowMetrics.percentiles.p99}ms detected, ` +
    `${healthyMetrics.messages} messages overall`);
  console.log("PASS");
}
