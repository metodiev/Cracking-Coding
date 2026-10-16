/** Task 628 - Implement distributed consensus simulation with network partitions (Raft harness). (Tier: Expert, CTCI Beyond — Distributed Systems)
 *
 * A faithful Raft core. Leaders are elected by a majority, and only by a candidate whose log is at
 * least as up to date as the voter's; replication uses AppendEntries with the prevLogIndex/prevLogTerm
 * consistency check, a follower truncates a conflicting suffix, and the commit index advances only for
 * an entry of the leader's *current* term that a majority has matched (older entries are committed
 * indirectly - Raft's answer to Figure 8). Every message carries the sender's term, so a higher term
 * demotes the receiver to follower.
 *
 * Simplifications: a discrete-event simulation, not a network. `step()` is one tick: every node runs
 * its election/heartbeat logic, then the whole message queue is delivered with per-node FIFO and a
 * fixed node order, so runs are reproducible. A partition is a list of groups that can talk among
 * themselves; `lossEvery` drops every n-th message to exercise retries. Commands are opaque strings,
 * the state machine is an ordered list, and snapshots, client deduplication and membership changes are
 * out of scope (the next task covers joint-consensus reconfiguration).
 *
 * Invariants asserted after every tick: (1) Election Safety - at most one leader per term; (2) Log
 * Matching - two logs sharing an (index, term) pair agree on every earlier entry, so checking the
 * highest shared index suffices; (3) State Machine Safety - no two nodes apply different commands at
 * the same index and an applied entry never changes; (4) Leader Completeness - committed entries stay
 * present and unchanged in every log that caught up, and a sub-quorum partition commits nothing.
 */
"use strict";

const FOLLOWER = "follower", CANDIDATE = "candidate", LEADER = "leader";
const HEARTBEAT = 2, ELECTION_BASE = 6;

class Network {
  constructor(names, { lossEvery = 0 } = {}) {
    this.names = names; this.inbox = new Map(names.map((name) => [name, []]));
    this.lossEvery = lossEvery; this.groups = null; this.sent = 0; this.dropped = 0;
  }
  partition(groups) { this.groups = groups.map((group) => new Set(group)); }
  heal() { this.groups = null; }
  send(from, to, message) {
    this.sent++;
    const reachable = this.groups === null || this.groups.some((group) => group.has(from) && group.has(to));
    if (!reachable || (this.lossEvery > 0 && this.sent % this.lossEvery === 0)) { this.dropped++; return; }
    this.inbox.get(to).push({ from, message });
  }
  /** Deterministic delivery: fixed node order, per-node FIFO preserved. */
  drain() {
    const delivered = [];
    for (const name of this.names) {
      for (const envelope of this.inbox.get(name).splice(0)) delivered.push({ to: name, envelope });
    }
    return delivered;
  }
}

class RaftNode {
  constructor(name) {
    this.name = name;
    this.term = 0; this.votedFor = null; this.state = FOLLOWER; this.leader = null;
    this.log = []; // 1-based: entry i is log[i - 1]
    this.commitIndex = 0; this.applied = []; this.votes = new Set();
    this.nextIndex = new Map(); this.matchIndex = new Map();
  }
  lastLogIndex() { return this.log.length; }
  lastLogTerm() { return this.log.length === 0 ? 0 : this.log[this.log.length - 1].term; }
  commands() { return this.log.map((entry) => entry.command); }
}

class RaftSim {
  constructor(names, { lossEvery = 0 } = {}) {
    this.names = names;
    this.network = new Network(names, { lossEvery });
    this.nodes = new Map(names.map((name) => [name, new RaftNode(name)]));
    this.timers = new Map(names.map((name, index) => [name, ELECTION_BASE + index]));
    this.tick = 0; this.truncations = 0; this.elections = 0; this.violations = [];
    this.leadersByTerm = new Map();
    this.committed = new Map(); // index -> entry: the global record of decided entries
  }
  majority() { return Math.floor(this.names.length / 2) + 1; }
  peers(name) { return this.names.filter((other) => other !== name); }
  resetTimer(node) { this.timers.set(node.name, ELECTION_BASE + this.names.indexOf(node.name)); }
  leaders() { return [...this.nodes.values()].filter((node) => node.state === LEADER).map((node) => node.name); }
  leaderName() { const leaders = this.leaders(); return leaders.length === 1 ? leaders[0] : null; }

  /** Append a command to a leader: the unique one, or the named one in a partition scenario. */
  propose(command, leaderName = null) {
    const name = leaderName === null ? this.leaderName() : leaderName;
    const leader = this.nodes.get(name);
    if (leader === undefined || leader.state !== LEADER) throw new Error(`no leader: ${name}`);
    leader.log.push({ term: leader.term, command });
    return leader.log.length;
  }
  step() {
    this.tick++;
    for (const name of this.names) {
      const node = this.nodes.get(name);
      this.timers.set(name, this.timers.get(name) - 1);
      if (node.state === LEADER) {
        if (this.tick % HEARTBEAT === 0) this.broadcastAppend(node);
      } else if (this.timers.get(name) <= 0) {
        this.becomeCandidate(node);
      }
    }
    for (const { to, envelope } of this.network.drain()) this.deliver(this.nodes.get(to), envelope);
    this.checkInvariants();
  }
  run(ticks) {
    for (let i = 0; i < ticks; i++) this.step();
    return this;
  }
  becomeFollower(node, term, leader) {
    if (term > node.term) { node.term = term; node.votedFor = null; }
    node.state = FOLLOWER; node.leader = leader; node.votes.clear();
    this.resetTimer(node);
  }
  becomeCandidate(node) {
    node.term++; node.state = CANDIDATE; node.votedFor = node.name; node.leader = null;
    node.votes = new Set([node.name]);
    this.elections++;
    this.resetTimer(node);
    if (node.votes.size >= this.majority()) return this.becomeLeader(node);
    for (const peer of this.peers(node.name)) {
      this.network.send(node.name, peer, {
        type: "requestVote", term: node.term, candidate: node.name,
        lastLogIndex: node.lastLogIndex(), lastLogTerm: node.lastLogTerm(),
      });
    }
  }
  becomeLeader(node) {
    node.state = LEADER;
    node.leader = node.name;
    const previous = this.leadersByTerm.get(node.term);
    if (previous !== undefined && previous !== node.name) {
      this.violations.push(`two leaders in term ${node.term}: ${previous} and ${node.name}`);
    }
    this.leadersByTerm.set(node.term, node.name);
    for (const peer of this.peers(node.name)) {
      node.nextIndex.set(peer, node.log.length + 1);
      node.matchIndex.set(peer, 0);
    }
    this.broadcastAppend(node);
  }
  broadcastAppend(node) {
    for (const peer of this.peers(node.name)) {
      const next = node.nextIndex.get(peer) || 1;
      const prevIndex = next - 1;
      this.network.send(node.name, peer, {
        type: "append", term: node.term, leader: node.name, prevIndex,
        prevTerm: prevIndex > 0 ? node.log[prevIndex - 1].term : 0,
        entries: node.log.slice(prevIndex), leaderCommit: node.commitIndex,
      });
    }
    this.advanceLeaderCommit(node); // the leader counts itself, so a lone leader commits at once
  }
  /** A vote goes only to a log at least as up to date: higher last term, or same term, more entries. */
  upToDate(node, lastLogIndex, lastLogTerm) {
    if (lastLogTerm !== node.lastLogTerm()) return lastLogTerm > node.lastLogTerm();
    return lastLogIndex >= node.lastLogIndex();
  }
  deliver(node, envelope) {
    const message = envelope.message;
    const handlers = {
      requestVote: () => this.onRequestVote(node, message),
      voteReply: () => this.onVoteReply(node, envelope.from, message),
      append: () => this.onAppend(node, envelope.from, message),
      appendReply: () => this.onAppendReply(node, envelope.from, message),
    };
    if (handlers[message.type] === undefined) this.violations.push(`unknown message type ${message.type}`);
    else handlers[message.type]();
  }
  onRequestVote(node, message) {
    if (message.term > node.term) this.becomeFollower(node, message.term, null);
    const granted = message.term === node.term
      && (node.votedFor === null || node.votedFor === message.candidate)
      && this.upToDate(node, message.lastLogIndex, message.lastLogTerm);
    if (granted) { node.votedFor = message.candidate; this.resetTimer(node); }
    this.network.send(node.name, message.candidate, { type: "voteReply", term: node.term, granted });
  }
  onVoteReply(node, from, message) {
    if (message.term > node.term) return this.becomeFollower(node, message.term, null);
    if (node.state !== CANDIDATE || message.term !== node.term || !message.granted) return;
    node.votes.add(from);
    if (node.votes.size >= this.majority()) this.becomeLeader(node);
  }
  onAppend(node, from, message) {
    const reply = (success, matchIndex) =>
      this.network.send(node.name, from, { type: "appendReply", term: node.term, success, matchIndex });
    if (message.term < node.term) return reply(false, 0);
    if (message.term > node.term) this.becomeFollower(node, message.term, message.leader);
    node.state = FOLLOWER;
    node.leader = message.leader;
    this.resetTimer(node);
    const mismatch = message.prevIndex > node.log.length
      || (message.prevIndex > 0 && node.log[message.prevIndex - 1].term !== message.prevTerm);    if (mismatch) return reply(false, 0);
    let index = message.prevIndex;
    for (const entry of message.entries) {
      index++;
      if (index > node.log.length) node.log.push(entry);
      else if (node.log[index - 1].term !== entry.term) {
        node.log = node.log.slice(0, index - 1);
        node.log.push(entry);
        this.truncations++;
      }
    }
    if (message.leaderCommit > node.commitIndex) this.applyUpTo(node, Math.min(message.leaderCommit, node.log.length));
    reply(true, index);
  }
  onAppendReply(node, from, message) {
    if (message.term > node.term) return this.becomeFollower(node, message.term, null);
    if (node.state !== LEADER || message.term !== node.term) return;
    if (!message.success) {
      node.nextIndex.set(from, Math.max(1, (node.nextIndex.get(from) || 1) - 1));
      return;
    }
    node.nextIndex.set(from, message.matchIndex + 1);
    node.matchIndex.set(from, message.matchIndex);
    this.advanceLeaderCommit(node);
  }
  /** Only an entry of the leader's own term may be committed by counting replicas. */
  advanceLeaderCommit(node) {
    let candidate = 0;
    for (let index = node.log.length; index > node.commitIndex; index--) {
      if (node.log[index - 1].term === node.term) { candidate = index; break; }
    }
    if (candidate === 0) return;
    let replicated = 1;
    for (const peer of this.peers(node.name)) if ((node.matchIndex.get(peer) || 0) >= candidate) replicated++;
    if (replicated >= this.majority()) this.applyUpTo(node, candidate);
  }
  applyUpTo(node, index) {
    while (node.commitIndex < index) {
      const next = node.commitIndex + 1;
      const entry = node.log[next - 1];
      node.commitIndex = next;
      node.applied.push(entry.command);
      const known = this.committed.get(next);
      if (known === undefined) this.committed.set(next, entry);
      else if (known.term !== entry.term || known.command !== entry.command) {
        this.violations.push(`state machine safety at ${next}: ${known.command} vs ${entry.command}`);
      }
    }
  }
  checkInvariants() {
    const nodes = [...this.nodes.values()];
    for (const [index, entry] of this.committed) {
      for (const node of nodes) {
        if (node.commitIndex < index) continue;
        const actual = node.log[index - 1];
        if (actual === undefined || actual.term !== entry.term || actual.command !== entry.command) {
          this.violations.push(`${node.name} lost committed entry ${index}`);
        }
      }
    }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i].log, b = nodes[j].log;
        for (let index = Math.min(a.length, b.length); index >= 1; index--) {
          if (a[index - 1].term !== b[index - 1].term) continue;
          for (let earlier = 1; earlier <= index; earlier++) {
            if (a[earlier - 1].term !== b[earlier - 1].term || a[earlier - 1].command !== b[earlier - 1].command) {
              this.violations.push(`log matching at ${index} (${nodes[i].name}, ${nodes[j].name})`);
            }
          }
          break; // the highest matching index implies the property for every lower one
        }
      }
    }
    for (const node of nodes) {
      if (node.commitIndex > node.log.length) this.violations.push(`${node.name} committed past its log`);
      if (node.applied.length !== node.commitIndex) this.violations.push(`${node.name} applied != commitIndex`);
    }
  }
  committedCommands() {
    return [...this.committed.keys()].sort((a, b) => a - b).map((index) => this.committed.get(index).command);
  }
}

module.exports = { FOLLOWER, CANDIDATE, LEADER, Network, RaftNode, RaftSim };

if (require.main === module) {
  const assert = require("node:assert");
  const names = ["n0", "n1", "n2", "n3", "n4"];
  const settled = ["a", "b", "c", "d", "e"];
  // --- fault-free election and replication ------------------------------------------------
  const sim = new RaftSim(names); const node = (name) => sim.nodes.get(name);
  sim.run(12); const first = sim.leaderName();
  assert.ok(first !== null, "a leader must be elected"); assert.deepStrictEqual(sim.leaders(), [first], "exactly one leader");
  assert.strictEqual(sim.elections, 1, "the staggered timeout wins on the first try");
  for (const command of settled) sim.propose(command);
  sim.run(40);
  assert.deepStrictEqual(sim.violations, [], "no safety violation in the fault-free run"); assert.strictEqual(sim.leadersByTerm.size, 1, "one leader across all terms");
  assert.deepStrictEqual(sim.committedCommands(), settled, "all commands committed in order");
  for (const name of names) {
    assert.deepStrictEqual(node(name).commands(), settled, `${name} converged`);
    assert.strictEqual(node(name).commitIndex, settled.length, `${name} committed everything`);
    assert.deepStrictEqual(node(name).applied, settled, `${name} applied them in order`);
  }
  // --- the leader is isolated: the majority elects a successor and commits ------------------
  const alive = names.filter((name) => name !== first);
  sim.network.partition([alive, [first]]); sim.run(30);
  const successors = alive.filter((name) => node(name).state === LEADER);
  assert.strictEqual(successors.length, 1, "the majority elects exactly one replacement");
  assert.ok(node(successors[0]).term > node(first).term, "in a higher term");
  assert.strictEqual(node(first).state, LEADER, "the deposed leader has not learned yet");
  sim.propose("f", successors[0]);
  sim.propose("doomed", first);
  sim.run(50);
  assert.deepStrictEqual(sim.committedCommands(), settled.concat("f"), "only the majority side commits");
  assert.strictEqual(node(first).commitIndex, settled.length, "the isolated leader advances nothing");
  assert.strictEqual(node(first).commands().length, settled.length + 1, "its extra entry sits in its log");
  // --- healing truncates the uncommitted entry ---------------------------------------------
  sim.network.heal(); sim.run(60);
  assert.deepStrictEqual(sim.violations, [], "healing violates nothing"); assert.ok(sim.truncations > 0, "the losing side had to truncate its uncommitted entry");
  assert.ok(sim.leaderName() !== null, "exactly one leader again after healing");
  for (const name of names) assert.deepStrictEqual(node(name).commands(), settled.concat("f"), `${name} healed`);
  // --- a 2/3 partition: two nodes can never form a quorum ---------------------------------
  const leadersBefore = sim.leaders();
  sim.network.partition([[names[0], names[1]], names.slice(2)]); sim.run(40);
  const smallSide = [names[0], names[1]].filter((name) => node(name).state === LEADER);
  const bigSide = names.slice(2).filter((name) => node(name).state === LEADER);
  assert.ok(smallSide.every((name) => leadersBefore.includes(name)), "no new leader can appear below quorum");
  assert.strictEqual(bigSide.length, 1, "the three-node side elects exactly one leader");
  assert.ok(smallSide.every((name) => node(name).term < node(bigSide[0]).term), "the stale side is behind");
  sim.propose("g", bigSide[0]); sim.run(40);
  assert.ok(sim.committedCommands().includes("g"), "the majority side keeps making progress");
  sim.network.heal(); sim.run(80);
  assert.deepStrictEqual(sim.violations, [], "no violation after the second heal");
  for (const name of names) {
    assert.deepStrictEqual(node(name).commands(), settled.concat(["f", "g"]), `${name} converged again`);
    assert.deepStrictEqual(node(name).applied, sim.committedCommands().slice(0, node(name).commitIndex));
  }
  // --- safety and liveness under message loss ----------------------------------------------
  const lossy = new RaftSim(["m0", "m1", "m2", "m3", "m4"], { lossEvery: 4 }); lossy.run(60);
  assert.deepStrictEqual(lossy.violations, [], "message loss must never break safety");
  assert.ok(lossy.leaderName() !== null, "a leader is still elected under loss"); lossy.propose("x");
  lossy.run(80);
  assert.ok(lossy.committedCommands().includes("x"), "retries eventually commit despite loss");
  assert.ok(lossy.network.dropped > 0, "the loss rule really dropped messages");
  // --- boundaries --------------------------------------------------------------------------
  const single = new RaftSim(["solo"]);
  assert.strictEqual(single.majority(), 1);
  single.run(8);
  assert.strictEqual(single.leaderName(), "solo", "a single node elects itself immediately");
  single.propose("only"); single.run(5);
  assert.deepStrictEqual(single.committedCommands(), ["only"]);
  const voter = new RaftSim(names);
  voter.run(12);
  const incumbent = voter.leaderName(); voter.nodes.get(incumbent).term += 5;
  voter.onRequestVote(voter.nodes.get(incumbent), { type: "requestVote", term: 1, candidate: names[4], lastLogIndex: 99, lastLogTerm: 99 });
  assert.strictEqual(voter.nodes.get(incumbent).term, 6, "a lower-term candidate cannot depose the leader");
  assert.notStrictEqual(voter.nodes.get(incumbent).votedFor, names[4], "and its vote is not granted");
  console.log(`raft ticks=${sim.tick} elections=${sim.elections} truncations=${sim.truncations} dropped=${sim.network.dropped + lossy.network.dropped}`);
  console.log("PASS");
}
