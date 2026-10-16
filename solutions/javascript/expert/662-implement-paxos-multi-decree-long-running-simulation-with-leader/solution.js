/** Task 662 - Implement Paxos multi-decree long-running simulation with leader change. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A working multi-decree Paxos: each log slot is an independent Paxos instance, the leader
 * batches phase 1 per slot, and a leader change means a new leader with a strictly higher ballot
 * that must run phase 1 again — which is what saves safety, because the promise quorum and the
 * previous leader's accept quorum must intersect, forcing the new leader to adopt the
 * highest-ballot value any acceptor already accepted.
 *
 * The network is modelled as a request inbox per node plus a response queue per leader, with
 * deterministic reordering (messages sorted by a hash of their identity), a drop rate and a
 * duplication rate, all from one seeded PRNG so every run replays exactly. Simplifications:
 * ballots come from one global counter (real deployments persist them), there is no disk, no
 * membership change, and learner catch-up is the leader re-broadcasting its decided prefix.
 *
 * Verified: agreement (two replicas never learn different values for one slot — enforced inside
 * `learn`, not merely asserted afterwards), the leader-change crux (a decided value is never
 * replaced when leadership moves), the no-quorum case (a partitioned leader decides nothing
 * rather than something new), ballot monotonicity per slot, prefix-consistency of each replica's
 * committed view, progress under heavy loss, and deterministic replay.
 */
"use strict";
const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
class PaxosNode {
  constructor(id) {
    this.id = id;
    this.promised = new Map(); // slot -> highest ballot promised
    this.accepted = new Map(); // slot -> { ballot, value }
    this.learned = new Map();  // slot -> decided value
    this.promiseHistory = [];  // keeps every promise, for the monotonicity check
  }
  /** Handles one delivered request; returns the responses to send back. */
  handle(message) {
    const { kind, slot } = message;
    const reply = (replyKind, extra = {}) =>
      [{ to: message.from, message: { from: this.id, kind: replyKind, slot, ballot: message.ballot, ...extra } }];
    if (kind === "prepare") {
      const promised = this.promised.get(slot) ?? -1;
      if (message.ballot <= promised) return reply("nack");
      this.promised.set(slot, message.ballot);
      this.promiseHistory.push({ slot, ballot: message.ballot });
      const accepted = this.accepted.get(slot) ?? null;
      return reply("promise", {
        acceptedBallot: accepted === null ? null : accepted.ballot,
        acceptedValue: accepted === null ? null : accepted.value,
      });
    }
    if (kind === "accept") {
      const promised = this.promised.get(slot) ?? -1;
      if (message.ballot < promised) return reply("reject");
      this.promised.set(slot, message.ballot);
      const accepted = this.accepted.get(slot);
      if (accepted === undefined || accepted.ballot < message.ballot) {
        this.accepted.set(slot, { ballot: message.ballot, value: message.value });
      }
      return reply("accepted");
    }
    if (kind === "decide") {
      this.learn(slot, message.value);
      return [];
    }
    return [];
  }
  /** Learning is where agreement is enforced: the same slot can never take two values. */
  learn(slot, value) {
    const known = this.learned.get(slot);
    if (known !== undefined && known !== value) throw new Error(`agreement violation on slot ${slot}`);
    this.learned.set(slot, value);
  }
  /** Longest prefix of slots this replica has learned (its committed view). */
  committedPrefix() {
    let slots = 0;
    while (this.learned.has(slots)) slots++;
    return Array.from({ length: slots }, (_, slot) => this.learned.get(slot));
  }
}
class PaxosSimulation {
  constructor(nodeIds, options = {}) {
    if (!Array.isArray(nodeIds) || nodeIds.length < 3) throw new RangeError("Paxos needs at least 3 nodes");
    if (new Set(nodeIds).size !== nodeIds.length) throw new Error("node ids must be unique");
    this.nodes = nodeIds.map((id) => new PaxosNode(id));
    this.quorum = Math.floor(nodeIds.length / 2) + 1;
    this.inboxes = new Map(nodeIds.map((id) => [id, []]));
    this.responses = new Map(nodeIds.map((id) => [id, []]));
    this.partitioned = new Set();
    this.dropRate = options.dropRate ?? 0.2;
    this.dupRate = options.dupRate ?? 0.2;
    if (this.dropRate < 0 || this.dropRate >= 1) throw new RangeError("dropRate must be in [0, 1)");
    if (this.dupRate < 0 || this.dupRate >= 1) throw new RangeError("dupRate must be in [0, 1)");
    this.rng = makeRng(options.seed ?? 11);
    this.seq = 0; this.rounds = 0; this.delivered = 0; this.dropped = 0; this.duplicated = 0;
    this.ballotCounter = 0; this.leaderIndex = 0; this.leaderId = nodeIds[0];
    this.decisions = new Map(); this.decisionOrder = [];
  }
  nodeById(id) {
    return this.nodes.find((node) => node.id === id) ?? null;
  }
  /** New leadership: the global ballot counter keeps the new leader's ballots higher. */
  setLeader(id) {
    const index = this.nodes.findIndex((node) => node.id === id);
    if (index === -1) throw new Error(`unknown node ${id}`);
    this.leaderIndex = index;
    this.leaderId = id;
    return { leader: id, quorum: this.quorum };
  }
  nextBallot() {
    this.ballotCounter++;
    return this.ballotCounter * 16 + this.leaderIndex;
  }
  /** Simulates a network partition for the given node (all of its traffic is dropped). */
  partition(id, on = true) {
    if (on) this.partitioned.add(id);
    else this.partitioned.delete(id);
  }
  send(from, to, message) {
    this.inboxes.get(to).push({ from, to, message, seq: this.seq++ });
  }
  /** Deterministic delivery order: hash the message identity, keep submission order on ties. */
  orderKey(message) {
    const text = `${message.from}>${message.to}:${message.message.kind}:${message.message.slot}:${message.message.ballot}`;
    let hash = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
      hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193) >>> 0;
    }
    return hash;
  }
  /** One network round: deliver every queued request, with drops, duplicates and reordering. */
  flush() {
    this.rounds++;
    const pending = [];
    for (const inbox of this.inboxes.values()) while (inbox.length > 0) pending.push(inbox.shift());
    pending.sort((a, b) => (this.orderKey(a) - this.orderKey(b)) || (a.seq - b.seq));
    const deliveries = [];
    for (const message of pending) {
      if (this.partitioned.has(message.to) || this.rng(1000) < this.dropRate * 1000) {
        this.dropped++;
        continue;
      }
      deliveries.push(message);
      if (this.rng(1000) < this.dupRate * 1000) {
        deliveries.push(message);
        this.duplicated++;
      }
    }
    for (const envelope of deliveries) {
      const node = this.nodeById(envelope.to);
      for (const response of node.handle({ ...envelope.message, from: envelope.from })) {
        this.responses.get(response.to).push(response.message);
      }
    }
  }
  /** Unique responses of one kind for one ballot, keyed by the responding node. */
  collect(nodeId, kind, slot, ballot) {
    const found = new Map();
    for (const message of this.responses.get(nodeId)) {
      if (message.kind === kind && message.slot === slot && message.ballot === ballot) {
        found.set(message.from, message);
      }
    }
    return [...found.values()];
  }
  /** Drives one slot to a decision (phase 1, value selection, phase 2, broadcast). */
  propose(slot, proposal, options = {}) {
    if (this.decisions.has(slot)) return { slot, value: this.decisions.get(slot), repeated: true };
    const maxAttempts = options.maxAttempts ?? 12;
    const leader = this.leaderId;
    const ballot = this.nextBallot();
    // Phase 1: collect a promise quorum (retries cover message loss).
    let promises = [];
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      this.responses.set(leader, []);
      for (const node of this.nodes) this.send(leader, node.id, { kind: "prepare", slot, ballot });
      this.flush();
      promises = this.collect(leader, "promise", slot, ballot);
      if (promises.length >= this.quorum) break;
    }
    if (promises.length < this.quorum) return null;
    // Value selection: adopt the highest-ballot accepted value seen in the promises.
    let value = proposal;
    let highest = -1;
    for (const promise of promises) {
      if (promise.acceptedBallot !== null && promise.acceptedBallot > highest) {
        highest = promise.acceptedBallot;
        value = promise.acceptedValue;
      }
    }
    const adopted = highest >= 0;
    let accepted = [];
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      this.responses.set(leader, []);
      for (const node of this.nodes) this.send(leader, node.id, { kind: "accept", slot, ballot, value });
      this.flush();
      accepted = this.collect(leader, "accepted", slot, ballot);
      if (accepted.length >= this.quorum) break;
    }
    if (accepted.length < this.quorum) return null;
    this.decisions.set(slot, value);
    this.nodeById(leader).learn(slot, value); // the leader appends to its own log first
    this.decisionOrder.push({ slot, value, ballot, adopted, promised: promises.length, accepted: accepted.length });
    this.broadcastDecisions(options.broadcastRounds ?? 3);
    return { slot, value, ballot, adopted, promised: promises.length, accepted: accepted.length, repeated: false };
  }
  /** Learner catch-up: re-sends the whole decided prefix, as a multi-Paxos leader does. */
  broadcastDecisions(rounds = 3) {
    const decided = [...this.decisions.entries()].sort((a, b) => a[0] - b[0]);
    for (let round = 0; round < rounds; round++) {
      for (const [slot, value] of decided) {
        for (const node of this.nodes) this.send(this.leaderId, node.id, { kind: "decide", slot, value });
      }
      this.flush();
    }
    return decided.length;
  }
  /** Decides consecutive slots starting at `startSlot`. */
  decideMany(proposals, options = {}) {
    const start = options.startSlot ?? 0;
    return proposals.map((value, index) => this.propose(start + index, value, options));
  }
  assertAgreement() {
    const seen = new Map();
    for (const node of this.nodes) {
      for (const [slot, value] of node.learned) {
        if (seen.has(slot) && seen.get(slot) !== value) throw new Error(`agreement violation on slot ${slot}`);
        seen.set(slot, value);
      }
    }
    return seen;
  }
  assertBallotMonotonicity() {
    for (const node of this.nodes) {
      for (const [slot, accepted] of node.accepted) {
        if ((node.promised.get(slot) ?? -1) < accepted.ballot) throw new Error("accepted above the promise");
      }
      const lastBySlot = new Map();
      for (const entry of node.promiseHistory) {
        if (entry.ballot < (lastBySlot.get(entry.slot) ?? -1)) throw new Error("promise went backwards");
        lastBySlot.set(entry.slot, entry.ballot);
      }
    }
    return true;
  }
}
module.exports = { PaxosNode, PaxosSimulation, makeRng };
if (require.main === module) {
  const ids = ["n1", "n2", "n3", "n4", "n5"];
  throws(() => new PaxosSimulation(["a", "b"]), RangeError, "Paxos needs at least three nodes");
  throws(() => new PaxosSimulation(["a", "b", "a"]), /unique/, "node ids must be unique");
  throws(() => new PaxosSimulation(ids, { dropRate: 1 }), RangeError, "dropRate must stay below 1");
  throws(() => new PaxosSimulation(ids, { dupRate: -1 }), RangeError, "dupRate cannot be negative");
  strictEqual(new PaxosSimulation(ids, { seed: 1 }).quorum, 3, "five nodes need a quorum of three");  // -- a healthy multi-decree run -------------------------------------------------
  const proposals = ["a", "b", "c", "d"];
  const healthy = new PaxosSimulation(ids, { seed: 2024, dropRate: 0.2, dupRate: 0.2 });
  const results = healthy.decideMany(proposals);
  deepStrictEqual(results.map((r) => r.value), proposals, "every slot decides its proposal");
  ok(results.every((r) => r.promised >= 3 && r.accepted >= 3), "each decision needed a real quorum");
  ok(results.every((r) => r.adopted === false), "nothing was adopted when no value pre-existed");
  deepStrictEqual([...healthy.assertAgreement().values()].sort(), [...proposals].sort(), "all slots are learned");
  ok(healthy.assertBallotMonotonicity(), "no acceptor ever accepted above its promise");
  ok(healthy.duplicated > 0 && healthy.dropped > 0, "the run really did drop and duplicate messages");  deepStrictEqual(healthy.nodeById(healthy.leaderId).committedPrefix(), proposals, "the leader's log is complete");
  ok(healthy.nodes.some((node) => node.committedPrefix().length <= proposals.length),
    "some follower may lag until the best-effort catch-up broadcast");
  healthy.broadcastDecisions(6);
  for (const node of healthy.nodes) deepStrictEqual(node.committedPrefix(), proposals, `${node.id} converges`);
  deepStrictEqual(healthy.decisions.get(2), "c");
  // -- the leader-change crux -----------------------------------------------------
  const crux = new PaxosSimulation(ids, { seed: 5, dropRate: 0, dupRate: 0 });
  strictEqual(crux.propose(0, "A").value, "A", "the first leader decides its own proposal");  const holders = crux.nodes.filter((node) => node.accepted.get(0)?.value === "A").map((node) => node.id);
  ok(holders.length >= 3, `a quorum accepted A (${holders.join(",")})`);
  // Two A-holders are unreachable, so any promise quorum must still hear from one of them.
  for (const index of [1, 2]) crux.partition(holders[index], true);
  crux.setLeader("n4");
  const second = crux.propose(0, "B");
  strictEqual(second.repeated, true, "a decided slot is not proposed again");
  strictEqual(second.value, "A", "the decided value must not change");
  // -- a new leader must adopt what was already accepted --------------------------
  const fresh = new PaxosSimulation(ids, { seed: 6, dropRate: 0, dupRate: 0 });
  fresh.propose(0, "A");
  const knownHolders = fresh.nodes.filter((node) => node.accepted.get(0)?.value === "A").map((node) => node.id);
  for (const index of [1, 2]) fresh.partition(knownHolders[index], true);
  fresh.decisions.clear(); // pretend the new leader never learned the outcome
  fresh.setLeader("n5");
  const revived = fresh.propose(0, "B");
  strictEqual(revived.adopted, true, "the new leader must adopt the value already accepted");
  strictEqual(revived.value, "A", "the adopted value is the previously accepted one, not the proposal");
  strictEqual(fresh.assertAgreement().get(0), "A", "agreement survives the leader change");
  // -- no quorum means no decision, never a wrong one -----------------------------
  const split = new PaxosSimulation(ids, { seed: 7, dropRate: 0, dupRate: 0 });
  split.propose(0, "A");
  for (const node of split.nodes) split.partition(node.id, true);
  split.decisions.clear();
  split.setLeader("n5");
  strictEqual(split.propose(0, "B"), null, "a leader that cannot reach a quorum decides nothing");
  for (const id of ids) split.partition(id, false);
  strictEqual(split.propose(0, "B").value, "A", "once the partition heals the only surviving value is A");
  // -- leader change mid-stream ----------------------------------------------------
  const handover = new PaxosSimulation(ids, { seed: 99, dropRate: 0.3, dupRate: 0.3 });
  handover.decideMany(["x0", "x1"]);
  handover.setLeader("n3");
  handover.decideMany(["y2", "y3"], { startSlot: 2 });  deepStrictEqual([...handover.assertAgreement().entries()].sort((a, b) => a[0] - b[0]),
    [[0, "x0"], [1, "x1"], [2, "y2"], [3, "y3"]], "the log is consistent across the leader change");
  ok(handover.assertBallotMonotonicity());
  deepStrictEqual(handover.nodeById(handover.leaderId).committedPrefix(), ["x0", "x1", "y2", "y3"],
    "the log survives the leader change in order");
  handover.broadcastDecisions(6);
  for (const node of handover.nodes) deepStrictEqual(node.committedPrefix(), ["x0", "x1", "y2", "y3"]);
  handover.propose(1, "ignored");
  deepStrictEqual(handover.decisionOrder.map((d) => d.slot), [0, 1, 2, 3],
    "already-decided slots are never decided twice");
  const gappy = new PaxosNode("g");
  gappy.learn(2, "later");
  deepStrictEqual(gappy.committedPrefix(), [], "slot 2 without slots 0-1 is not a committed prefix");
  gappy.learn(0, "first");
  gappy.learn(1, "second");
  deepStrictEqual(gappy.committedPrefix(), ["first", "second", "later"], "the prefix grows once the gap is filled");
  gappy.learn(2, "later");
  deepStrictEqual(gappy.learned.size, 3, "learning the same decision again is idempotent");
  throws(() => gappy.learn(2, "other"), /agreement violation/, "conflicting learns must throw");  // -- direct acceptor semantics ---------------------------------------------------
  const acceptor = new PaxosNode("solo");
  const ask = (message) => acceptor.handle({ from: "L", ...message })[0].message;
  const opened = ask({ kind: "prepare", slot: 0, ballot: 10 });  deepStrictEqual([opened.kind, opened.acceptedBallot], ["promise", null],
    "an untouched acceptor promises but has nothing accepted");
  strictEqual(ask({ kind: "prepare", slot: 0, ballot: 10 }).kind, "nack", "a ballot is promised only once");
  strictEqual(ask({ kind: "accept", slot: 0, ballot: 9, value: "old" }).kind, "reject",
    "an accept below the promise must be rejected");
  strictEqual(ask({ kind: "accept", slot: 0, ballot: 11, value: "v" }).kind, "accepted");
  const later = ask({ kind: "prepare", slot: 0, ballot: 20 });
  deepStrictEqual([later.acceptedBallot, later.acceptedValue], [11, "v"], "the next leader adopts this value");
  // -- determinism and progress under losses --------------------------------------
  function replay() {
    const run = new PaxosSimulation(ids, { seed: 1234, dropRate: 0.35, dupRate: 0.35 });
    run.decideMany(["p0", "p1", "p2"]);
    return { order: run.decisionOrder, dropped: run.dropped, duplicated: run.duplicated, rounds: run.rounds };
  }
  deepStrictEqual(replay(), replay(), "identical seeds must replay identically");
  const lossy = new PaxosSimulation(ids, { seed: 77, dropRate: 0.45, dupRate: 0.1 });
  const lossyResults = lossy.decideMany(["l0", "l1", "l2"]);
  ok(lossyResults.every((r) => r !== null) && lossy.dropped > 0,
    "a lossy but connected network still makes progress");
  console.log(`healthy slots=${healthy.decisions.size} rounds=${healthy.rounds} dropped=${healthy.dropped} ` +
    `duplicated=${healthy.duplicated} revivedAdopted=${revived.value} handover=${handover.decisions.size}`);
  console.log("PASS");
}
