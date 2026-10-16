/** Task 661 - Implement chain replication simulation for strong consistency. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Chain replication (van Renesse & Schneider): the replicas form a singly linked chain. All
 * `update`s enter at the *head* and are forwarded hop by hop (store and forward — a hop is only
 * scheduled once the previous hop delivered); an entry is *committed* when it reaches the
 * *tail*, and reads are served by the tail only. That split gives strong consistency for free:
 * the tail's log is exactly the committed prefix of the head's log.
 *
 * Simulation model: deterministic per-node hop latencies, a discrete event list, writes
 * submitted at chosen times. Simplifications (documented, not hidden): a crashed node is removed
 * rather than repaired; recovery re-propagates the new head's suffix through the shortened chain
 * instead of running the real repair protocol; there is no durability model and no client retry.
 *
 * Invariants verified, continuously and by assertion:
 *   * prefix invariant — after *every* delivery, each replica's log is a prefix of its
 *     predecessor's log, so a replica can never show an entry the head has not produced yet;
 *   * commit visibility — an entry is unreadable until it reaches the tail, and the tail's log
 *     is monotone: reads never go backwards;
 *   * atomic append — ids are appended in order, duplicate deliveries are idempotent and
 *     out-of-order deliveries are rejected;
 *   * reconfiguration — crashing any node preserves the prefix invariant, and draining after the
 *     crash brings the new tail back level with the new head, so the surviving data is intact.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

class ChainReplication {
  constructor(nodeIds, options = {}) {
    if (!Array.isArray(nodeIds) || nodeIds.length === 0) throw new RangeError("a chain needs at least one node");
    if (new Set(nodeIds).size !== nodeIds.length) throw new Error("node ids must be unique");
    const hopLatency = options.hopLatency ?? nodeIds.map((_, i) => 5 + 2 * i);
    if (hopLatency.length !== nodeIds.length) throw new RangeError("one hop latency per node is required");
    if (hopLatency.some((latency) => !(latency > 0))) throw new RangeError("hop latencies must be positive");
    this.nodes = nodeIds.map((id) => ({ id, log: [] }));
    this.latencies = new Map(nodeIds.map((id, i) => [id, hopLatency[i]]));
    this.events = [];
    this.seq = 0;
    this.clock = 0;
    this.nextWriteId = 0;
    this.committed = [];
    this.droppedEvents = 0;
    this.gapDrops = 0;
    this.reconfigurations = 0;
    this.invariantChecks = 0;
  }

  get head() {
    return this.nodes[0];
  }

  get tail() {
    return this.nodes[this.nodes.length - 1];
  }

  nodeById(id) {
    return this.nodes.find((node) => node.id === id) ?? null;
  }

  successorOf(id) {
    const index = this.nodes.findIndex((node) => node.id === id);
    if (index === -1) return null;
    return index + 1 < this.nodes.length ? this.nodes[index + 1] : null;
  }

  schedule(event) {
    this.events.push({ ...event, seq: this.seq++ });
  }

  /** Submits a write at the head. The commit time is only known once it reaches the tail. */
  update(value, startTime = this.clock) {
    const head = this.head;
    const writeId = this.nextWriteId++;
    this.schedule({ time: startTime + this.latencies.get(head.id), nodeId: head.id, writeId, value });
    return { writeId };
  }

  /** Store and forward: hand the entry to the next replica, or commit when at the tail. */
  forward(node, writeId, value, time) {
    if (node.id === this.tail.id) {
      this.committed.push({ writeId, value, commitTime: time });
      return;
    }
    const next = this.successorOf(node.id);
    this.schedule({ time: time + this.latencies.get(next.id), nodeId: next.id, writeId, value });
  }

  deliver(event) {
    const node = this.nodeById(event.nodeId);
    if (node === null) {
      this.droppedEvents++; // the replica crashed while the entry was in flight
      return;
    }
    if (event.writeId < node.log.length) return; // duplicate delivery: idempotent
    if (event.writeId > node.log.length) {
      if (this.reconfigurations === 0) throw new Error("out-of-order delivery: the chain has a gap");
      this.gapDrops++; // a reconfiguration removed an earlier entry; the repair pass owns it
      return;
    }
    node.log.push({ writeId: event.writeId, value: event.value });
    this.forward(node, event.writeId, event.value, event.time);
    this.assertPrefixInvariant();
  }

  drain(until = Infinity) {
    this.events.sort((a, b) => (a.time - b.time) || (a.seq - b.seq));
    while (this.events.length > 0 && this.events[0].time <= until) {
      const event = this.events.shift();
      this.clock = Math.max(this.clock, event.time);
      this.deliver(event);
    }
    if (Number.isFinite(until) && until > this.clock) this.clock = until;
  }

  /** True when every replica's log is a prefix of its predecessor's (data flows head -> tail). */
  assertPrefixInvariant() {
    for (let i = 1; i < this.nodes.length; i++) {
      const leader = this.nodes[i - 1].log;
      const follower = this.nodes[i].log;
      if (follower.length > leader.length) {
        throw new Error("prefix violation: a replica holds an entry its predecessor has not seen");
      }
      for (let k = 0; k < follower.length; k++) {
        if (follower[k].writeId !== leader[k].writeId || follower[k].value !== leader[k].value) {
          throw new Error(`prefix violation between ${this.nodes[i - 1].id} and ${this.nodes[i].id} at ${k}`);
        }
      }
    }
    this.invariantChecks++;
    return true;
  }

  /** Reads are served by the tail only, and only for entries that have arrived there. */
  readFrom(nodeId, until = Infinity) {
    this.drain(until);
    const node = this.nodeById(nodeId);
    if (node === null) throw new Error(`node ${nodeId} is not in the chain`);
    if (node.id !== this.tail.id) throw new Error(`reads must be served by the tail; ${nodeId} is not the tail`);
    return node.log.length === 0 ? null : { ...node.log[node.log.length - 1] };
  }

  read(until = Infinity) {
    return this.readFrom(this.tail.id, until);
  }

  /** The committed prefix as the tail sees it — the linearisable view of the data. */
  readAll(until = Infinity) {
    this.drain(until);
    return this.tail.log.map((entry) => ({ ...entry }));
  }

  /** Removes a replica and re-propagates whatever the new head holds beyond the new tail. */
  crash(nodeId) {
    if (this.nodeById(nodeId) === null) throw new Error(`node ${nodeId} is not in the chain`);
    if (this.nodes.length === 1) throw new Error("cannot crash the last remaining node");
    this.nodes = this.nodes.filter((node) => node.id !== nodeId);
    this.reconfigurations++;
    const head = this.head;
    for (let k = this.tail.log.length; k < head.log.length; k++) {
      this.forward(head, head.log[k].writeId, head.log[k].value, this.clock);
    }
    this.assertPrefixInvariant();
    return { chain: this.nodes.map((node) => node.id), head: this.head.id, tail: this.tail.id };
  }
}

module.exports = { ChainReplication };

if (require.main === module) {
  // -- three hops, commit time is the sum of the hop latencies -------------------
  const chain = new ChainReplication(["r1", "r2", "r3"], { hopLatency: [4, 6, 8] });
  chain.update("v0", 0);
  chain.update("v1", 0);
  chain.drain();
  deepStrictEqual(chain.committed.map((c) => c.value), ["v0", "v1"], "commit order follows submission order");
  strictEqual(chain.committed[0].commitTime, 18, "the first commit takes the full chain latency");
  deepStrictEqual(chain.readAll().map((e) => e.value), ["v0", "v1"]);
  strictEqual(chain.read().value, "v1", "the tail serves the newest committed value");
  ok(chain.invariantChecks > 4, `the prefix invariant is checked after every delivery (${chain.invariantChecks})`);
  ok(chain.assertPrefixInvariant());
  deepStrictEqual(chain.head.log.length === chain.tail.log.length, true, "a drained chain is level");

  // -- a write is invisible until it reaches the tail ---------------------------
  const stream = new ChainReplication(["a", "b", "c"], { hopLatency: [5, 5, 5] });
  stream.update("w0", 0);
  strictEqual(stream.read(4), null, "nothing is committed before the first hop");
  strictEqual(stream.read(9), null, "the head and middle alone cannot make a write readable");
  strictEqual(stream.read(14), null, "still not at the tail");
  strictEqual(stream.read(15).value, "w0", "at 15ms the entry has reached the tail");
  const seen = [];
  for (const time of [0, 3, 7, 11, 14, 15, 16, 30]) seen.push(stream.read(time)?.writeId ?? -1);
  for (let i = 1; i < seen.length; i++) ok(seen[i] >= seen[i - 1], "reads must never go backwards");
  throws(() => stream.readFrom("a"), /served by the tail/, "the head must refuse reads");
  throws(() => stream.readFrom("zz"), /not in the chain/, "an unknown node cannot serve reads");

  // -- head crash: the tail is untouched, so committed data is untouched ---------
  const durable = new ChainReplication(["h", "m", "t"], { hopLatency: [1, 1, 1] });
  for (let i = 0; i < 4; i++) durable.update(`d${i}`, 0);
  durable.drain();
  const acked = durable.readAll().map((e) => e.value);
  deepStrictEqual(acked, ["d0", "d1", "d2", "d3"]);
  const afterHeadCrash = durable.crash("h");
  deepStrictEqual(afterHeadCrash.chain, ["m", "t"], "the chain shortens by one");
  strictEqual(afterHeadCrash.tail, "t", "the tail is unchanged by a head crash");
  deepStrictEqual(durable.readAll().map((e) => e.value), acked, "every committed write is still readable");
  durable.update("d4");
  durable.drain();
  deepStrictEqual(durable.readAll().map((e) => e.value), [...acked, "d4"], "the shortened chain keeps committing");

  // -- middle crash: stale events are dropped and the repair pass catches up -----
  const middle = new ChainReplication(["a", "b", "c"], { hopLatency: [2, 2, 2] });
  middle.update("keep", 0);
  middle.drain(3); // "keep" sits in a, its hop to b is scheduled but not delivered
  strictEqual(middle.readAll(3).length, 0, "nothing has reached the tail yet");
  const afterMiddleCrash = middle.crash("b");
  deepStrictEqual(afterMiddleCrash.chain, ["a", "c"]);
  middle.drain();
  ok(middle.droppedEvents >= 1, "the in-flight event addressed to the crashed node is dropped");
  deepStrictEqual(middle.readAll().map((e) => e.value), ["keep"],
    "the repair pass re-propagates the new head's suffix to the new tail");
  ok(middle.assertPrefixInvariant());
  middle.update("fresh", 10);
  middle.drain();
  deepStrictEqual(middle.readAll().map((e) => e.value), ["keep", "fresh"], "new writes flow through the new chain");

  // -- tail crash: the predecessor takes over and catches up ---------------------
  const tailCrash = new ChainReplication(["x", "y", "z"], { hopLatency: [3, 3, 3] });
  tailCrash.update("t0", 0);
  tailCrash.update("t1", 0);
  tailCrash.drain(7); // both entries reached y, the tail z is still one hop away
  strictEqual(tailCrash.nodeById("z").log.length, 0, "the doomed tail had not received the entries yet");
  const afterTailCrash = tailCrash.crash("z");
  deepStrictEqual(afterTailCrash.chain, ["x", "y"]);
  strictEqual(afterTailCrash.tail, "y", "the predecessor becomes the new tail");
  ok(tailCrash.tail.log.length <= tailCrash.head.log.length,
    "the tail can never hold more than the head: new data only ever comes from the head");
  tailCrash.drain();
  deepStrictEqual(tailCrash.readAll().map((e) => e.value), ["t0", "t1"],
    "in-flight entries survive in the head and are re-propagated to the new tail");
  strictEqual(tailCrash.head.log.length, tailCrash.tail.log.length, "head and tail are level after repair");
  ok(tailCrash.invariantChecks > 0);

  // -- degenerate chains --------------------------------------------------------
  const solo = new ChainReplication(["only"], { hopLatency: [7] });
  solo.update("x", 3);
  strictEqual(solo.read(9), null, "a single node chain still needs its hop");
  strictEqual(solo.read(10).value, "x");
  strictEqual(solo.committed[0].commitTime, 10, "a single node chain commits after one hop");
  throws(() => solo.crash("only"), /last remaining node/, "the last node cannot be crashed");
  throws(() => new ChainReplication([]), RangeError, "an empty chain is invalid");
  throws(() => new ChainReplication(["a", "a"]), /unique/, "duplicate node ids are invalid");
  throws(() => new ChainReplication(["a", "b"], { hopLatency: [1] }), RangeError,
    "one latency per node is required");
  throws(() => new ChainReplication(["a"], { hopLatency: [0] }), RangeError, "latencies must be positive");

  // -- duplicate delivery is idempotent, gaps are rejected ----------------------
  const dup = new ChainReplication(["p", "q"], { hopLatency: [1, 1] });
  dup.update("once", 0);
  dup.drain();
  const before = dup.tail.log.length;
  dup.deliver({ time: 99, nodeId: "q", writeId: 0, value: "once" });
  strictEqual(dup.tail.log.length, before, "a repeated delivery must change nothing");
  throws(() => dup.deliver({ time: 99, nodeId: "q", writeId: 5, value: "gap" }),
    /out-of-order/, "a skipped write id must be rejected while the chain is intact");

  // -- determinism --------------------------------------------------------------
  function replay() {
    const run = new ChainReplication(["x", "y", "z"], { hopLatency: [3, 3, 3] });
    for (let i = 0; i < 5; i++) run.update(`v${i}`, i * 2);
    run.drain();
    return { committed: run.committed, tail: run.tail.log, checks: run.invariantChecks };
  }
  deepStrictEqual(replay(), replay(), "identical schedules must replay identically");

  console.log(`chain=[r1,r2,r3] committed=${chain.committed.length} checks=${chain.invariantChecks} ` +
    `dropped=${middle.droppedEvents} gapDrops=${middle.gapDrops} headCrashChain=[${afterHeadCrash.chain.join("->")}]`);
  console.log("PASS");
}
