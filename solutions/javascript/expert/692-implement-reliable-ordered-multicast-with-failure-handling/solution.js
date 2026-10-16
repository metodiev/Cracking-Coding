/** Task 692 - Implement reliable ordered multicast with failure handling. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Model: sequencer-based total-order (atomic) multicast over a fixed membership. One member acts as
 * the sequencer; every `send` is queued and the sequencer stamps the queued messages with
 * consecutive sequence numbers at the start of the next round, appending them to an ordered log.
 * Members receive messages into a buffer, acknowledge them, and may deliver the next expected
 * sequence number only when
 *     (a) every earlier sequence number has already been delivered by that member (FIFO/total order),
 *     (b) the message is *stable*: every member the failure detector still expects has acknowledged
 *         it. A crashed member stops being expected once it has been silent for `failureThreshold`
 *         rounds, so a dead member can stall delivery only for that bounded window.
 * Losses are injected per (member, sequence) and are retransmitted by the sequencer after the
 * configured delay; a recovering member is repaired by state transfer from the sequencer's log, so
 * it ends up delivering exactly the same sequence of messages as everybody else.
 *
 * Simplified versus a real deployable protocol: round granularity, one loss-failure-detector with a
 * timeout in rounds instead of heartbeats over a real network, no leader election (the sequencer is
 * fixed, so its crash stops ordering - no message can be *created* while it is down - but never
 * breaks the safety of the messages already ordered), no view change, no message batching, and
 * state transfer is assumed to be complete in one round.
 *
 * Verified: total order (the delivered sequence is identical at every member), no duplicates and no
 * gaps, per-sender FIFO and sequencer order, no message created by a crashed process, retransmission
 * of an injected loss with the order preserved, bounded stalling when a member crashes (delivery
 * resumes once the failure detector times the member out) and no retransmission to a member declared
 * failed, catch-up after recovery, safety while the sequencer itself is down, and determinism.
 */
"use strict";

class SequencerMulticast {
  constructor(members, { failureThreshold = 3, sequencer = null } = {}) {
    if (!Number.isInteger(failureThreshold) || failureThreshold < 1) {
      throw new RangeError("failureThreshold must be a positive integer");
    }
    this.members = [...members].sort((a, b) => a - b);
    if (this.members.length === 0) throw new Error("membership must not be empty");
    this.sequencer = sequencer === null ? this.members[0] : sequencer;
    if (!this.members.includes(this.sequencer)) throw new Error("the sequencer must be a member");
    this.failureThreshold = failureThreshold;

    this.up = new Map(this.members.map((id) => [id, true]));
    this.declared = new Map(this.members.map((id) => [id, false]));
    this.silence = new Map(this.members.map((id) => [id, 0]));
    this.buffers = new Map(this.members.map((id) => [id, new Map()]));
    this.delivered = new Map(this.members.map((id) => [id, []]));
    this.deliveredSeqs = new Map(this.members.map((id) => [id, new Set()]));
    this.nextExpected = new Map(this.members.map((id) => [id, 1]));

    this.log = [];                                  // ordered messages, index + 1 === sequence number
    this.acknowledgers = new Map();                 // sequence -> Set(memberId)
    this.pending = [];                              // send intents not ordered yet
    this.drops = new Map();                         // `${member}|${seq}` -> rounds of artificial loss
    this.round = 0;
    this.stats = { retransmissions: 0, blockedRounds: 0, declarations: 0, stateTransfers: 0 };
    this.trace = [];
  }

  isUp(id) {
    return this.up.get(id) === true;
  }

  /**
   * A member the failure detector still expects. A crash is not instantly known to the other
   * members, so a member stays expected (and can therefore delay delivery) until the detector has
   * seen it silent for `failureThreshold` rounds.
   */
  isExpected(id) {
    return this.declared.get(id) !== true;
  }

  /** Returns the queue position, or null when the sender cannot create a message. */
  send(sender, payload) {
    if (!this.members.includes(sender)) throw new Error(`unknown member ${sender}`);
    if (!this.isUp(sender)) return null;             // no message creation by a crashed process
    this.pending.push({ sender, payload });
    return this.pending.length;
  }

  /** Artificially lose one multicast to `member` for `rounds` rounds (the sequencer retransmits). */
  injectDrop(member, seq, rounds = 1) {
    this.drops.set(`${member}|${seq}`, rounds);
  }

  crash(member) {
    this.up.set(member, false);
    this.silence.set(member, 0);
  }

  /** Recovery is a state transfer from the sequencer's log plus an empty failure-detector state. */
  recover(member) {
    this.up.set(member, true);
    this.declared.set(member, false);
    this.silence.set(member, 0);
    this.stats.stateTransfers++;
  }

  message(seq) {
    return seq >= 1 && seq <= this.log.length ? this.log[seq - 1] : null;
  }

  /** A message is stable when every expected member has acknowledged it. */
  isStable(seq) {
    const acked = this.acknowledgers.get(seq);
    if (acked === undefined) return false;
    for (const id of this.members) {
      if (this.isExpected(id) && !acked.has(id)) return false;
    }
    return true;
  }

  step() {
    this.round++;
    const events = { round: this.round, ordered: [], received: 0, delivered: [], declaredFailed: [], blocked: [] };

    // 1. the sequencer stamps the queued messages with the next sequence numbers
    if (this.isUp(this.sequencer)) {
      for (const intent of this.pending.splice(0)) {
        const seq = this.log.length + 1;
        this.log.push({ seq, sender: intent.sender, payload: intent.payload, orderedAt: this.round });
        this.acknowledgers.set(seq, new Set());
        events.ordered.push(seq);
      }
    }                                                 // otherwise the intents stay queued for later

    // 2. multicast: every up member collects the log entries it is missing (or waits for a resend)
    for (const id of this.members) {
      if (!this.isUp(id)) continue;
      let heard = false;
      for (const entry of this.log) {
        if (this.deliveredSeqs.get(id).has(entry.seq) || this.buffers.get(id).has(entry.seq)) continue;
        const key = `${id}|${entry.seq}`;
        const remaining = this.drops.get(key);
        if (remaining !== undefined) {
          this.stats.retransmissions++;
          if (remaining <= 1) this.drops.delete(key);
          else this.drops.set(key, remaining - 1);
          continue;                                   // the sequencer will resend it
        }
        this.buffers.get(id).set(entry.seq, entry);
        this.acknowledgers.get(entry.seq).add(id);
        events.received++;
        heard = true;
      }
      if (heard) this.silence.set(id, 0);
    }

    // 3. failure detector: silence is measured in rounds without an acknowledgement
    for (const id of this.members) {
      if (this.isUp(id)) {
        if (this.silence.get(id) > 0) this.silence.set(id, this.silence.get(id) - 1);
      } else if (this.declared.get(id) !== true) {
        this.silence.set(id, this.silence.get(id) + 1);
        if (this.silence.get(id) >= this.failureThreshold) {
          this.declared.set(id, true);
          this.stats.declarations++;
          events.declaredFailed.push(id);
        }
      }
    }

    // 4. delivery: the in-order prefix whose messages are also stable
    for (const id of this.members) {
      if (!this.isUp(id)) continue;
      for (;;) {
        const seq = this.nextExpected.get(id);
        const entry = this.buffers.get(id).get(seq);
        if (entry === undefined) break;
        if (!this.isStable(seq)) {
          this.stats.blockedRounds++;
          events.blocked.push([id, seq]);
          break;
        }
        this.buffers.get(id).delete(seq);
        this.delivered.get(id).push({ seq, sender: entry.sender, payload: entry.payload, round: this.round });
        this.deliveredSeqs.get(id).add(seq);
        this.nextExpected.set(id, seq + 1);
        events.delivered.push([id, seq]);
      }
    }

    this.trace.push(events);
    return events;
  }

  run(rounds) {
    const out = [];
    for (let i = 0; i < rounds; i++) out.push(this.step());
    return out;
  }

  deliveredPayloads(member) {
    return this.delivered.get(member).map((entry) => `${entry.sender}:${entry.payload}`);
  }

  /** Total order: every member that is currently up delivered exactly the same sequence. */
  agreementViolations() {
    const problems = [];
    const upMembers = this.members.filter((id) => this.isUp(id));
    if (upMembers.length === 0) return problems;
    const reference = upMembers[0];
    for (const id of upMembers.slice(1)) {
      const left = this.deliveredPayloads(reference).join(",");
      const right = this.deliveredPayloads(id).join(",");
      if (left !== right) problems.push(`${reference} and ${id} disagree: ${left} vs ${right}`);
    }
    return problems;
  }

  checkIntegrity(member) {
    const problems = [];
    const seqs = this.delivered.get(member).map((entry) => entry.seq);
    for (let i = 0; i < seqs.length; i++) {
      if (seqs[i] !== i + 1) problems.push(`${member}: gap or duplicate at position ${i} (seq ${seqs[i]})`);
    }
    return problems;
  }
}

module.exports = { SequencerMulticast };

if (require.main === module) {
  const assert = require("node:assert");

  function newGroup(members = [1, 2, 3], options = {}) {
    return new SequencerMulticast(members, options);
  }

  // --- the happy path: total order, no gaps, per-sender FIFO ---------------------------------
  const group = newGroup();
  group.send(1, "a");
  group.send(3, "b");
  group.send(2, "c");
  group.run(4);
  for (const id of [1, 2, 3]) {
    assert.deepStrictEqual(group.deliveredPayloads(id), ["1:a", "3:b", "2:c"], `member ${id} order`);
    assert.deepStrictEqual(group.checkIntegrity(id), [], `member ${id} integrity`);
  }
  assert.deepStrictEqual(group.agreementViolations(), [], "everybody agrees");
  assert.strictEqual(group.log.length, 3);
  assert.strictEqual(group.log[2].seq, 3, "the sequencer stamps consecutive sequence numbers");
  assert.strictEqual(group.delivered.get(1)[0].round, 1, "a queued message is ordered, multicast and delivered in one round");
  assert.deepStrictEqual(group.trace[0].ordered, [1, 2, 3], "all three queued messages are ordered at once");
  assert.strictEqual(group.stats.blockedRounds, 0, "nothing blocks when everyone is healthy");
  assert.strictEqual(group.stats.retransmissions, 0);

  // FIFO per sender is implied by the sequence numbers, and repeated sends order by arrival
  const fifo = newGroup();
  fifo.send(2, "m1");
  fifo.send(2, "m2");
  fifo.send(1, "m3");
  fifo.run(4);
  assert.deepStrictEqual(fifo.deliveredPayloads(3), ["2:m1", "2:m2", "1:m3"]);
  assert.deepStrictEqual(fifo.delivered.get(1).map((entry) => entry.seq), [1, 2, 3], "sequence order");

  // --- no message creation by a crashed process ---------------------------------------------
  const crashedSender = newGroup();
  crashedSender.send(1, "before-crash");
  crashedSender.crash(2);
  assert.strictEqual(crashedSender.send(2, "from-the-dead"), null, "a crashed member cannot send");
  crashedSender.send(3, "after-crash");
  crashedSender.run(6);
  assert.deepStrictEqual(crashedSender.deliveredPayloads(1), ["1:before-crash", "3:after-crash"]);
  assert.strictEqual(crashedSender.log.length, 2, "the rejected send never entered the log");

  // --- an injected loss is retransmitted, and atomic delivery means everyone waits ------------
  const lossy = newGroup();
  lossy.injectDrop(2, 1, 2);
  lossy.send(1, "first");
  lossy.send(3, "second");
  lossy.run(2);
  assert.ok(lossy.stats.retransmissions >= 2, "the sequencer resends while the loss lasts");
  assert.deepStrictEqual(lossy.deliveredPayloads(2), [], "member 2 cannot deliver out of order");
  assert.deepStrictEqual(lossy.deliveredPayloads(1), [],
    "and nobody else can deliver either: the first message is not stable yet");
  assert.strictEqual(lossy.stats.blockedRounds, 4, "two members wait two rounds each");
  lossy.run(2);
  for (const id of [1, 2, 3]) {
    assert.deepStrictEqual(lossy.deliveredPayloads(id), ["1:first", "3:second"], `member ${id} caught up`);
    assert.strictEqual(lossy.delivered.get(id)[0].round, 3, `member ${id} delivers once the resend arrives`);
  }
  assert.deepStrictEqual(lossy.agreementViolations(), []);
  assert.deepStrictEqual(lossy.delivered.get(1).map((entry) => entry.round), [3, 3], "both released together");

  // --- a crash stalls delivery only for the failure-detector window --------------------------
  const failing = newGroup([1, 2, 3], { failureThreshold: 3 });
  failing.run(2);
  failing.crash(2);
  failing.send(1, "after-crash-1");
  failing.send(3, "after-crash-2");
  const events = failing.run(6);
  const declarations = failing.trace.filter((entry) => entry.declaredFailed.length > 0);
  assert.strictEqual(declarations.length, 1, "the failure detector fires exactly once");
  assert.deepStrictEqual(declarations[0].declaredFailed, [2]);
  assert.strictEqual(declarations[0].round, 5, "after three silent rounds (crash at round 3)");
  const blocked = failing.trace.filter((entry) => entry.blocked.length > 0).map((entry) => entry.round);
  assert.deepStrictEqual(blocked, [3, 4], "delivery waits for the window, then resumes");
  assert.deepStrictEqual(failing.deliveredPayloads(1), ["1:after-crash-1", "3:after-crash-2"],
    "the healthy members make progress");
  assert.deepStrictEqual(failing.deliveredPayloads(3), failing.deliveredPayloads(1));
  assert.deepStrictEqual(failing.delivered.get(1).map((entry) => entry.round), [5, 5], "released together once the detector fires");
  assert.strictEqual(failing.stats.blockedRounds, 4, "two members blocked during two rounds");
  assert.deepStrictEqual(failing.agreementViolations(), [], "the crash never broke agreement");
  assert.deepStrictEqual(events.flatMap((entry) => entry.declaredFailed), [2]);

  // after the declaration the dead member no longer causes work
  const resendsAfter = failing.stats.retransmissions;
  failing.step();
  assert.strictEqual(failing.stats.retransmissions, resendsAfter, "no retransmission to a dead member");
  assert.strictEqual(failing.isExpected(2), false);
  assert.strictEqual(failing.isUp(2), false);

  // --- recovery with state transfer ----------------------------------------------------------
  failing.send(1, "while-down");
  failing.run(2);
  assert.deepStrictEqual(failing.deliveredPayloads(1), ["1:after-crash-1", "3:after-crash-2", "1:while-down"]);
  assert.deepStrictEqual(failing.deliveredPayloads(2), [], "the crashed member is frozen");
  failing.recover(2);
  failing.run(3);
  assert.deepStrictEqual(failing.deliveredPayloads(2), failing.deliveredPayloads(1),
    "after recovery it delivers exactly the agreed sequence");
  assert.strictEqual(failing.deliveredSeqs.get(2).size, 3, "no duplicates during catch-up");
  assert.deepStrictEqual(failing.checkIntegrity(2), []);
  assert.deepStrictEqual(failing.agreementViolations(), []);
  assert.strictEqual(failing.stats.stateTransfers, 1);

  // --- the sequencer crashing keeps safety but stops progress --------------------------------
  const leaderless = newGroup([1, 2, 3]);
  leaderless.send(1, "ordered-before");
  leaderless.run(2);
  leaderless.crash(1);                              // member 1 is the sequencer
  assert.ok(leaderless.send(2, "queued-while-down") !== null,
    "a sender can queue a message, but no sequence number can be assigned");
  leaderless.run(3);
  assert.strictEqual(leaderless.log.length, 1, "the log stays frozen while the sequencer is down");
  for (const id of [2, 3]) {
    assert.deepStrictEqual(leaderless.deliveredPayloads(id), ["1:ordered-before"],
      "already ordered messages are still delivered");
    assert.deepStrictEqual(leaderless.checkIntegrity(id), []);
  }
  assert.deepStrictEqual(leaderless.agreementViolations(), []);
  leaderless.recover(1);
  leaderless.send(3, "ordered-after-recovery");
  leaderless.run(2);
  assert.strictEqual(leaderless.log.length, 3, "a recovered sequencer drains the queue and resumes ordering");
  assert.deepStrictEqual(leaderless.deliveredPayloads(2),
    ["1:ordered-before", "2:queued-while-down", "3:ordered-after-recovery"]);
  assert.deepStrictEqual(leaderless.agreementViolations(), [], "and everybody agrees afterwards");

  // --- empty run and degenerate membership ---------------------------------------------------
  const idle = newGroup();
  idle.run(5);
  for (const id of [1, 2, 3]) assert.deepStrictEqual(idle.delivered.get(id), []);
  assert.strictEqual(idle.log.length, 0);
  assert.strictEqual(idle.stats.blockedRounds, 0);
  const single = newGroup([7]);
  assert.strictEqual(single.sequencer, 7);
  single.send(7, "alone");
  single.run(2);
  assert.deepStrictEqual(single.deliveredPayloads(7), ["7:alone"], "a one-member group is trivial");
  assert.deepStrictEqual(single.agreementViolations(), []);
  assert.throws(() => new SequencerMulticast([]), Error);
  assert.throws(() => new SequencerMulticast([1], { sequencer: 2 }), Error);
  assert.throws(() => newGroup().send(9, "ghost"), Error);
  assert.throws(() => new SequencerMulticast([1], { failureThreshold: 0 }), RangeError);

  // --- determinism ---------------------------------------------------------------------------
  const replayA = newGroup([1, 2, 3], { failureThreshold: 3 });
  const replayB = newGroup([1, 2, 3], { failureThreshold: 3 });
  for (const instance of [replayA, replayB]) {
    instance.injectDrop(3, 2, 1);
    instance.send(1, "x");
    instance.send(2, "y");
    instance.crash(3);
    instance.run(6);
  }
  assert.deepStrictEqual(replayB.deliveredPayloads(1), replayA.deliveredPayloads(1));
  assert.deepStrictEqual(replayB.deliveredPayloads(2), replayA.deliveredPayloads(2));
  assert.deepStrictEqual(replayB.stats, replayA.stats);
  assert.deepStrictEqual(replayB.agreementViolations(), []);

  console.log(`rounds=${failing.trace.length} retransmissions=${failing.stats.retransmissions} declarations=${failing.stats.declarations}`);
  console.log("PASS");
}
