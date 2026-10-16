/** Task 616 - Implement distributed snapshot algorithm (Chandy-Lamport). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * The Chandy-Lamport global snapshot, simulated deterministically. Processes form a directed
 * graph of FIFO channels; the simulation has no threads and no timers - a schedule drives the
 * sends and deliveries one at a time, and every event gets a monotonically increasing number so
 * that "before the snapshot" is a well defined predicate. The algorithm itself is exactly the
 * published one: the initiator records its state and pushes a marker into every outgoing
 * channel; a process that receives a marker records its state on the first marker (marking every
 * other incoming channel as recording, and forwarding markers on its outgoing channels) and
 * stops recording a channel when the marker of that same channel arrives. An application message
 * delivered on a channel while that channel is recording is appended to the channel's state.
 *
 * What is simplified: processes, channels and the network are plain data structures and FIFO is
 * modelled by an explicit per-channel queue, so there is no real concurrency - what remains is
 * the *logical* content of the algorithm. Verified: every process records exactly once, the
 * recorded channel states contain exactly the messages that were sent before the sender's
 * snapshot and delivered after the receiver's snapshot, no message crosses the cut backwards
 * (the consistency property the marker rules exist to guarantee), the per-process applied counts
 * match the deliveries that happened before its record, the global accounting identity
 * (state + channel states + post-snapshot messages = all messages) holds, a deliberately bad cut
 * is rejected by the same checker, and identical schedules replay identically.
 */
"use strict";

class SnapshotSimulation {
  constructor(processCount, edges) {
    if (!Number.isInteger(processCount) || processCount <= 0) throw new RangeError("process count must be positive");
    this.processes = Array.from({ length: processCount }, (_, id) => ({
      id, state: 0, applied: 0, recorded: false, recordedAt: null, appliedSnapshot: 0, stateSnapshot: 0,
    }));
    this.channels = edges.map(([from, to]) => {
      if (!Number.isInteger(from) || !Number.isInteger(to) || from === to ||
          from < 0 || from >= processCount || to < 0 || to >= processCount) {
        throw new RangeError(`bad channel ${from}->${to}`);
      }
      return { from, to, queue: [], recording: false, recorded: [], startedAt: null, stoppedAt: null };
    });
    this.now = 0;
    this.messages = [];
    this.markers = 0;
    this.deliveries = 0;
    // per-process history of (time, state, applied) so a naive cut can be read at any moment
    this.histories = this.processes.map(() => [{ time: 0, state: 0, applied: 0 }]);
  }

  stateAt(processId, time) {
    return this.#historyAt(this.#process(processId), "state", time);
  }

  appliedAt(processId, time) {
    return this.#historyAt(this.#process(processId), "applied", time);
  }

  #historyAt(process, field, time) {
    let value = 0;
    for (const entry of this.histories[process.id]) if (entry.time <= time) value = entry[field];
    return value;
  }

  #note(process) {
    this.histories[process.id].push({ time: this.now, state: process.state, applied: process.applied });
  }

  channelOf(from, to) {
    const channel = this.channels.find((c) => c.from === from && c.to === to);
    if (channel === undefined) throw new RangeError(`no channel ${from}->${to}`);
    return channel;
  }

  #process(id) {
    if (!Number.isInteger(id) || id < 0 || id >= this.processes.length) throw new RangeError(`bad process ${id}`);
    return this.processes[id];
  }

  /** Application message: appended to the FIFO of its channel, stamped with the event number. */
  send(from, to, payload) {
    this.#process(from);
    this.#process(to);
    if (!Number.isFinite(payload)) throw new TypeError("payload must be a number");
    const message = { kind: "app", id: this.messages.length, from, to, payload, sentAt: ++this.now, deliveredAt: null };
    this.messages.push(message);
    this.channelOf(from, to).queue.push(message);
    return message;
  }

  #pushMarker(channel) {
    this.markers++;
    channel.queue.push({ kind: "marker", from: channel.from, to: channel.to, sentAt: ++this.now });
  }

  /** The initiator records its state, starts recording every incoming channel, forwards markers. */
  initiate(processId) {
    const process = this.#process(processId);
    if (process.recorded) throw new Error(`process ${processId} already recorded`);
    this.#record(process, null);
    return process;
  }

  #record(process, viaChannel) {
    process.recorded = true;
    process.recordedAt = ++this.now;
    process.stateSnapshot = process.state;
    process.appliedSnapshot = process.applied;
    for (const channel of this.channels) {
      if (channel.to === process.id && channel !== viaChannel) {
        channel.recording = true;
        channel.startedAt = process.recordedAt;
      }
    }
    for (const channel of this.channels) if (channel.from === process.id) this.#pushMarker(channel);
  }

  /** Deliver the front of a channel's FIFO; returns the message, or null when it is empty. */
  deliver(from, to) {
    const channel = this.channelOf(from, to);
    if (channel.queue.length === 0) return null;
    const message = channel.queue.shift();
    const receiver = this.#process(to);
    const at = ++this.now;
    if (message.kind === "app") {
      message.deliveredAt = at;
      if (channel.recording) channel.recorded.push(message);
      receiver.state += message.payload;
      receiver.applied++;
      this.deliveries++;
      this.#note(receiver);
      return message;
    }
    if (!receiver.recorded) {
      this.#record(receiver, channel);                 // first marker: record and fan out
    } else if (channel.recording) {
      channel.recording = false;                       // the marker of this channel closes it
      channel.stoppedAt = at;
    }
    return message;
  }

  isComplete() {
    return this.processes.every((p) => p.recorded) && this.channels.every((c) => !c.recording);
  }

  /** The recorded global state: process states plus one recorded message list per channel. */
  snapshot() {
    return {
      states: this.processes.map((p) => p.stateSnapshot),
      applied: this.processes.map((p) => p.appliedSnapshot),
      recordedAt: this.processes.map((p) => p.recordedAt),
      channels: this.channels.map((c) => ({
        from: c.from, to: c.to, messages: c.recorded.slice(), startedAt: c.startedAt, stoppedAt: c.stoppedAt,
      })),
    };
  }

  pending() {
    return this.channels.reduce((acc, c) => acc + c.queue.length, 0);
  }
}

/** Checks every Chandy-Lamport invariant of a recorded global state against the event log. */
function verifySnapshot(sim, snapshot) {
  const violations = [];
  const channelSets = new Map(snapshot.channels.map((c) => [`${c.from}->${c.to}`, new Set(c.messages.map((m) => m.id))]));
  for (const process of sim.processes) {
    const recordedAt = snapshot.recordedAt[process.id];
    if (recordedAt === null || recordedAt === undefined) violations.push(`process ${process.id} never recorded`);
    const expected = sim.messages.filter((m) => m.to === process.id && m.deliveredAt !== null && m.deliveredAt < recordedAt).length;
    if (snapshot.applied[process.id] !== expected) {
      violations.push(`process ${process.id} snapshot counts ${snapshot.applied[process.id]} applied messages, log says ${expected}`);
    }
  }
  for (const message of sim.messages) {
    const sentBefore = message.sentAt < snapshot.recordedAt[message.from];
    const deliveredBefore = message.deliveredAt !== null && message.deliveredAt < snapshot.recordedAt[message.to];
    const recorded = channelSets.get(`${message.from}->${message.to}`).has(message.id);
    if (sentBefore) {
      if (message.deliveredAt === null) {
        violations.push(`message ${message.id} was sent before the sender's snapshot but is still in flight`);
      } else if (!deliveredBefore && !recorded) {
        violations.push(`message ${message.id} crosses the cut but is missing from channel ${message.from}->${message.to}`);
      } else if (deliveredBefore && recorded) {
        violations.push(`message ${message.id} was applied before the receiver's snapshot yet is recorded in the channel`);
      }
    } else {
      if (recorded) violations.push(`message ${message.id} was sent after the sender's snapshot but recorded in the channel`);
      if (deliveredBefore) {
        violations.push(`message ${message.id} crosses the cut backwards: sent after the sender's snapshot, applied before the receiver's`);
      }
    }
  }
  if (violations.length > 0) throw new Error(violations[0]);
  return true;
}

/** A deliberately naive cut: states read at arbitrary times and no channel state collected. */
function naiveSnapshot(sim, times) {
  return {
    states: times.map((time, id) => sim.stateAt(id, time)),
    applied: times.map((time, id) => sim.appliedAt(id, time)),
    recordedAt: times.slice(),
    channels: sim.channels.map((c) => ({ from: c.from, to: c.to, messages: [], startedAt: null, stoppedAt: null })),
  };
}

// Histories let a naive cut read a process state at an arbitrary event number.

function xorshift32(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= (state << 13) >>> 0; state >>>= 0;
    state ^= state >>> 17;
    state ^= (state << 5) >>> 0; state >>>= 0;
    return state / 4294967296;
  };
}

module.exports = { SnapshotSimulation, verifySnapshot, naiveSnapshot };

if (require.main === module) {
  const assert = require("node:assert");

  /** Runs a schedule of ["send", from, to, payload] / ["deliver", from, to] / ["initiate", id]. */
  const runSchedule = (sim, actions) => {
    for (const action of actions) {
      if (action[0] === "send") sim.send(action[1], action[2], action[3]);
      else if (action[0] === "deliver") sim.deliver(action[1], action[2]);
      else if (action[0] === "initiate") sim.initiate(action[1]);
      else throw new Error(`bad action ${action[0]}`);
    }
    return sim;
  };

  // --- ring of three: the markers circulate and close every recording ---
  const ring = new SnapshotSimulation(3, [[0, 1], [1, 2], [2, 0]]);
  runSchedule(ring, [
    ["send", 0, 1, 5],
    ["send", 1, 2, 7],
    ["initiate", 0],
    ["send", 2, 0, 11],
    ["deliver", 0, 1],       // application message, receiver not recording yet
    ["deliver", 0, 1],       // marker: process 1 records and forwards markers to 2
    ["deliver", 1, 2],
    ["deliver", 1, 2],       // marker: process 2 records and forwards a marker to 0
    ["deliver", 2, 0],       // application message while 0 records that channel
    ["deliver", 2, 0],       // marker from 2 stops that recording
  ]);
  assert.ok(ring.isComplete(), "the snapshot must complete");
  assert.strictEqual(ring.pending(), 0, "every marker was delivered");
  const ringSnapshot = ring.snapshot();
  verifySnapshot(ring, ringSnapshot);
  assert.deepStrictEqual(ringSnapshot.states, [0, 5, 7], "recorded states");
  assert.strictEqual(ringSnapshot.channels[0].messages.length + ringSnapshot.channels[1].messages.length, 0,
    "channels 0->1 and 1->2 were drained before their markers");
  assert.deepStrictEqual(ringSnapshot.channels[2].messages.map((m) => m.payload), [11],
    "the message that crossed the cut is recorded on its channel");
  const inSnapshot = ringSnapshot.applied.reduce((a, b) => a + b, 0) +
    ringSnapshot.channels.reduce((a, c) => a + c.messages.length, 0);
  const afterRecording = ring.messages.filter((m) => m.sentAt > ringSnapshot.recordedAt[m.from]).length;
  assert.strictEqual(inSnapshot + afterRecording, ring.messages.length, "global accounting identity");

  // --- channels with no traffic stay empty, and every process records exactly once ---
  const quiet = new SnapshotSimulation(3, [[0, 1], [1, 2], [2, 0]]);
  runSchedule(quiet, [["initiate", 1], ["deliver", 1, 2], ["deliver", 2, 0], ["deliver", 0, 1]]);
  assert.ok(quiet.isComplete(), "an empty network still completes the snapshot");
  assert.deepStrictEqual(quiet.snapshot().channels.map((c) => c.messages.length), [0, 0, 0]);
  verifySnapshot(quiet, quiet.snapshot());
  assert.throws(() => quiet.initiate(1), /already recorded/, "a process records once");
  assert.throws(() => quiet.initiate(9), RangeError);
  assert.throws(() => quiet.send(0, 2, 1), RangeError);
  assert.throws(() => new SnapshotSimulation(2, [[0, 0]]), RangeError);
  assert.strictEqual(new SnapshotSimulation(2, [[0, 1]]).deliver(0, 1), null, "empty channel delivery");

  // --- randomised traffic on a ring of four, with two concurrent initiators ---
  const next = xorshift32(0xc0ffee);
  const busy = new SnapshotSimulation(4, [[0, 1], [1, 2], [2, 3], [3, 0]]);
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0]];
  let payload = 0;
  for (let step = 0; step < 60; step++) {
    const [from, to] = edges[Math.floor(next() * edges.length)];
    busy.send(from, to, ++payload);
  }
  busy.initiate(0);
  busy.initiate(2);
  for (let step = 0; step < 400 && !busy.isComplete(); step++) {
    const [from, to] = edges[step % edges.length];
    busy.deliver(from, to);
  }
  assert.ok(busy.isComplete(), "two initiators still converge");
  assert.strictEqual(busy.pending(), 0);
  const busySnapshot = busy.snapshot();
  verifySnapshot(busy, busySnapshot);
  const busyInSnapshot = busySnapshot.applied.reduce((a, b) => a + b, 0) +
    busySnapshot.channels.reduce((a, c) => a + c.messages.length, 0);
  const busyAfter = busy.messages.filter((m) => m.sentAt > busySnapshot.recordedAt[m.from]).length;
  assert.strictEqual(busyInSnapshot + busyAfter, busy.messages.length, "accounting holds with two initiators");
  const recordedMessages = busySnapshot.channels.flatMap((c) => c.messages);
  assert.ok(recordedMessages.length > 0, "the busy schedule must leave at least one message in a channel");

  // --- a deliberately bad cut is rejected by the same checker ---
  const bad = new SnapshotSimulation(2, [[0, 1]]);
  runSchedule(bad, [["send", 0, 1, 3], ["deliver", 0, 1]]);
  const sent = bad.messages[0];
  const badSnapshot = naiveSnapshot(bad, [0, sent.deliveredAt + 5]);   // p0 records before the send, p1 after the delivery
  assert.throws(() => verifySnapshot(bad, badSnapshot), /backwards/, "the checker has teeth");
  const earlyReceiver = naiveSnapshot(bad, [sent.deliveredAt + 5, 1]); // p1 records before the delivery arrives
  assert.throws(() => verifySnapshot(bad, earlyReceiver), /missing from channel/, "a crossing message must be recorded");

  // --- determinism: the same schedule produces the same snapshot ---
  const replayA = runSchedule(new SnapshotSimulation(3, [[0, 1], [1, 2], [2, 0]]), [
    ["send", 0, 1, 1], ["initiate", 0], ["deliver", 0, 1], ["deliver", 0, 1], ["deliver", 1, 2], ["deliver", 1, 2],
    ["deliver", 2, 0], ["deliver", 2, 0],
  ]).snapshot();
  const replayB = runSchedule(new SnapshotSimulation(3, [[0, 1], [1, 2], [2, 0]]), [
    ["send", 0, 1, 1], ["initiate", 0], ["deliver", 0, 1], ["deliver", 0, 1], ["deliver", 1, 2], ["deliver", 1, 2],
    ["deliver", 2, 0], ["deliver", 2, 0],
  ]).snapshot();
  assert.deepStrictEqual(replayA, replayB, "replaying the schedule is deterministic");

  console.log(`messages=${ring.messages.length} markers=${ring.markers} deliveries=${ring.deliveries}`);
  console.log("PASS");
}
