/** Task 760 - Implement distributed snapshot-based auditing for integrity. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A Chandy-Lamport snapshot over a scripted distributed run, plus the integrity audit that makes
 * the captured state trustworthy. Each process keeps an append-only log and a hash chain
 * `head_i = mix(head_{i-1} ^ hash(record_i))`, so one edited record changes every later link and
 * the first divergent link index pinpoints it (a 32-bit unkeyed avalanche: a checksum, not a
 * cryptographic hash). The protocol is the classic one: the initiator records its own state and
 * pushes a MARKER into each outgoing channel; the first marker a process sees makes it record its
 * state, start capturing every incoming channel and forward markers; a channel's capture ends when
 * its own marker arrives, and every data message delivered meanwhile becomes an in-flight message
 * of that channel state. FIFO channels make the capture exact: a message is captured iff it was
 * *sent* inside the sender's recorded prefix and *received* outside the receiver's.
 * The audit re-derives everything independently of the capture code: the cut is the union of the
 * recorded event ids, its consistency is checked against the causal predecessors (local order plus
 * message links) built from the real execution, and the captured in-flight set must equal the set
 * computed from the final history. Phantom, missing and duplicated in-flight messages, a message
 * crossing the cut backwards, and any edited record are all reported by kind.
 *
 * What is simplified: the network is a scripted FIFO event list, the chain is 32-bit and unkeyed,
 * processes never fail, and there is no checkpoint store nor a stop-the-world comparison. What is
 * verified: a quiet snapshot reproduces the true state; with traffic the captured in-flight set
 * matches the computed one exactly and the cut passes the causal-closure test; a later snapshot
 * refines the earlier one monotonically; every single-record edit is detected at that record;
 * dropping, duplicating or inventing an in-flight message is reported; markers are sent once per
 * channel; sent = received + in-flight is conserved; and runs are reproducible.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

/** 32-bit avalanche used as the (toy) chain compression function. */
function mix32(value) {
  let hash = value >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x7feb352d) >>> 0;
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 0x846ca68b) >>> 0;
  hash ^= hash >>> 16;
  return hash >>> 0;
}
function hashRecord(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) hash = (Math.imul(hash, 0x01000193) ^ text.charCodeAt(i)) >>> 0;
  return hash >>> 0;
}
const chainOf = (records) => records.reduce((head, record) => mix32(head ^ hashRecord(record)), 1);
const channelKey = (from, to) => `${from}->${to}`;
const ringEdges = (n) => Array.from({ length: n }, (_, id) => [id, (id + 1) % n]);
const allEdges = (n) => Array.from({ length: n }, (_, from) => Array.from({ length: n }, (_, to) => [from, to]))
  .flat().filter(([from, to]) => from !== to);
class SnapshotSimulation {
  constructor(processCount, edges = ringEdges(processCount)) {
    if (!Number.isInteger(processCount) || processCount < 1) throw new RangeError("processCount must be >= 1");
    this.n = processCount;
    this.edges = edges.map((edge) => [...edge]);
    this.processes = Array.from({ length: processCount }, (_, id) => ({ id, records: [], chain: [1], eventIds: [] }));
    this.channels = new Map(this.edges.map(([from, to]) => [channelKey(from, to), []]));
    this.messages = new Map();
    this.events = [];
    this.links = new Map();
    this.nextMessageId = 1;
    this.snapshot = null;
    this.markersSent = 0;
    this.snapshotsTaken = 0;
    this.delivered = [];
  }
  #record(processId, text, receiveOf = null) {
    const process = this.processes[processId];
    const eventId = this.events.length;
    const predecessors = new Set();
    if (process.eventIds.length > 0) predecessors.add(process.eventIds[process.eventIds.length - 1]);
    if (receiveOf !== null) predecessors.add(receiveOf);
    process.records.push(text);
    process.chain.push(mix32(process.chain[process.chain.length - 1] ^ hashRecord(text)));
    process.eventIds.push(eventId);
    this.events.push({ id: eventId, process: processId, text });
    this.links.set(eventId, predecessors);
    return eventId;
  }
  /** Captures the process prefix; that prefix is what the audit has to verify later. */
  #capture(processId) {
    const process = this.processes[processId];
    const state = {
      process: processId, records: process.records.slice(), chain: process.chain.slice(),
      head: process.chain[process.chain.length - 1], eventIds: process.eventIds.slice(),
    };
    this.snapshot.states.set(processId, state);
    this.snapshot.recording.add(processId);
    for (const key of this.channels.keys()) if (key.endsWith(`->${processId}`)) this.snapshot.captureActive.set(key, true);
    return state;
  }
  #sendMarkers(processId) {
    for (const [from, to] of this.edges) {
      if (from !== processId) continue;
      const message = { id: this.nextMessageId++, type: "marker", from, to, payload: "marker" };
      this.messages.set(message.id, message);
      this.channels.get(channelKey(from, to)).push(message);
      this.markersSent++;
    }
  }
  initiate(processId) {
    if (!this.processes[processId]) throw new RangeError(`unknown process ${processId}`);
    this.snapshot = {
      initiator: processId,
      states: new Map(),
      channels: new Map([...this.channels.keys()].map((key) => [key, []])),
      captureActive: new Map(),
      recording: new Set(),
      markersReceived: 0,
      markersSentAtCapture: this.markersSent,
    };
    this.snapshotsTaken++;
    this.#capture(processId);
    this.#sendMarkers(processId);
    return this.snapshot;
  }
  send(from, to, payload) {
    const key = channelKey(from, to);
    if (!this.channels.has(key)) throw new RangeError(`no channel ${key}`);
    const message = { id: this.nextMessageId++, type: "data", from, to, payload };
    message.sendEventId = this.#record(from, `send:${payload}#${message.id}`);
    message.receiveEventId = null;
    this.messages.set(message.id, message);
    this.channels.get(key).push(message);
    return message;
  }
  internal(processId, payload) { return this.#record(processId, `internal:${payload}`); }
  /** Delivers the oldest message of a channel and returns it. */
  deliver(from, to) {
    const key = channelKey(from, to);
    if (!this.channels.has(key)) throw new RangeError(`no channel ${key}`);
    const message = this.channels.get(key).shift();
    if (message === undefined) throw new RangeError(`channel ${key} is empty`);
    this.delivered.push(message.id);
    if (message.type === "marker") {
      if (this.snapshot === null) return message;
      this.snapshot.markersReceived++;
      if (!this.snapshot.recording.has(to)) { this.#capture(to); this.#sendMarkers(to); }
      this.snapshot.captureActive.set(key, false);
      return message;
    }
    message.receiveEventId = this.#record(to, `recv:${message.payload}#${message.id}`, message.sendEventId);
    if (this.snapshot !== null && this.snapshot.captureActive.get(key) === true) this.snapshot.channels.get(key).push(message);
    return message;
  }
  run(schedule) {
    for (const step of schedule) {
      if (step.type === "send") this.send(step.from, step.to, step.payload);
      else if (step.type === "deliver") this.deliver(step.from, step.to);
      else if (step.type === "internal") this.internal(step.at, step.payload);
      else if (step.type === "initiate") this.initiate(step.at);
      else throw new TypeError(`unknown step '${step.type}'`);
    }
    return this;
  }
  cutOf(snapshot = this.snapshot) {
    const cut = new Set();
    for (const state of snapshot.states.values()) for (const eventId of state.eventIds) cut.add(eventId);
    return cut;
  }
  /** Cut consistency: the whole causal past of every event in the cut is inside the cut. */
  isConsistentCut(cut) {
    const seen = new Set(cut);
    const pending = [...cut];
    while (pending.length > 0) {
      for (const predecessor of this.links.get(pending.pop()) ?? []) {
        if (!cut.has(predecessor)) return { ok: false, missing: predecessor };
        if (!seen.has(predecessor)) { seen.add(predecessor); pending.push(predecessor); }
      }
    }
    return { ok: true };
  }
  /** In-flight messages at the cut, computed from the final history (never from the capture). */
  inFlightAtCut(snapshot = this.snapshot) {
    const cut = this.cutOf(snapshot);
    const inFlight = new Map([...this.channels.keys()].map((key) => [key, new Set()]));
    for (const message of this.messages.values()) {
      if (message.type === "marker" || !cut.has(message.sendEventId)) continue;
      if (message.receiveEventId !== null && cut.has(message.receiveEventId)) continue;
      inFlight.get(channelKey(message.from, message.to)).add(message.id);
    }
    return inFlight;
  }
  /** Conservation law over the whole run: every sent message is delivered or still in a channel. */
  accounting() {
    const data = [...this.messages.values()].filter((message) => message.type === "data");
    const queued = [...this.channels.values()].reduce((total, queue) =>
      total + queue.filter((message) => message.type === "data").length, 0);
    const received = data.filter((message) => message.receiveEventId !== null).length;
    return { sent: data.length, received, queued, balanced: data.length === received + queued };
  }
}
/** The integrity audit of a captured snapshot. */
function auditSnapshot(simulation, snapshot = simulation.snapshot) {
  const violations = [];
  // (1) every recorded prefix must recompute to the recorded chain, link by link
  for (const state of snapshot.states.values()) {
    let head = 1;
    let firstDivergence = -1;
    for (let index = 0; index < state.records.length; index++) {
      head = mix32(head ^ hashRecord(state.records[index]));
      if (firstDivergence < 0 && (state.chain[index + 1] !== head || state.records[index] !== simulation.processes[state.process].records[index])) {
        firstDivergence = index;
      }
    }
    if (head !== state.head) violations.push({ kind: "chain-mismatch", process: state.process, firstDivergentRecord: firstDivergence });
  }
  // (2) the cut must be causally closed, and no captured message may cross it backwards
  const cut = simulation.cutOf(snapshot);
  const consistency = simulation.isConsistentCut(cut);
  if (!consistency.ok) violations.push({ kind: "inconsistent-cut", missing: consistency.missing });
  for (const [key, captured] of snapshot.channels) {
    for (const message of captured) {
      if (!cut.has(message.sendEventId)) violations.push({ kind: "message-sent-after-cut", channel: key, message: message.id });
      if (message.receiveEventId !== null && cut.has(message.receiveEventId)) {
        violations.push({ kind: "message-received-before-cut", channel: key, message: message.id });
      }
    }
  }
  // (3) the captured in-flight set must be exactly the set the history says was in flight
  const computed = simulation.inFlightAtCut(snapshot);
  for (const [key, captured] of snapshot.channels) {
    const expected = computed.get(key) ?? new Set();
    const seen = new Set();
    for (const message of captured) {
      if (seen.has(message.id)) violations.push({ kind: "duplicate-in-flight", channel: key, message: message.id });
      seen.add(message.id);
      if (!expected.has(message.id)) violations.push({ kind: "phantom-in-flight", channel: key, message: message.id });
    }
    for (const id of expected) if (!captured.some((message) => message.id === id)) violations.push({ kind: "missing-in-flight", channel: key, message: id });
  }
  return { violations, ok: violations.length === 0, cut, inFlight: computed };
}
/** Deep copy of a snapshot, so the tests can corrupt it without touching the original. */
const cloneSnapshot = (snapshot) => structuredClone(snapshot);
module.exports = { mix32, hashRecord, chainOf, channelKey, ringEdges, allEdges, SnapshotSimulation, auditSnapshot, cloneSnapshot };

if (require.main === module) {
  const script = (steps) => steps.map(([type, a, b, payload]) => (type === "send" ? { type, from: a, to: b, payload }
    : type === "deliver" ? { type, from: a, to: b } : { type, at: a, payload: b }));
  const traffic = script([
    ["internal", 0, "boot"], ["send", 0, 1, "read"], ["internal", 1, "cache"], ["deliver", 0, 1],
    ["send", 2, 1, "replica"], ["initiate", 0],
    ["deliver", 0, 1],                                     // the marker: process 1 records its state here
    ["send", 2, 1, "late"],
    ["deliver", 1, 2],                                     // the marker: process 2 records its state
    ["deliver", 2, 0], ["deliver", 2, 1], ["deliver", 2, 1],   // "replica" and "late" arrive inside the capture window of 2->1
    ["deliver", 2, 1],                                     // the marker on 2->1 closes that window
    ["deliver", 1, 0], ["deliver", 0, 2],
    ["send", 2, 1, "post-snapshot"], ["deliver", 2, 1],    // after the cut: received, never captured
  ]);
  // 1. a snapshot taken on a quiet system reproduces the true state, with nothing in flight
  const quiet = new SnapshotSimulation(3);
  quiet.internal(0, "a");
  quiet.send(0, 1, "hello");
  quiet.deliver(0, 1);
  quiet.internal(2, "b");
  const quietTruth = quiet.processes.map((process) => process.records.slice());
  quiet.initiate(0);
  for (const step of [[0, 1], [1, 2], [2, 0]]) quiet.deliver(...step);
  const quietAudit = auditSnapshot(quiet);
  ok(quietAudit.ok, `a quiet snapshot must audit clean (${JSON.stringify(quietAudit.violations)})`);
  deepStrictEqual([...quiet.snapshot.states.values()].map((state) => state.records), quietTruth, "the captured state is the real one");
  for (const [, captured] of quiet.snapshot.channels) deepStrictEqual(captured, [], "no channel is holding a message");

  // 2. with traffic, one data message is genuinely in flight at the cut and is captured exactly once
  const bus = new SnapshotSimulation(3, allEdges(3));
  bus.run(traffic);
  const busAudit = auditSnapshot(bus);
  ok(busAudit.ok, `the traffic snapshot must audit clean (${JSON.stringify(busAudit.violations)})`);
  const capturedLate = bus.snapshot.channels.get("2->1").map((message) => message.payload);
  deepStrictEqual(capturedLate, ["replica", "late"], "both in-flight messages are captured on their channel");
  strictEqual(busAudit.inFlight.get("2->1").size, 2, "exactly two messages were in flight on 2->1");
  // 3. the cut is causally closed and contains no backwards crossing
  strictEqual(bus.isConsistentCut(busAudit.cut).ok, true);
  deepStrictEqual(busAudit.violations.filter((violation) => violation.kind.endsWith("cut")), []);
  // 4. conservation: sent = received + queued over the whole run
  const busAccounting = bus.accounting();
  ok(busAccounting.balanced, `sent=${busAccounting.sent} received=${busAccounting.received} queued=${busAccounting.queued}`);
  // 5. every single-record edit is detected, at exactly that record
  const original = bus.snapshot.states.get(1).records.slice();
  for (let index = 0; index < original.length; index++) {
    const corrupted = cloneSnapshot(bus.snapshot);
    corrupted.states.get(1).records[index] = `${original[index]}#tampered`;
    const mismatch = auditSnapshot(bus, corrupted).violations.find((violation) => violation.kind === "chain-mismatch");
    strictEqual(mismatch?.firstDivergentRecord, index, `editing record ${index} must be detected at that index`);
  }

  // 6. corrupted in-flight sets are reported by kind
  const corrupt = (mutate) => {
    const copy = cloneSnapshot(bus.snapshot);
    mutate(copy);
    return auditSnapshot(bus, copy).violations.map((violation) => violation.kind);
  };
  deepStrictEqual(corrupt((copy) => copy.channels.set("2->1", [])),
    new Array(capturedLate.length).fill("missing-in-flight"), "every lost in-flight message is reported");
  deepStrictEqual(corrupt((copy) => copy.channels.get("2->1").push(copy.channels.get("2->1")[0])), ["duplicate-in-flight"]);
  deepStrictEqual(corrupt((copy) => copy.channels.get("2->1").push({ id: 4242, type: "data", from: 2, to: 1, payload: "ghost", sendEventId: 0, receiveEventId: null })),
    ["phantom-in-flight"], "an invented in-flight message is reported");

  // 7. a later snapshot is a monotone refinement of the earlier one and is also consistent
  const later = new SnapshotSimulation(3, allEdges(3));
  later.run(traffic);
  later.initiate(1);
  for (const step of [[1, 0], [1, 2], [2, 0], [2, 1], [0, 1], [0, 2]]) later.deliver(...step);
  const laterAudit = auditSnapshot(later);
  ok(laterAudit.ok, `the second snapshot must audit clean (${JSON.stringify(laterAudit.violations)})`);
  for (const [id, state] of later.snapshot.states) {
    const earlier = bus.snapshot.states.get(id);
    deepStrictEqual(state.records.slice(0, earlier.records.length), earlier.records, `process ${id} refines its earlier prefix`);
  }
  ok(later.cutOf(later.snapshot).size >= later.cutOf(bus.snapshot).size, "the later cut contains the earlier one");

  // 8. marker accounting: one marker per channel, forwarded exactly once, duplicates ignored
  strictEqual(quiet.markersSent, 3, "a ring of three processes sends three markers");
  strictEqual(quiet.snapshot.markersReceived, 3, "and receives three");
  const duplicates = new SnapshotSimulation(3);
  duplicates.initiate(0);
  for (const step of [[0, 1], [1, 2]]) duplicates.deliver(...step);
  const capturedBefore = duplicates.snapshot.channels.get("2->0").length;
  duplicates.deliver(2, 0);
  duplicates.channels.get("2->0").push({ id: 999, type: "marker", from: 2, to: 0 });
  const repeated = duplicates.deliver(2, 0);
  strictEqual(repeated.id, 999, "the repeated marker is delivered");
  strictEqual(duplicates.snapshot.channels.get("2->0").length, capturedBefore, "a repeated marker leaves the captured state alone");

  // 9. boundary: a single-process system, and input validation
  const alone = new SnapshotSimulation(1, []);
  alone.internal(0, "only");
  alone.initiate(0);
  const aloneAudit = auditSnapshot(alone);
  ok(aloneAudit.ok, "a single process audits clean");
  strictEqual(alone.snapshot.states.size, 1);
  strictEqual(alone.snapshot.channels.size, 0, "one process has no channels");
  strictEqual(alone.isConsistentCut(new Set()).ok, true, "the empty cut is consistent");
  throws(() => new SnapshotSimulation(0), RangeError);
  throws(() => alone.send(0, 1, "nowhere"), RangeError);
  throws(() => alone.deliver(0, 0), RangeError);
  throws(() => new SnapshotSimulation(2).run([{ type: "teleport" }]), TypeError);
  // 10. determinism: the same script produces the same capture
  const digest = (simulation) => [...simulation.snapshot.states.values()]
    .map((state) => `${state.process}:${state.head}:${state.eventIds.join(",")}`).join("|");
  const first = new SnapshotSimulation(3, allEdges(3)).run(traffic);
  const second = new SnapshotSimulation(3, allEdges(3)).run(traffic);
  strictEqual(digest(first), digest(second), "same script, same snapshot");
  console.log(
    `processes=3 inFlightAtCut=${[...busAudit.inFlight.values()].reduce((total, set) => total + set.size, 0)} ` +
    `cutSize=${busAudit.cut.size} markers=${quiet.markersSent} tamperChecks=${original.length}`);
  console.log("PASS");
}
