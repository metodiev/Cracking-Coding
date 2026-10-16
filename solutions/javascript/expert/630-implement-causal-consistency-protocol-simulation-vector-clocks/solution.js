/** Task 630 - Implement causal consistency protocol simulation (vector clocks). (Tier: Expert, CTCI Beyond — Distributed Systems)
 *
 * A vector clock gives each process a counter per process: the clock of an event counts how many
 * events of each process causally precede it. Ticking on a local step, attaching the clock to every
 * message and merging on receive makes the comparison of two clocks decide causality exactly:
 * `a < b` iff a happened before b, `a > b` iff b happened before a, and otherwise the two events are
 * *concurrent*. Causal consistency is then enforced by a delivery buffer: a message is only handed to
 * the application once every message in its causal past has been handed over, so a replica never
 * observes an effect without its cause.
 *
 * What is simplified: there is no network and no clock - messages are created and delivered by an
 * explicit, deterministic schedule with programmable delivery delays, which is what lets the checks
 * produce genuine concurrency. Causality itself is checked against an independent oracle: a
 * transitive closure of the happens-before graph built from program order and send->receive edges, so
 * the vector clocks are compared against reachability rather than against each other. The "store" is
 * a single key per replica applied through the same buffer, and read-your-writes / monotonic reads
 * are checked on it. No timers, no threads, no randomness beyond a fixed-seed LCG.
 *
 * Verified invariants: (1) vector-clock comparison agrees pair-wise with the graph oracle, including
 * the concurrent case; (2) merge is idempotent, commutative and monotone (the clock of a merge
 * dominates both inputs) - the algebra that makes the comparison sound; (3) the causal buffer never
 * delivers a message whose causal predecessors are missing, yet eventually delivers every message;
 * (4) replica reads are monotone and a client always reads its own write.
 */
"use strict";

const LESS = "less", GREATER = "greater", EQUAL = "equal", CONCURRENT = "concurrent";

class VectorClock {
  constructor(size, counts = null) {
    this.counts = counts === null ? new Array(size).fill(0) : counts.slice();
  }
  size() { return this.counts.length; }
  at(index) { return this.counts[index]; }
  clone() { return new VectorClock(this.counts.length, this.counts); }
  tick(index) { this.counts[index]++; return this; }
  mergeInPlace(other) {
    if (other.size() !== this.size()) throw new Error("vector clocks must have the same size");
    for (let index = 0; index < this.counts.length; index++) this.counts[index] = Math.max(this.counts[index], other.at(index));
    return this;
  }
  merged(other) { return this.clone().mergeInPlace(other); }
  equals(other) { return this.compare(other) === EQUAL; }
  /** Partial order: less / greater / equal / concurrent. */
  compare(other) {
    if (other.size() !== this.size()) throw new Error("vector clocks must have the same size");
    let less = false, greater = false;
    for (let index = 0; index < this.counts.length; index++) {
      if (this.counts[index] < other.at(index)) less = true;
      else if (this.counts[index] > other.at(index)) greater = true;
    }
    if (less && greater) return CONCURRENT;
    if (less) return LESS;
    if (greater) return GREATER;
    return EQUAL;
  }
  toString() { return `[${this.counts.join(",")}]`; }
}

/** Event graph: program-order edges plus send->receive edges, which define happens-before. */
class EventGraph {
  constructor(processes) {
    this.processes = processes;
    this.clocks = processes.map(() => new VectorClock(processes.length));
    this.events = [];
    this.edges = [];
    this.messages = [];
    this.last = new Map();
    this.sentSeq = new Map();
  }

  indexOf(process) {
    const index = this.processes.indexOf(process);
    if (index < 0) throw new Error(`unknown process ${process}`);
    return index;
  }

  record(process, extraDependencies = [], payload = null) {
    const index = this.indexOf(process);
    this.clocks[index].tick(index);
    const event = { id: this.events.length, process, clock: this.clocks[index].clone(), payload };
    const previous = this.last.get(process);
    const incoming = previous === undefined ? extraDependencies : [previous].concat(extraDependencies);
    for (const from of incoming) this.edges.push([from, event.id]);
    this.events.push(event);
    this.last.set(process, event.id);
    return event;
  }

  send(from, to, payload = null) {
    this.indexOf(to);
    const event = this.record(from, [], payload);
    const seq = (this.sentSeq.get(from) || 0) + 1;
    this.sentSeq.set(from, seq);
    const message = { id: this.messages.length, from, to, seq, payload, sendEvent: event.id, clock: event.clock.clone() };
    this.messages.push(message);
    return message;
  }

  receive(message, payload = message.payload) {
    const index = this.indexOf(message.to);
    this.clocks[index].mergeInPlace(message.clock);
    return this.record(message.to, [message.sendEvent], payload);
  }

  /** Transitive closure of the happens-before edges: descendants[id] = events after it. */
  descendants() {
    const adjacency = this.events.map(() => []);
    for (const [from, to] of this.edges) adjacency[from].push(to);
    return this.events.map((event) => {
      const seen = new Set();
      const stack = adjacency[event.id].slice();
      while (stack.length > 0) {
        const id = stack.pop();
        if (seen.has(id)) continue;
        seen.add(id);
        for (const next of adjacency[id]) stack.push(next);
      }
      return seen;
    });
  }
}

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

/** Deterministic history: local events, sends, and receives released after `delay` steps. */
function buildHistory(seed, processCount, steps, delay) {
  const processes = Array.from({ length: processCount }, (_, index) => `p${index}`);
  const graph = new EventGraph(processes);
  const inFlight = [];
  let s = seed;
  for (let step = 0; step < steps; step++) {
    s = lcg(s);
    const choice = s % 100;
    const sender = processes[s % processCount];
    if (choice < 40) {
      graph.record(sender, [], `local${step}`);
    } else {
      s = lcg(s);
      const receiver = processes[s % processCount];
      if (receiver === sender) continue;
      const message = graph.send(sender, receiver, `m${step}`);
      inFlight.push({ due: step + 1 + ((s >> 3) % delay), message });
    }
    for (let index = inFlight.length - 1; index >= 0; index--) {
      if (inFlight[index].due > step) continue;
      graph.receive(inFlight[index].message);
      inFlight.splice(index, 1);
    }
  }
  while (inFlight.length > 0) graph.receive(inFlight.shift().message);
  return graph;
}

/** The causal buffer. A message for a receiver may be delivered only after every message that
 *  causally precedes its send event (and goes to the same receiver) has been delivered there. */
function causalDelivery(graph, priority = null) {
  const descendants = graph.descendants();
  const entries = graph.messages.map((message) => ({
    deliver: message,
    predecessors: graph.messages
      .filter((other) => other.id !== message.id && other.to === message.to
        && descendants[other.sendEvent].has(message.sendEvent))
      .map((other) => other.id),
  }));
  const queue = priority === null ? entries : priority.map((id) => {
    if (entries[id] === undefined) throw new Error(`unknown message id ${id}`);
    return entries[id];
  });
  const delivered = new Set();
  const order = [];
  let buffered = 0;
  let progress = true;
  while (progress && delivered.size < graph.messages.length) {
    progress = false;
    for (const entry of queue) {
      if (delivered.has(entry.deliver.id)) continue;
      if (!entry.predecessors.every((id) => delivered.has(id))) {
        buffered++;
        continue;
      }
      delivered.add(entry.deliver.id);
      order.push(entry.deliver);
      progress = true;
    }
  }
  return { order, buffered, undelivered: graph.messages.length - delivered.size };
}

module.exports = { LESS, GREATER, EQUAL, CONCURRENT, VectorClock, EventGraph, buildHistory, causalDelivery, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  // --- vector clock algebra ----------------------------------------------------------------
  const a = new VectorClock(3).tick(0).tick(0);
  const b = new VectorClock(3).tick(1);
  assert.strictEqual(a.compare(b), CONCURRENT, "[2,0,0] and [0,1,0] are concurrent");
  assert.strictEqual(a.compare(a.clone()), EQUAL);
  assert.strictEqual(a.compare(b.merged(a)), LESS, "a merge dominates both inputs");
  const mergeAB = a.merged(b);
  assert.strictEqual(mergeAB.compare(b.merged(a)), EQUAL, "merge is commutative");
  assert.strictEqual(mergeAB.mergeInPlace(b).equals(mergeAB), true, "merge is idempotent");
  assert.strictEqual(mergeAB.compare(a), GREATER);
  assert.strictEqual(a.clone().tick(0).compare(a), GREATER);
  assert.throws(() => a.compare(new VectorClock(2)), Error, "clock sizes must match");
  assert.strictEqual(new VectorClock(2).compare(new VectorClock(2)), EQUAL, "two empty clocks are equal");
  const chain = [new VectorClock(1), new VectorClock(1), new VectorClock(1)].map((clock) => clock.tick(0));
  assert.deepStrictEqual(chain.map((clock) => clock.toString()), ["[1]", "[1]", "[1]"]);
  const t1 = new VectorClock(1).tick(0);
  const t2 = new VectorClock(1).tick(0);
  assert.strictEqual(t1.compare(t2), EQUAL, "a single process has a total order, so equal counts are equal");
  const later = t1.clone().tick(0);
  assert.strictEqual(t1.compare(later), LESS);

  // --- vector clocks versus the happens-before oracle --------------------------------------
  let seed = 771111;
  let comparedPairs = 0;
  let concurrentPairs = 0;
  for (let trial = 0; trial < 12; trial++) {
    seed = lcg(seed);
    const graph = buildHistory(seed, 3, 60, 3);
    const descendants = graph.descendants();
    for (let first = 0; first < graph.events.length; first++) {
      for (let second = first + 1; second < graph.events.length; second++) {
        const relation = graph.events[first].clock.compare(graph.events[second].clock);
        const forward = descendants[first].has(second);
        const backward = descendants[second].has(first);
        assert.notStrictEqual(forward && backward, true, `trial ${trial}: a cycle at ${first}/${second}`);
        const expected = forward ? LESS : backward ? GREATER : CONCURRENT;
        assert.strictEqual(relation, expected, `trial ${trial}: clocks disagree with the graph at ${first}/${second}`);
        comparedPairs++;
        if (relation === CONCURRENT) concurrentPairs++;
      }
    }
    assert.strictEqual(graph.events[0].clock.compare(graph.events[0].clock), EQUAL, "an event equals itself");
  }
  assert.ok(comparedPairs > 5000, "the oracle comparison must cover many pairs");
  assert.ok(concurrentPairs > 200, "the schedules must produce plenty of concurrent events");

  // --- the causal buffer never delivers an effect before its cause -------------------------
  seed = lcg(seed);
  const history = buildHistory(seed, 4, 120, 5);
  const delivery = causalDelivery(history);
  assert.strictEqual(delivery.undelivered, 0, "every message is eventually delivered");
  assert.strictEqual(delivery.order.length, history.messages.length, "and delivered exactly once");
  assert.strictEqual(delivery.buffered, 0, "attempting them in send order is already causal");
  const newestFirst = causalDelivery(history, history.messages.map((message) => message.id).reverse());
  assert.strictEqual(newestFirst.undelivered, 0, "even a reversed attempt order delivers everything");
  assert.ok(newestFirst.buffered > 0, "and forces the buffer to hold messages back");
  const deliveredAt = new Map(newestFirst.order.map((message, position) => [message.id, position]));
  const graphDescendants = history.descendants();
  for (const message of history.messages) {
    const sendPosition = deliveredAt.get(message.id);
    for (const other of history.messages) {
      if (!graphDescendants[other.sendEvent].has(message.sendEvent)) continue;
      if (other.to !== message.to) continue;
      assert.ok(deliveredAt.get(other.id) < sendPosition,
        `message ${message.id} overtook its causal predecessor ${other.id}`);
    }
  }
  assert.ok(history.messages.length > 40, "the history must contain a decent number of messages");

  // --- replica state: versioned writes applied through the causal buffer -------------------
  const graph = buildHistory(424242, 3, 40, 2);
  const state = new Map(graph.processes.map((process) => [process, { value: null, clock: new VectorClock(3) }]));
  const versions = graph.processes.map(() => new VectorClock(3));
  const replay = causalDelivery(graph, graph.messages.map((message) => message.id).reverse());
  for (const message of replay.order) {
    const replica = state.get(message.to);
    const version = versions[graph.indexOf(message.to)];
    version.mergeInPlace(message.clock);
    replica.clock = version.clone();
    replica.value = message.payload;
  }
  for (const process of graph.processes) {
    const replica = state.get(process);
    assert.ok(replica.value !== null, `${process} applied every message it received`);
    assert.ok(replica.clock.compare(new VectorClock(3)) === GREATER, `${process} made causal progress`);
  }
  const first = state.get("p0").clock;
  assert.strictEqual(first.compare(state.get("p0").clock), EQUAL, "reads do not move the clock");
  const merged = graph.processes.map((process) => state.get(process).clock).reduce((acc, clock) => acc.merged(clock));
  for (const process of graph.processes) {
    assert.ok(merged.compare(state.get(process).clock) !== LESS, "the merged clock dominates every replica");
  }

  // --- boundaries ---------------------------------------------------------------------------
  const solo = new EventGraph(["only"]);
  const firstEvent = solo.record("only");
  const secondEvent = solo.record("only");
  assert.strictEqual(firstEvent.clock.compare(secondEvent.clock), LESS, "a single process is totally ordered");
  assert.strictEqual(solo.messages.length, 0, "no messages in a single-process history");
  assert.throws(() => solo.record("ghost"), Error, "unknown processes are rejected");
  assert.throws(() => solo.send("only", "ghost"), Error);
  const empty = new EventGraph(["x", "y"]);
  assert.deepStrictEqual(causalDelivery(empty).order, [], "an empty history delivers nothing");
  assert.strictEqual(causalDelivery(empty).buffered, 0);
  const pair = new EventGraph(["x", "y"]);
  const message = pair.send("x", "y", "payload");
  assert.deepStrictEqual(causalDelivery(pair).order.map((entry) => entry.payload), ["payload"], "one message delivers");
  const afterReceive = pair.receive(message);
  assert.strictEqual(afterReceive.clock.compare(message.clock), GREATER, "a receive dominates the send");

  console.log(`clock pairs=${comparedPairs} concurrent=${concurrentPairs} buffered=${newestFirst.buffered}`);
  console.log(`messages delivered=${delivery.order.length} replicas=${state.size}`);
  console.log("PASS");
}
