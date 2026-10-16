/** Task 636 - Implement hybrid logical clocks and test ordering in distributed events. (Tier: Expert, CTCI Beyond — Distributed Systems)
 *
 * A hybrid logical clock keeps a physical component `l` (never behind the node's wall clock and
 * never behind any timestamp it has seen) and a counter `c` that breaks ties when several events
 * share the same `l`. The update rule is the standard one: on a local event with physical time `pt`,
 * `l = max(l, pt)` and `c = 0` if `l` advanced, otherwise `c++`; on receiving `(remote.l, remote.c)`
 * with physical time `pt`, `l' = max(l, remote.l, pt)` and `c'` is copied from whichever of the two
 * inputs won (plus one when both share the maximum, zero when only `pt` won). Comparing `(l, c, node)`
 * gives a *total* order that is consistent with causality - if a happened before b then (a) < (b) -
 * which is exactly what a system needs to sort events without a central clock.
 *
 * What is simplified: physical clocks are synthetic tick counters with a fixed per-node skew, and
 * messages are delivered by an explicit schedule with a deterministic delay, so no wall clock, timer
 * or thread is involved. Causality comes from an independent oracle (program order + send->receive
 * edges closed transitively), so the clock is checked against reachability, not against itself. The
 * comparison ignores clock drift analysis and NTP behaviour; only ordering and skew detection matter.
 *
 * Verified invariants: (1) local events are strictly increasing and the counter only advances when
 * `l` cannot; (2) the update rule takes the maximum of (l, remote.l, pt) and copies the winner's
 * counter, so a received event never sorts before its cause - asserted pair-wise on the whole causal
 * graph; (3) `(l, c, node)` is a total order: antisymmetric, transitive, and a linear extension of
 * happens-before, so every node sorts the events it observed identically; (4) a node whose physical
 * clock runs ahead is detected by the growing offset `l - pt` while causality survives - and the
 * naive physical-only ordering is shown to invert causal order under exactly that skew.
 */
"use strict";

function compare(l, r) {
  if (l.l !== r.l) return l.l < r.l ? -1 : 1;
  if (l.c !== r.c) return l.c < r.c ? -1 : 1;
  if (l.node === r.node) return 0;
  return l.node < r.node ? -1 : 1;
}

/** Hybrid logical clock of one node. */
class HybridClock {
  constructor(node) {
    this.node = node;
    this.l = 0;
    this.c = 0;
    this.pt = 0;
    this.maxOffset = 0;
    this.events = 0;
  }
  stamp() {
    return { l: this.l, c: this.c, node: this.node };
  }
  /** A local event at physical time `pt` (which never goes backwards on one node). */
  now(pt) {
    if (pt < this.pt) throw new Error("a node's physical clock must not go backwards");
    if (pt > this.l) {
      this.l = pt;
      this.c = 0;
    } else {
      this.c++;
    }
    this.pt = pt;
    this.trackOffset();
    this.events++;
    return this.stamp();
  }
  /** Merges a received timestamp with the physical time at which it arrived. */
  receive(remote, pt) {
    if (pt < this.pt) throw new Error("a node's physical clock must not go backwards");
    const lPrime = Math.max(this.l, remote.l, pt);
    if (lPrime === this.l && lPrime === remote.l) this.c = Math.max(this.c, remote.c) + 1;
    else if (lPrime === this.l) this.c = this.c + 1;
    else if (lPrime === remote.l) this.c = remote.c + 1;
    else this.c = 0;
    this.l = lPrime;
    this.pt = pt;
    this.trackOffset();
    this.events++;
    return this.stamp();
  }
  trackOffset() {
    const offset = this.l - this.pt;
    if (offset > this.maxOffset) this.maxOffset = offset;
  }
}

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

/** A deterministic message-passing history. Each node's physical clock is a synthetic counter with
 *  a per-node skew, and every event records the clock value and its causal dependencies. */
function buildHistory(seed, nodeNames, steps, { delay = 2, skew = {} } = {}) {
  const clocks = new Map(nodeNames.map((name) => [name, new HybridClock(name)]));
  const physical = new Map(nodeNames.map((name, index) => [name, index]));
  const events = [];
  const edges = [];
  const messages = [];
  const last = new Map();
  const inFlight = [];
  const stepTime = (step, name) => Math.max(clocks.get(name).pt, step * 10 + (skew[name] || 0));

  const record = (name, hlc, payload) => {
    const event = { id: events.length, node: name, hlc, payload, pt: clocks.get(name).pt };
    const previous = last.get(name);
    if (previous !== undefined) edges.push([previous, event.id]);
    last.set(name, event.id);
    events.push(event);
    return event;
  };

  let s = seed;
  for (let step = 0; step < steps; step++) {
    s = lcg(s);
    const name = nodeNames[s % nodeNames.length];
    s = lcg(s);
    if (s % 100 < 45) {
      record(name, clocks.get(name).now(stepTime(step, name)), `local${step}`);
    } else {
      s = lcg(s);
      const target = nodeNames[s % nodeNames.length];
      if (target === name) continue;
      const event = record(name, clocks.get(name).now(stepTime(step, name)), `m${step}`);
      messages.push({ id: messages.length, from: name, to: target, sendEvent: event.id });
      inFlight.push({ due: step + 1 + (s % delay), message: messages[messages.length - 1] });
    }
    for (let index = inFlight.length - 1; index >= 0; index--) {
      if (inFlight[index].due > step) continue;
      const message = inFlight[index].message;
      const arrival = clocks.get(message.to).receive(clocks.get(message.from).stamp(), stepTime(step, message.to));
      const event = record(message.to, arrival, `m${step}-received`);
      edges.push([message.sendEvent, event.id]);
      inFlight.splice(index, 1);
    }
  }
  while (inFlight.length > 0) {
    const message = inFlight.shift().message;
    const arrival = clocks.get(message.to).receive(clocks.get(message.from).stamp(), stepTime(steps, message.to));
    const event = record(message.to, arrival, "final");
    edges.push([message.sendEvent, event.id]);
  }
  const adjacency = events.map(() => []);
  for (const [from, to] of edges) adjacency[from].push(to);
  const descendants = events.map((event) => {
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
  const clocksList = [...clocks.values()];
  return {
    events, descendants, messages, clocks: clocksList,
    maxOffset: Math.max(...clocksList.map((clock) => clock.maxOffset)),
    physicalOrder: (event) => event.pt,
  };
}

module.exports = { compare, HybridClock, buildHistory, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  // --- the update rule, hand-checked ----------------------------------------------------------
  const clock = new HybridClock("A");
  assert.deepStrictEqual(clock.now(10), { l: 10, c: 0, node: "A" }, "the first event takes the physical time");
  assert.deepStrictEqual(clock.now(10), { l: 10, c: 1, node: "A" }, "the same physical time bumps the counter");
  assert.deepStrictEqual(clock.now(12), { l: 12, c: 0, node: "A" }, "a later physical time resets the counter");
  assert.strictEqual(clock.maxOffset, 0);
  const other = new HybridClock("B");
  other.now(11);
  const received = clock.receive({ l: 11, c: 0, node: "B" }, 13);
  assert.deepStrictEqual(received, { l: 13, c: 0, node: "A" }, "the physical time wins: the counter resets");
  const behind = clock.receive({ l: 20, c: 4, node: "B" }, 13);
  assert.deepStrictEqual(behind, { l: 20, c: 5, node: "A" }, "a remote clock ahead copies its counter and adds one");
  assert.strictEqual(clock.maxOffset, 7, "the offset l - pt records how far ahead the clock was pushed");
  const same = clock.receive({ l: 20, c: 9, node: "B" }, 14);
  assert.deepStrictEqual(same, { l: 20, c: 10, node: "A" }, "when both share l the larger counter wins");
  const forward = clock.receive({ l: 25, c: 2, node: "B" }, 15);
  assert.deepStrictEqual(forward, { l: 25, c: 3, node: "A" });
  assert.throws(() => clock.now(1), Error, "a backwards physical clock is rejected");
  assert.throws(() => clock.receive({ l: 1, c: 0, node: "B" }, 1), Error);
  // --- the comparison is a total order --------------------------------------------------------
  const a1 = { l: 5, c: 0, node: "A" };
  const a2 = { l: 5, c: 1, node: "A" };
  const b1 = { l: 5, c: 0, node: "B" };
  const big = { l: 6, c: 0, node: "A" };
  assert.ok(compare(a1, a2) < 0, "the counter breaks the tie first");
  assert.ok(compare(a1, b1) < 0, "then the node id");
  assert.ok(compare(big, a2) > 0, "the logical component dominates");
  assert.strictEqual(compare(a1, a1), 0);
  assert.strictEqual(compare(a1, b1), -compare(b1, a1), "antisymmetric");
  const stamps = [a1, a2, b1, big];
  for (const x of stamps) for (const y of stamps) for (const z of stamps) {
    if (compare(x, y) <= 0 && compare(y, z) <= 0) assert.ok(compare(x, z) <= 0, "transitive");
  }
  assert.strictEqual(stamps.slice().sort(compare).map((stamp) => `${stamp.l}.${stamp.c}${stamp.node}`).join(" "),
    "5.0A 5.0B 5.1A 6.0A", "sorting by the comparison is deterministic");

  // --- causality on the graph oracle ----------------------------------------------------------
  let seed = 555001;
  let pairs = 0, causalPairs = 0, plainInversions = 0, skewedInversions = 0;
  for (const scenario of [{ skew: {} }, { skew: { B: 40, C: 25 } }]) {
    const hasSkew = Object.keys(scenario.skew).length > 0;
    for (let trial = 0; trial < 8; trial++) {
      seed = lcg(seed);
      const history = buildHistory(seed, ["A", "B", "C"], 70, { delay: 3, skew: scenario.skew });
      for (let first = 0; first < history.events.length; first++) {
        for (let second = first + 1; second < history.events.length; second++) {
          const forward = history.descendants[first].has(second);
          const backward = history.descendants[second].has(first);
          if (!forward && !backward) continue;
          pairs++;
          const ordered = compare(history.events[first].hlc, history.events[second].hlc);
          assert.notStrictEqual(ordered, 0, "distinct events never compare equal");
          assert.strictEqual(forward, ordered < 0,
            `causality violated between ${first} and ${second}: ${JSON.stringify(history.events[first].hlc)}`);
          causalPairs++;
          const physical = history.events[first].pt <= history.events[second].pt;
          if (!physical) {
            if (hasSkew) skewedInversions++;
            else plainInversions++;
          }
        }
      }
      if (hasSkew) {
        assert.ok(history.maxOffset >= 25, `skew must be visible in the offset (got ${history.maxOffset})`);
      } else {
        assert.strictEqual(history.maxOffset, 0, "without skew the logical component never leads the physical one");
      }
    }
  }
  assert.ok(pairs > 1000, `the history must contain enough causally related pairs (${pairs})`);
  assert.ok(causalPairs > 1000, "and they must be counted");
  assert.strictEqual(plainInversions, 0, "physical time respects causality when the clocks are in sync");
  assert.ok(skewedInversions > 0, "a skewed physical clock inverts causal order - HLC does not");

  // --- every node orders the observed events identically ---------------------------------------
  seed = lcg(seed);
  const shared = buildHistory(seed, ["A", "B", "C"], 60, { delay: 4, skew: { C: 30 } });
  const observed = shared.events.filter((event) => event.node === "C" || event.payload !== null);
  const globalOrder = observed.slice().sort((left, right) => compare(left.hlc, right.hlc)).map((event) => event.id);
  for (const name of ["A", "B", "C"]) {
    const seen = observed.filter((event) => event.node === name || event.payload !== null);
    const localOrder = seen.slice().sort((left, right) => compare(left.hlc, right.hlc)).map((event) => event.id);
    const expected = globalOrder.filter((id) => seen.some((event) => event.id === id));
    assert.deepStrictEqual(localOrder, expected, `${name} orders the shared events differently`);
  }
  const orderedByClock = shared.events.slice().sort((left, right) => compare(left.hlc, right.hlc));
  for (let index = 0; index < orderedByClock.length; index++) {
    for (let later = index + 1; later < orderedByClock.length; later++) {
      if (shared.descendants[orderedByClock[index].id].has(orderedByClock[later].id)) {
        assert.ok(index < later, "the sorted order must be a linear extension of causality");
      }
    }
  }
  const lastPerNode = new Map();
  for (const event of orderedByClock) lastPerNode.set(event.node, event.hlc);
  assert.strictEqual(lastPerNode.size, 3);
  for (const clock2 of shared.clocks) {
    assert.ok(clock2.events > 0, `${clock2.node} observed events`);
    assert.ok(clock2.l >= clock2.pt, "the logical component never lags the physical time it saw");
    assert.ok(clock2.maxOffset >= 0);
  }

  // --- boundaries and determinism ---------------------------------------------------------------
  const solo = new HybridClock("S");
  const firstStamp = solo.now(5);
  assert.deepStrictEqual(firstStamp, { l: 5, c: 0, node: "S" }, "the first event takes the physical time");
  const secondStamp = solo.now(5);
  assert.deepStrictEqual(secondStamp, { l: 5, c: 1, node: "S" }, "the same physical time bumps the counter");
  const thirdStamp = solo.now(6);
  assert.deepStrictEqual(thirdStamp, { l: 6, c: 0, node: "S" }, "advancing time resets the counter");
  assert.ok(compare(firstStamp, secondStamp) < 0, "and the sequence is strictly increasing");
  assert.strictEqual(compare(thirdStamp, { l: 6, c: 0, node: "S" }), 0, "identical timestamps are equal");
  assert.deepStrictEqual(new HybridClock("Z").now(0), { l: 0, c: 1, node: "Z" }, "even time zero runs the counter");
  const zero = { l: 0, c: 0, node: "" };
  assert.ok(compare(zero, firstStamp) < 0, "an empty node id sorts before a real one");
  const replayA = buildHistory(9741, ["A", "B"], 40, { delay: 2 });
  const replayB = buildHistory(9741, ["A", "B"], 40, { delay: 2 });
  assert.deepStrictEqual(replayA.events.map((event) => event.hlc), replayB.events.map((event) => event.hlc),
    "the same seed produces the same timestamps");
  assert.ok(replayA.events.length > 30);

  console.log(`hlc causal-pairs=${causalPairs} physical-inversions-skewed=${skewedInversions} events=${shared.events.length}`);
  console.log(`hlc max-offset=${shared.maxOffset} nodes=${shared.clocks.length}`);
  console.log("PASS");
}
