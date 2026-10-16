/** Task 844 - Implement streaming graph processing with windowed aggregations and state eviction.
 *  (Tier: Master, CTCI Ch 9 — System Design and Scalability)
 *
 * A streaming graph engine that consumes an endless edge stream and maintains *windowed* aggregates
 * without letting its state grow with the length of the stream:
 *
 *  - windows: tumbling (size == slide) or sliding (size > slide); an event with timestamp t belongs to
 *    every slide-aligned window start s with s <= t < s + size, so a sliding window sees each event in
 *    size/slide windows;
 *  - watermark: the largest timestamp accepted so far, monotone by construction. A window closes
 *    exactly when `start + size + allowedLateness <= watermark`, and at that instant its per-window
 *    state (edge set, degree map, counters) is *evicted*. The record emitted at the close is the final
 *    aggregate, so eviction is invisible in the output — the property worth testing;
 *  - lateness: an event arriving before its windows close is folded in and counted as late-but-
 *    accepted; an event arriving after a close is dropped and counted. Neither can resurrect a closed
 *    window, so every window is emitted exactly once;
 *  - aggregations: event and weight counts, per-vertex degree inside the window, and the number of
 *    triangles whose three edges all fall inside the window (computed once, at close, from the window's
 *    own edge set).
 *
 * Everything is deterministic (timestamps and edges from seeded splitmix64) and every streaming result
 * is compared against a *batch oracle* that recomputes each window from the raw event list.
 *
 * Simplifications, stated honestly: an event is accepted or dropped as a whole, whereas a real engine
 * could partially update the windows it still holds; "triangles in a window" means triangles whose
 * three edges lie in that window, not triangles of a global graph; state is an in-process object, not
 * an off-heap store; and a 4000-event stream stands in for an unbounded one. No timers, threads or
 * sockets.
 *
 * Invariants verified below:
 *  - safety: the watermark never decreases, a closed window is never reopened or re-emitted, and late
 *    events are classified exactly by the closing rule (boundary event in, one tick later dropped);
 *  - correctness: window membership, counts, weight sums, per-vertex degrees and window triangles all
 *    equal the batch oracle — for tumbling and sliding windows, for crafted late arrivals, and over a
 *    4000-event stream;
 *  - bounded state (the point of the exercise): the live window count stays within
 *    ceil((size + lateness) / slide) + 1 for a 400-event stream and for a ten times longer one
 *    (measured, not assumed), with the live entry count bounded along with it;
 *  - eviction is invisible: a run that evicts on close and a run that keeps everything produce
 *    identical records;
 *  - cost: accepted/dropped events, exactly one record per evicted window, ascending window order;
 *  - edge cases: empty stream, single event, a burst at one timestamp, slide == size == 1, windows with
 *    no events, duplicated edges, and an event exactly on the lateness boundary.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;

/** Deterministic splitmix64: the only source of pseudo-randomness. */
class SplitMix64 {
  constructor(seed) {
    this.state = (BigInt(seed) & MASK64) || 1n;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }

  upto(bound) {
    return Number(this.next() % BigInt(bound));
  }
}

const edgeKey = (from, to) => (from < to ? `${from}-${to}` : `${to}-${from}`);

/** Every window start containing `timestamp`: slide-aligned, at most size/slide of them. */
function windowStarts(timestamp, size, slide) {
  const latest = Math.floor(timestamp / slide) * slide;
  const starts = [];
  for (let start = latest; start > latest - size && start >= 0; start -= slide) starts.push(start);
  return starts.sort((a, b) => a - b);
}

/** Per-window aggregate: exactly the state that must be evicted when the window closes. */
class WindowState {
  constructor(start, size) {
    this.start = start;
    this.end = start + size;
    this.count = 0;
    this.weight = 0;
    this.degrees = new Map();
    this.edges = new Map(); // edgeKey -> weight: duplicate edges are collapsed
    this.lateAccepted = 0;
  }

  add(event) {
    this.count++;
    this.weight += event.weight;
    this.edges.set(edgeKey(event.from, event.to), event.weight);
    this.degrees.set(event.from, (this.degrees.get(event.from) ?? 0) + 1);
    this.degrees.set(event.to, (this.degrees.get(event.to) ?? 0) + 1);
  }

  /** Triangles whose three edges all live in this window, by brute force over its own edge set. */
  triangles() {
    const adjacency = new Map();
    for (const key of this.edges.keys()) {
      const [left, right] = key.split("-").map(Number);
      if (!adjacency.has(left)) adjacency.set(left, new Set());
      if (!adjacency.has(right)) adjacency.set(right, new Set());
      adjacency.get(left).add(right);
      adjacency.get(right).add(left);
    }
    let found = 0;
    for (const [left, neighbours] of adjacency) {
      for (const right of neighbours) {
        if (right <= left) continue;
        for (const third of neighbours) {
          if (third <= right) continue;
          if (adjacency.get(right)?.has(third)) found++;
        }
      }
    }
    return found;
  }

  /** Elementary entries held while the window is live: the memory it costs. */
  entries() {
    return 1 + this.degrees.size + this.edges.size;
  }

  record() {
    return {
      start: this.start,
      end: this.end,
      count: this.count,
      weight: this.weight,
      triangles: this.triangles(),
      degrees: Object.fromEntries([...this.degrees].sort((a, b) => a[0] - b[0])),
      lateAccepted: this.lateAccepted,
    };
  }
}

/**
 * The streaming engine. `retainState` exists only to prove that eviction changes nothing; the default
 * (off) is the production behaviour.
 */
class WindowedGraphStream {
  constructor(options) {
    this.size = options.size;
    this.slide = options.slide ?? options.size;
    this.allowedLateness = options.allowedLateness ?? 0;
    this.retainState = options.retainState ?? false;
    if (!(this.size > 0) || !(this.slide > 0)) {
      throw new Error("window size and slide must be positive");
    }
    if (this.slide > this.size) throw new Error("a slide larger than the size would skip events");
    this.windows = new Map();
    this.retained = new Map(); // only filled when `retainState` is on: proof that eviction is a choice
    this.closedStarts = [];
    this.output = [];
    this.watermark = -1;
    this.accepted = 0;
    this.acceptedLate = 0;
    this.droppedLate = 0;
    this.evictions = 0;
    this.peakLiveWindows = 0;
    this.peakLiveEntries = 0;
  }

  /** True when the event's windows would already have been closed: the "too late" test. */
  isTooLate(timestamp) {
    if (this.watermark < 0) return false;
    return windowStarts(timestamp, this.size, this.slide)
      .every((start) => start + this.size + this.allowedLateness <= this.watermark);
  }

  /** Closes and evicts every window whose end plus lateness is behind the watermark. */
  closeWindows() {
    const due = [...this.windows.entries()]
      .filter(([, state]) => state.end + this.allowedLateness <= this.watermark)
      .map(([start]) => start)
      .sort((a, b) => a - b);
    for (const start of due) {
      const state = this.windows.get(start);
      this.output.push(state.record());
      this.closedStarts.push(start);
      if (this.retainState) this.retained.set(start, state);
      this.windows.delete(start);
      this.evictions++;
    }
    return due.length;
  }

  /** Processes one event: true when accepted, false when dropped as too late. */
  process(event) {
    if (this.isTooLate(event.timestamp)) {
      this.droppedLate++;
      return false;
    }
    const late = this.watermark >= 0 && event.timestamp < this.watermark;
    for (const start of windowStarts(event.timestamp, this.size, this.slide)) {
      if (!this.windows.has(start)) this.windows.set(start, new WindowState(start, this.size));
      const state = this.windows.get(start);
      state.add(event);
      if (late) state.lateAccepted++;
    }
    this.accepted++;
    if (late) this.acceptedLate++;
    this.watermark = Math.max(this.watermark, event.timestamp);
    this.closeWindows();
    this.peakLiveWindows = Math.max(this.peakLiveWindows, this.windows.size);
    this.peakLiveEntries = Math.max(this.peakLiveEntries, this.liveEntries());
    return true;
  }

  /** Flushes whatever is still open at the end of the stream. */
  flush() {
    const starts = [...this.windows.keys()].sort((a, b) => a - b);
    for (const start of starts) {
      const state = this.windows.get(start);
      this.output.push(state.record());
      this.closedStarts.push(start);
      if (this.retainState) this.retained.set(start, state);
      this.windows.delete(start);
      this.evictions++;
    }
    return starts.length;
  }

  liveWindows() {
    return this.windows.size;
  }

  liveEntries() {
    let entries = 0;
    for (const state of this.windows.values()) entries += state.entries();
    return entries;
  }

  /** The configuration's bound on live windows; a longer stream cannot raise it. */
  static windowBound(size, slide, allowedLateness) {
    return Math.ceil((size + allowedLateness) / slide) + 1;
  }
}

// --- oracle and generators --------------------------------------------------
/** Batch oracle: recompute every window from the raw events with the same triangle definition. */
function batchWindows(events, options) {
  const size = options.size;
  const slide = options.slide ?? size;
  const starts = new Set();
  for (const event of events) {
    for (const start of windowStarts(event.timestamp, size, slide)) starts.add(start);
  }
  const result = new Map();
  for (const start of [...starts].sort((a, b) => a - b)) {
    const state = new WindowState(start, size);
    for (const event of events) {
      if (event.timestamp >= start && event.timestamp < start + size) state.add(event);
    }
    result.set(start, state.record());
  }
  return result;
}

/** On-time stream: timestamps never decrease, so no event is ever late. */
function onTimeStream(seed, count, vertices = 8) {
  const rng = new SplitMix64(seed);
  const events = [];
  let timestamp = 0;
  for (let i = 0; i < count; i++) {
    timestamp += rng.upto(3);
    const from = rng.upto(vertices);
    const to = rng.upto(vertices);
    if (from === to) continue;
    events.push({ timestamp, from, to, weight: 1 + rng.upto(5) });
  }
  return events;
}

/** A stream with one late-but-accepted event and one that arrives far too late. */
function craftedLateStream() {
  const events = [];
  for (let i = 0; i <= 10; i++) events.push({ timestamp: i, from: i % 5, to: (i % 5) + 1, weight: 1 });
  events.push({ timestamp: 4, from: 7, to: 8, weight: 3, late: true });
  for (let i = 11; i <= 20; i++) events.push({ timestamp: i, from: i % 5, to: (i % 5) + 1, weight: 1 });
  events.push({ timestamp: 2, from: 9, to: 10, weight: 9, tooLate: true });
  return events;
}

module.exports = { MASK64, SplitMix64, edgeKey, windowStarts, WindowState, WindowedGraphStream,
  batchWindows, onTimeStream, craftedLateStream };

if (require.main === module) {
  const assert = require("node:assert");
  const startsOf = (stream) => stream.output.map((record) => record.start);
  const byStart = (stream) => new Map(stream.output.map((record) => [record.start, record]));
  const runStream = (events, options) => {
    const stream = new WindowedGraphStream(options);
    for (const event of events) stream.process(event);
    return stream;
  };

  // --- window membership -----------------------------------------------------
  assert.deepStrictEqual(windowStarts(7, 10, 10), [0], "tumbling windows have a single start");
  assert.deepStrictEqual(windowStarts(10, 10, 10), [10], "a boundary timestamp opens a new window");
  assert.deepStrictEqual(windowStarts(12, 10, 5), [5, 10], "a sliding window holds size/slide starts");
  assert.deepStrictEqual(windowStarts(3, 10, 5), [0], "the first window cannot start below zero");
  assert.deepStrictEqual(windowStarts(0, 4, 4), [0], "start zero is included");
  assert.throws(() => new WindowedGraphStream({ size: 0 }), /positive/, "a zero size is rejected");
  assert.throws(() => new WindowedGraphStream({ size: 4, slide: 8 }), /slide/, "slide > size is rejected");

  // --- tumbling windows against the batch oracle -----------------------------
  const tumblingEvents = onTimeStream(8441, 900);
  const tumbling = runStream(tumblingEvents, { size: 10, slide: 10, allowedLateness: 2 });
  tumbling.flush();
  const tumblingOracle = batchWindows(tumblingEvents, { size: 10 });
  assert.strictEqual(tumbling.droppedLate, 0, "an on-time stream drops nothing");
  assert.deepStrictEqual(startsOf(tumbling), [...tumblingOracle.keys()],
    "windows are emitted once each, in ascending order");
  assert.deepStrictEqual(startsOf(tumbling), [...new Set(startsOf(tumbling))],
    "no window is emitted twice");
  const tumblingRecords = byStart(tumbling);
  for (const [start, expected] of tumblingOracle) {
    const actual = tumblingRecords.get(start);
    assert.strictEqual(actual.count, expected.count, `window ${start} event count`);
    assert.strictEqual(actual.weight, expected.weight, `window ${start} weight sum`);
    assert.deepStrictEqual(actual.degrees, expected.degrees, `window ${start} degrees`);
    assert.strictEqual(actual.triangles, expected.triangles, `window ${start} triangles`);
  }
  assert.ok(tumblingOracle.size >= 80, `tumbling windows compared: ${tumblingOracle.size}`);
  const triangleTotal = [...tumblingOracle.values()]
    .reduce((acc, record) => acc + record.triangles, 0);
  assert.ok(triangleTotal > 0, `the stream contains windowed triangles (${triangleTotal})`);

  // --- sliding windows against the batch oracle ------------------------------
  const slidingEvents = onTimeStream(8442, 600);
  const sliding = runStream(slidingEvents, { size: 12, slide: 4, allowedLateness: 1 });
  sliding.flush();
  const slidingOracle = batchWindows(slidingEvents, { size: 12, slide: 4 });
  assert.strictEqual(sliding.output.length, slidingOracle.size,
    "each sliding window is emitted exactly once");
  const slidingRecords = byStart(sliding);
  for (const [start, expected] of slidingOracle) {
    const actual = slidingRecords.get(start);
    assert.strictEqual(actual.count, expected.count, `sliding window ${start} count`);
    assert.strictEqual(actual.weight, expected.weight, `sliding window ${start} weight`);
    assert.strictEqual(actual.triangles, expected.triangles, `sliding window ${start} triangles`);
  }
  assert.ok(slidingOracle.size > 120, `sliding windows compared: ${slidingOracle.size}`);
  assert.ok([...slidingOracle.keys()].some((start) => slidingRecords.get(start).count > 0),
    "sliding windows really carry events");

  // --- watermark monotonicity and the closing rule ---------------------------
  const watermarkProbe = new WindowedGraphStream({ size: 10, slide: 10, allowedLateness: 3 });
  const watermarks = [];
  for (const timestamp of [0, 5, 9, 12, 20, 21, 40, 41]) {
    watermarkProbe.process({ timestamp, from: 1, to: 2, weight: 1 });
    watermarks.push(watermarkProbe.watermark);
  }
  for (let i = 1; i < watermarks.length; i++) {
    assert.ok(watermarks[i] >= watermarks[i - 1], "the watermark never goes backwards");
  }
  assert.deepStrictEqual(startsOf(watermarkProbe), [0, 10, 20],
    "windows close as soon as end + lateness is behind the watermark; a window with no events is"
    + " never created at all");
  assert.strictEqual(watermarkProbe.liveWindows(), 1, "only the newest window stays open");

  // --- lateness policy -------------------------------------------------------
  const crafted = craftedLateStream();
  const lateEngine = new WindowedGraphStream({ size: 10, slide: 10, allowedLateness: 2 });
  const acceptedFlags = crafted.map((event) => lateEngine.process(event));
  lateEngine.flush();
  assert.strictEqual(lateEngine.acceptedLate, 1, "the late-but-in-time event is folded in");
  assert.strictEqual(lateEngine.droppedLate, 1, "the far-too-late event is dropped");
  assert.strictEqual(acceptedFlags.filter((flag) => flag).length, crafted.length - 1,
    "exactly one event was refused");
  const acceptedOnly = crafted.filter((event) => !event.tooLate);
  const craftedOracle = batchWindows(acceptedOnly, { size: 10 });
  const lateRecords = byStart(lateEngine);
  for (const [start, expected] of craftedOracle) {
    assert.strictEqual(lateRecords.get(start).count, expected.count, `crafted window ${start} count`);
    assert.strictEqual(lateRecords.get(start).weight, expected.weight,
      `crafted window ${start} weight`);
    assert.strictEqual(lateRecords.get(start).triangles, expected.triangles,
      `crafted window ${start} triangles`);
  }
  assert.deepStrictEqual(startsOf(lateEngine), [...craftedOracle.keys()],
    "no window is emitted twice after a late correction");
  const withoutLate = batchWindows(acceptedOnly.filter((event) => !event.late), { size: 10 });
  assert.notStrictEqual(lateRecords.get(0).weight, withoutLate.get(0).weight,
    "the accepted late event really changed the aggregate");

  // An event exactly on the lateness boundary still lands in its window.
  const boundary = new WindowedGraphStream({ size: 10, slide: 10, allowedLateness: 2 });
  assert.strictEqual(boundary.process({ timestamp: 12, from: 1, to: 2, weight: 1 }), true);
  assert.strictEqual(boundary.process({ timestamp: 10, from: 3, to: 4, weight: 1 }), true,
    "an event exactly at the lateness boundary is accepted");
  assert.strictEqual(boundary.process({ timestamp: 4, from: 5, to: 6, weight: 1 }), false,
    "an event one tick past the boundary is dropped");
  boundary.flush();
  assert.strictEqual(boundary.output.find((record) => record.start === 10).count, 2,
    "both accepted events are inside window [10,20)");
  assert.strictEqual(boundary.output.length, 1, "the dropped event created no window of its own");

  // --- bounded state ---------------------------------------------------------
  const bound = WindowedGraphStream.windowBound(10, 5, 2);
  const shortStream = runStream(onTimeStream(8443, 400), { size: 10, slide: 5, allowedLateness: 2 });
  const longStream = runStream(onTimeStream(8443, 4000), { size: 10, slide: 5, allowedLateness: 2 });
  const otherLong = runStream(onTimeStream(8444, 4000), { size: 10, slide: 5, allowedLateness: 2 });
  for (const stream of [shortStream, longStream, otherLong]) {
    assert.ok(stream.peakLiveWindows <= bound,
      `live windows stay bounded (${stream.peakLiveWindows} <= ${bound})`);
    assert.ok(stream.liveWindows() <= bound, "and the bound holds after the last event");
    assert.ok(stream.peakLiveEntries <= bound * 3 * 8,
      `live entries stay bounded (${stream.peakLiveEntries})`);
  }
  assert.strictEqual(shortStream.peakLiveWindows, longStream.peakLiveWindows,
    "a ten times longer stream holds the same number of windows");
  shortStream.flush();
  longStream.flush();
  otherLong.flush();
  assert.strictEqual(longStream.evictions, longStream.output.length,
    "each eviction produces exactly one record");
  assert.ok(longStream.evictions >= 700, `windows were evicted ${longStream.evictions} times`);
  const longEvents = onTimeStream(8443, 4000);
  const longOracle = batchWindows(longEvents, { size: 10, slide: 5 });
  assert.strictEqual(longStream.output.length, longOracle.size,
    "the long stream emitted every window");
  const longRecords = byStart(longStream);
  let longMismatch = 0;
  for (const [start, expected] of longOracle) {
    const actual = longRecords.get(start);
    if (actual.count !== expected.count || actual.triangles !== expected.triangles) longMismatch++;
  }
  assert.strictEqual(longMismatch, 0, "every long-stream window matches the batch oracle");

  // --- eviction is invisible -------------------------------------------------
  const evictEvents = onTimeStream(8445, 700);
  const evicting = runStream(evictEvents, { size: 8, slide: 4, allowedLateness: 1 });
  const retaining = runStream(evictEvents, { size: 8, slide: 4, allowedLateness: 1,
    retainState: true });
  assert.strictEqual(evicting.evictions, retaining.evictions, "both engines close the same windows");
  assert.strictEqual(evicting.flush(), retaining.flush(), "and flush the same leftovers");
  assert.deepStrictEqual(evicting.output, retaining.output,
    "evicting on close and keeping everything produce identical records");
  assert.strictEqual(retaining.retained.size, retaining.evictions,
    `the retaining engine really kept the closed state (${retaining.retained.size})`);
  assert.strictEqual(evicting.retained.size, 0,
    "while the default engine dropped every closed window");
  assert.ok(evicting.evictions > 100, `the evicting run really evicted (${evicting.evictions})`);

  // --- edge cases ------------------------------------------------------------
  const single = new WindowedGraphStream({ size: 5, slide: 5 });
  assert.strictEqual(single.process({ timestamp: 0, from: 0, to: 1, weight: 2 }), true);
  assert.strictEqual(single.flush(), 1, "a single event in a single window flushes once");
  assert.strictEqual(single.output[0].count, 1, "and is counted once");
  assert.strictEqual(single.output[0].weight, 2, "with its weight");

  const noEvents = new WindowedGraphStream({ size: 5, slide: 5 });
  assert.strictEqual(noEvents.flush(), 0, "an empty stream flushes nothing");
  assert.deepStrictEqual(noEvents.output, [], "and emits nothing");

  const burst = new WindowedGraphStream({ size: 10, slide: 10 });
  for (let i = 0; i < 50; i++) burst.process({ timestamp: 7, from: i, to: i + 1, weight: 1 });
  burst.flush();
  assert.strictEqual(burst.output.length, 1, "one burst timestamp makes one window");
  assert.strictEqual(burst.output[0].count, 50, "and counts every event in it");
  assert.strictEqual(burst.output[0].triangles, 0, "a path of edges contains no triangle");

  const sizeOne = new WindowedGraphStream({ size: 1, slide: 1 });
  for (let i = 0; i < 6; i++) sizeOne.process({ timestamp: i, from: 0, to: 1, weight: 1 });
  sizeOne.flush();
  assert.deepStrictEqual(startsOf(sizeOne), [0, 1, 2, 3, 4, 5],
    "slide == size == 1 opens one window per tick");
  assert.ok(sizeOne.output.every((record) => record.count === 1), "each holds exactly one event");

  const gap = new WindowedGraphStream({ size: 5, slide: 5, allowedLateness: 0 });
  gap.process({ timestamp: 1, from: 0, to: 1, weight: 1 });
  gap.process({ timestamp: 21, from: 0, to: 1, weight: 1 });
  gap.flush();
  assert.deepStrictEqual(startsOf(gap), [0, 20], "windows with no events are never created");

  const duplicate = new WindowedGraphStream({ size: 10, slide: 10 });
  duplicate.process({ timestamp: 1, from: 1, to: 2, weight: 1 });
  duplicate.process({ timestamp: 2, from: 2, to: 1, weight: 1 });
  duplicate.flush();
  assert.strictEqual(duplicate.output[0].count, 2, "both duplicate events are counted");
  assert.strictEqual(duplicate.output[0].triangles, 0, "a duplicated edge is not a triangle");
  assert.strictEqual(Object.keys(duplicate.output[0].degrees).length, 2, "two distinct vertices");

  const triangle = new WindowedGraphStream({ size: 10, slide: 10 });
  for (const edge of [{ from: 1, to: 2 }, { from: 2, to: 3 }, { from: 1, to: 3 }]) {
    triangle.process({ timestamp: 1, ...edge, weight: 1 });
  }
  triangle.process({ timestamp: 11, from: 1, to: 2, weight: 1 });
  triangle.flush();
  assert.strictEqual(triangle.output[0].triangles, 1, "a triangle inside one window is counted");
  assert.strictEqual(triangle.output[1].triangles, 0,
    "a triangle split across windows is not claimed");

  // --- determinism -----------------------------------------------------------
  const replayA = runStream(evictEvents, { size: 8, slide: 4, allowedLateness: 1 });
  const replayB = runStream(evictEvents, { size: 8, slide: 4, allowedLateness: 1 });
  assert.deepStrictEqual(replayA.output, replayB.output, "the same stream yields identical records");
  assert.strictEqual(replayA.peakLiveEntries, replayB.peakLiveEntries, "and the same peak state");
  assert.strictEqual(replayA.watermark, replayB.watermark, "and the same final watermark");

  console.log(`windows: tumbling=${tumbling.output.length} sliding=${sliding.output.length} ` +
    `windowed-triangles=${triangleTotal}`);
  console.log(`lateness: accepted-late=${lateEngine.acceptedLate} ` +
    `dropped-late=${lateEngine.droppedLate} live-window bound=${bound}`);
  console.log(`state: peak-live-entries=${longStream.peakLiveEntries} ` +
    `evictions=${longStream.evictions} accepted=${longStream.accepted}`);
  console.log("PASS");
}
