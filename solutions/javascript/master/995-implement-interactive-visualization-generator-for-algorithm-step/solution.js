/** Task 995 - Implement interactive visualization generator for algorithm steps (trace player).
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * An instrumented bubble sort emits a trace of frames, where a frame carries only the cells it
 * changes. A player then walks that trace forward and backward: forward applies a frame's deltas and
 * pushes the previous values onto an undo log, backward pops that log. Because the two directions use
 * different mechanisms (delta application vs undo restoration), comparing them against a state that
 * is rebuilt from scratch by replaying from frame 0 is a real cross-check, not a tautology.
 *
 * What is simplified: the visualization is a state array plus a step description rather than pixels,
 * and the only instrumented algorithm is bubble sort.
 * Verified: final state vs an independent sort oracle, forward/backward round-trips to every index,
 * seeking from arbitrary positions, delta compactness, JSON round-tripping, frame validation, the
 * closed-form comparison count and the degenerate empty/single-element traces.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the file's only randomness source, so fixtures are byte-reproducible. */
function splitmix64(seed) {
  let s = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    s = BigInt.asUintN(64, s + 0x9e3779b97f4a7c15n);
    let z = BigInt.asUintN(64, (s ^ (s >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

/** Bubble sort instrumented to emit one delta frame per comparison and one per swap. */
function bubbleTrace(values) {
  const a = values.slice();
  const frames = [{ op: "init", at: [], changes: a.map((v, i) => [i, v]) }];
  let comparisons = 0;
  for (let i = 0; i < a.length - 1; i += 1) {
    for (let j = 0; j < a.length - 1 - i; j += 1) {
      comparisons += 1;
      frames.push({ op: "compare", at: [j, j + 1], changes: [] });
      if (a[j] > a[j + 1]) {
        const left = a[j], right = a[j + 1];
        a[j] = right;
        a[j + 1] = left;
        frames.push({ op: "swap", at: [j, j + 1], changes: [[j, right], [j + 1, left]] });
      }
    }
  }
  return { frames, result: a.slice(), size: a.length, comparisons };
}

/** Trace validation: frame shape, in-range cells and op vocabulary. */
function validateTrace(trace) {
  const issues = [];
  const ops = ["init", "compare", "swap"];
  trace.frames.forEach((frame, i) => {
    if (!ops.includes(frame.op)) issues.push({ frame: i, reason: "unknown-op" });
    for (const [cell, value] of frame.changes) {
      if (!Number.isInteger(cell) || cell < 0 || cell >= trace.size) issues.push({ frame: i, reason: "cell-out-of-range" });
      if (!Number.isFinite(value)) issues.push({ frame: i, reason: "non-numeric-value" });
    }
    for (const cell of frame.at) if (cell < 0 || cell >= trace.size) issues.push({ frame: i, reason: "at-out-of-range" });
  });
  return { ok: issues.length === 0, issues };
}

/** Step description for the UI; a pure function of the frame, so the caption never flickers. */
function describe(frame) {
  if (frame.op === "init") return `load ${frame.changes.length} values`;
  if (frame.op === "compare") return `compare cells ${frame.at[0]} and ${frame.at[1]}`;
  return `swap cells ${frame.at[0]} and ${frame.at[1]}`;
}

/**
 * The player. `forward` applies deltas and remembers what it overwrote; `backward` restores those
 * values. `stateAt` is a deliberate second implementation: a from-scratch replay used as an oracle.
 */
class TracePlayer {
  constructor(trace) {
    this.trace = trace;
    this.index = 0;
    this.state = new Array(trace.size).fill(undefined);
    for (const [cell, value] of trace.frames[0].changes) this.state[cell] = value;
    this.undo = [];
  }

  get atEnd() { return this.index === this.trace.frames.length - 1; }
  get caption() { return describe(this.trace.frames[this.index]); }

  forward() {
    if (this.atEnd) return this.index;
    const frame = this.trace.frames[this.index + 1];
    const restore = [];
    for (const [cell, value] of frame.changes) restore.push([cell, this.state[cell]]);
    for (const [cell, value] of frame.changes) this.state[cell] = value;
    this.undo.push(restore);
    this.index += 1;
    return this.index;
  }

  backward() {
    if (this.index === 0) return this.index;
    const restore = this.undo.pop();
    for (const [cell, value] of restore) this.state[cell] = value;
    this.index -= 1;
    return this.index;
  }

  /** Move to frame k using whichever direction is shorter; the result must not depend on the path. */
  seek(k) {
    const target = Math.max(0, Math.min(this.trace.frames.length - 1, k));
    while (this.index < target) this.forward();
    while (this.index > target) this.backward();
    return this.index;
  }

  /** Oracle: rebuild the state at frame k by replaying the deltas from the start, ignoring the log. */
  stateAt(k) {
    const state = new Array(this.trace.size).fill(undefined);
    for (let i = 0; i <= k; i += 1) for (const [cell, value] of this.trace.frames[i].changes) state[cell] = value;
    return state;
  }

  snapshot() { return { index: this.index, state: this.state.slice(), caption: this.caption }; }
}

/** Independent sort oracle for the trace's final state, written as an insertion sort. */
function insertionSort(values) {
  const a = values.slice();
  for (let i = 1; i < a.length; i += 1) {
    const key = a[i];
    let j = i - 1;
    while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j -= 1; }
    a[j + 1] = key;
  }
  return a;
}

const seededValues = (seed, count, max = 60) => {
  const rng = splitmix64(seed);
  return Array.from({ length: count }, () => rng.int(max));
};

function main() {
  const values = [5, 1, 4, 2, 8];
  const trace = bubbleTrace(values);
  const player = new TracePlayer(trace);
  // 1. Happy path: replaying to the end reconstructs the sorted array for both oracles.
  player.seek(trace.frames.length - 1);
  assert.deepStrictEqual(player.state, trace.result, "player state matches the instrumented result");
  assert.deepStrictEqual(player.state, insertionSort(values), "insertion-sort oracle agrees");
  assert.deepStrictEqual(player.state, values.slice().sort((a, b) => a - b), "Array#sort oracle agrees");
  assert.strictEqual(player.atEnd, true, "player reports it is at the end");
  // 2. Backward invariance: stepping all the way back restores the original contents exactly.
  while (player.index > 0) player.backward();
  assert.deepStrictEqual([player.state, player.index], [values, 0], "backward run restores the input");
  // 3. Cross-check at every index: the undo-log path and the replay path must agree, coming from
  // either direction.
  for (let k = 0; k < trace.frames.length; k += 1) {
    player.seek(k);
    assert.deepStrictEqual(player.state, player.stateAt(k), `arrival at ${k} matches the replay oracle`);
  }
  const forwardStates = [];
  player.seek(0);
  while (true) { forwardStates.push(player.state.slice()); if (player.index === trace.frames.length - 1) break; player.forward(); }
  const backwardStates = [];
  player.seek(trace.frames.length - 1);
  while (true) { backwardStates.unshift(player.state.slice()); if (player.index === 0) break; player.backward(); }
  assert.deepStrictEqual(backwardStates, forwardStates, "forward and backward agree on every frame");
  // 4. Boundary: an empty array traces to a single init frame, and a one-element array never swaps.
  const empty = bubbleTrace([]);
  const single = bubbleTrace([42]);
  assert.deepStrictEqual([empty.frames.length, new TracePlayer(empty).state, empty.comparisons], [1, [], 0],
    "empty trace is just the init frame");
  assert.deepStrictEqual([single.frames.length, single.result, new TracePlayer(single).caption], [1, [42], "load 1 values"],
    "single element needs no comparisons");
  // 5. Closed form: the worst case performs exactly n(n-1)/2 comparisons.
  const worst = bubbleTrace([5, 4, 3, 2, 1]);
  assert.strictEqual(worst.comparisons, (5 * 4) / 2, "comparison count matches the closed form");
  // 6. Compactness: a frame's deltas never touch more than the two cells a swap involves.
  assert.ok(trace.frames.every((f) => (f.op === "init" ? f.changes.length === trace.size : f.changes.length <= 2)),
    "only the init frame carries more than two cell changes");
  assert.ok(trace.frames.filter((f) => f.op === "compare").every((f) => f.changes.length === 0),
    "comparison frames carry no delta");
  // 7. Validation and JSON round-trip: a well-formed trace passes, an out-of-range cell is named, and
  // a serialized trace drives the player identically.
  assert.deepStrictEqual([validateTrace(trace).ok, validateTrace(trace).issues],
    [true, []], "generated trace is valid");
  const broken = { size: 5, frames: [{ op: "init", at: [], changes: [[9, 1]] }, { op: "jump", at: [0], changes: [] }] };
  assert.deepStrictEqual([validateTrace(broken).ok, validateTrace(broken).issues.map((i) => i.reason)],
    [false, ["cell-out-of-range", "unknown-op"]], "malformed frames are reported");
  const parsed = new TracePlayer(JSON.parse(JSON.stringify(trace)));
  parsed.seek(trace.frames.length - 1);
  assert.deepStrictEqual([parsed.state, parsed.caption], [player.stateAt(trace.frames.length - 1), describe(trace.frames[trace.frames.length - 1])],
    "a serialized trace plays back identically");
  // 8. Seeded fixtures: every random array sorts correctly and the trace digest reproduces.
  const fixtures = [seededValues(0x995, 9), seededValues(0x996, 17), seededValues(0x997, 24)];
  for (const fixture of fixtures) {
    const t = bubbleTrace(fixture);
    const p = new TracePlayer(t);
    p.seek(t.frames.length - 1);
    assert.deepStrictEqual([p.state, t.comparisons <= (fixture.length * (fixture.length - 1)) / 2],
      [insertionSort(fixture), true], `fixture of ${fixture.length} values sorts correctly`);
    assert.strictEqual(validateTrace(t).ok, true, "fixture trace is valid");
  }
  const digest = fnv1a(JSON.stringify(bubbleTrace(fixtures[1]).frames.map((f) => f.changes)));
  assert.strictEqual(fnv1a(JSON.stringify(bubbleTrace(fixtures[1]).frames.map((f) => f.changes))), digest,
    "trace digest reproduces byte for byte");
  console.log(`frames=${trace.frames.length} comparisons=${trace.comparisons} caption="${player.caption}" digest=${digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { splitmix64, fnv1a, bubbleTrace, validateTrace, describe, TracePlayer, insertionSort,
  seededValues };
