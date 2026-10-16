/** Task 936 - Implement transparent operator checkpointing and resume in compute graph.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * A streaming graph (source -> map/count -> running sum per key -> sink) driven by explicit ticks.
 * The coordinator injects Chandy-Lamport barriers after every `interval` records; each operator
 * snapshots its own state while forwarding a barrier and the sink completes the checkpoint when the
 * barrier reaches it. A crash rolls the graph back to the last completed checkpoint: operator states
 * are restored, channels are dropped, the source replays from the barrier's offset and the sink is
 * truncated to the checkpoint length - so the resumed output equals the crash-free output exactly.
 *
 * Simplified: state lives in memory rather than in a durable store, channels are FIFO, in-flight
 * messages are not persisted (the barrier guarantees a consistent cut) and only a linear chain is
 * modelled. Verified: every checkpoint is a consistent cut checked against a batch oracle that
 * recomputes the prefix from the seed, exactly-once output after crashes at many ticks, replayable
 * source offsets, and operator state continuity across the restart.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the only source of randomness; the same stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

/** The source stream: record i is produced by the i-th pair of draws of splitmix64(seed). */
const nextRecord = (seq) => ({ value: seq.int(10), key: seq.int(3) });

/** Batch oracle: prefix sums of the same stream, computed without any streaming machinery. */
function expected(seed, n) {
  const seq = splitmix64(seed);
  const sums = new Map();
  const out = [];
  for (let i = 0; i < n; i += 1) {
    const { value, key } = nextRecord(seq);
    const total = (sums.get(key) ?? 0) + value + 1;
    sums.set(key, total);
    out.push(`${key}:${total}`);
  }
  return { out, sums };
}

const eqMap = (a, b) => a.size === b.size && [...a].every(([k, v]) => b.get(k) === v);
const count = (states) => states[0].count;

function newRun({ n, seed, interval }) {
  return { n, seed, interval, seq: splitmix64(seed), offset: 0, epoch: 0, endSent: false, tick: 0,
    chan: [[], [], []], count: 0, sums: new Map(), sink: [], pending: new Map(),
    checkpoints: [{ epoch: -1, offset: 0, sinkLength: 0, states: [{ count: 0 }, { sums: new Map() }] }] };
}

/**
 * One tick: the source emits one record (and possibly a barrier), each operator consumes at most one
 * message, and the sink consumes one. A barrier snapshot is keyed by epoch because several barriers
 * can be in flight at once.
 */
function step(run, cap = 4000) {
  if (run.offset < run.n) {
    run.chan[0].push({ kind: "rec", ...nextRecord(run.seq) });
    run.offset += 1;
    if (run.tick % run.interval === run.interval - 1) {
      run.chan[0].push({ kind: "barrier", epoch: run.epoch, offset: run.offset });
      run.epoch += 1;
    }
  } else if (!run.endSent) {
    run.endSent = true;
    run.chan[0].push({ kind: "barrier", epoch: run.epoch, offset: run.offset });
  }
  for (let op = 0; op < 2; op += 1) {
    const msg = run.chan[op][0];
    if (!msg) continue;
    run.chan[op].shift();
    if (msg.kind === "rec") {
      if (op === 0) { run.count += 1; run.chan[1].push({ kind: "rec", key: msg.key, value: msg.value + 1 }); }
      else {
        const sum = (run.sums.get(msg.key) ?? 0) + msg.value;
        run.sums.set(msg.key, sum);
        run.chan[2].push({ text: `${msg.key}:${sum}` });
      }
    } else {
      const snap = run.pending.get(msg.epoch) ?? { epoch: msg.epoch, offset: msg.offset, states: [null, null] };
      snap.states[op] = op === 0 ? { count: run.count } : { sums: new Map(run.sums) };
      run.pending.set(msg.epoch, snap);
      run.chan[op + 1].push(msg);
    }
  }
  const head = run.chan[2][0];
  if (!head) { run.tick += 1; return run; }
  run.chan[2].shift();
  if (head.text) run.sink.push(head.text);
  else {
    const snap = run.pending.get(head.epoch);
    assert(snap && snap.states.every(Boolean), `barrier ${head.epoch} must pass every operator before the sink`);
    assert(head.offset === run.sink.length, "the sink prefix must match the source offset at the barrier");
    run.checkpoints.push({ epoch: head.epoch, offset: head.offset, sinkLength: run.sink.length, states: snap.states });
    run.pending.delete(head.epoch);
  }
  run.tick += 1;
  return run;
}

const drained = (run) => run.offset >= run.n && run.endSent && run.chan.every((c) => c.length === 0);

function advance(run, until, cap = 4000) {
  let guard = 0;
  while (!drained(run) && (until === undefined || run.tick < until)) {
    step(run);
    guard += 1;
    assert(guard <= cap, "the graph must terminate inside the tick budget");
  }
  return run;
}

/** Restore a consistent cut: replayable source, snapshot states, truncated sink, empty channels. */
function resume(run, checkpoint) {
  run.seq = splitmix64(run.seed);
  for (let i = 0; i < checkpoint.offset; i += 1) nextRecord(run.seq);
  run.offset = checkpoint.offset;
  run.epoch = checkpoint.epoch + 1;
  run.endSent = false;
  run.chan = [[], [], []];
  run.pending = new Map();
  run.count = count(checkpoint.states);
  run.sums = new Map(checkpoint.states[1].sums);
  run.sink.length = checkpoint.sinkLength;
  run.checkpoints = run.checkpoints.filter((c) => c.epoch <= checkpoint.epoch);
  return run;
}

function runClean(opts) { return advance(newRun(opts)); }

function runWithCrash(opts, crashTick) {
  const run = advance(newRun(opts), crashTick);
  const checkpoint = run.checkpoints[run.checkpoints.length - 1];
  const atCrash = run.checkpoints.length;
  resume(run, checkpoint);
  advance(run);
  return { run, checkpoint, atCrash };
}

function main() {
  const opts = { n: 40, seed: 936, interval: 5 };
  const truth = expected(opts.seed, opts.n);
  const clean = runClean(opts);

  // 1. Happy path: streaming output equals the batch oracle, final state included.
  assert(clean.sink.join("|") === truth.out.join("|"), "clean output equals the batch oracle");
  assert(clean.count === opts.n && eqMap(clean.sums, truth.sums), "final operator state equals the oracle state");
  assert(clean.checkpoints.length === 2 + Math.floor(opts.n / opts.interval), "one checkpoint per barrier plus the end barrier");

  // 2. Independence of the oracle: every checkpoint is a consistent cut for exactly its offset.
  for (const cp of clean.checkpoints.slice(1)) {
    const prefix = expected(opts.seed, cp.offset);
    assert(cp.sinkLength === cp.offset && count(cp.states) === cp.offset, `checkpoint ${cp.epoch} cuts at ${cp.offset}`);
    assert(clean.sink.slice(0, cp.sinkLength).join("|") === prefix.out.join("|"), "checkpoint sink prefix equals the oracle prefix");
    assert(eqMap(cp.states[1].sums, prefix.sums), "checkpoint sum state equals the oracle prefix state");
  }

  // 3. Crash and resume: the resumed sink is identical to the crash-free sink (exactly-once).
  for (const tick of [1, 2, 3, 4, 5, 6, 7, 13, 26, 39]) {
    const { run } = runWithCrash(opts, tick);
    assert(run.sink.join("|") === clean.sink.join("|"), `crash at tick ${tick} duplicates or loses output`);
  }

  // 4. Boundary: crashing before the first barrier completes restarts from the empty checkpoint.
  const early = runWithCrash(opts, 2);
  assert(early.checkpoint.epoch === -1 && early.atCrash === 1, "an early crash resumes from the initial checkpoint");
  assert(early.run.sink.join("|") === clean.sink.join("|"), "resuming from the initial checkpoint is still exactly-once");

  // 5. Boundary: interval = 1 (a barrier per record) and interval > n (only the end barrier).
  for (const interval of [1, 2, 41]) {
    const o = { ...opts, interval };
    const base = runClean(o);
    assert(base.sink.join("|") === truth.out.join("|"), `interval ${interval} produces the same output`);
    for (const tick of [3, 11]) {
      assert(runWithCrash(o, tick).run.sink.join("|") === base.sink.join("|"), `interval ${interval}, crash at ${tick}`);
    }
  }
  const onlyEnd = runClean({ ...opts, interval: 99 });
  assert(onlyEnd.checkpoints.length === 2, "interval larger than the stream yields just the end barrier");

  // 6. Boundary: empty stream drains without producing output or state.
  const empty = runClean({ n: 0, seed: 7, interval: 4 });
  assert(empty.sink.length === 0 && empty.count === 0 && empty.sums.size === 0, "empty stream yields an empty result");

  // 7. State continuity: after a restart the operator state equals the clean run's final state.
  const mid = runWithCrash(opts, 17);
  assert(mid.run.count === clean.count && eqMap(mid.run.sums, clean.sums), "operator state is continuous across the restart");

  // 8. Barrier accounting after the restart: epochs grow, offsets are monotone, the end barrier is last.
  const epochs = mid.run.checkpoints.map((c) => c.epoch);
  const offsets = mid.run.checkpoints.map((c) => c.offset);
  assert(epochs.every((e, i) => i === 0 || e > epochs[i - 1]), `checkpoint epochs must increase: ${epochs}`);
  assert(offsets.every((o, i) => i === 0 || o >= offsets[i - 1]), "checkpoint offsets must not go backwards");
  assert(offsets[offsets.length - 1] === opts.n && mid.run.checkpoints.at(-1).sinkLength === opts.n, "the end barrier commits the full stream");

  // 9. Fuzzed crashes across several seeds and intervals.
  let crashes = 0;
  for (const seed of [11, 12, 13, 14]) {
    const o = { n: 30, seed, interval: 4 };
    const base = runClean(o);
    for (const tick of [0, 6, 12, 19, 27]) {
      assert(runWithCrash(o, tick).run.sink.join("|") === base.sink.join("|"), `seed ${seed}, crash at ${tick}`);
      crashes += 1;
    }
  }

  console.log(`checkpointing: records=${opts.n} checkpoints=${clean.checkpoints.length} crash-resumes=${crashes + 13} all exactly-once`);
  console.log("PASS");
}
module.exports = { splitmix64, expected, newRun, step, advance, resume, runClean, runWithCrash, drained };
if (require.main === module) main();
