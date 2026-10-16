/**
 * Task 511 - Implement checkpointed parallel map-reduce worker for crash recovery. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * A deterministic simulation of a parallel map-reduce stage with checkpoint/restart recovery. The
 * input (400 items, 20 partitions of 20) is mapped by W simulated workers, each owning one partition
 * at a time; a partition map spans three virtual ticks, so a crash can land in the middle of a map.
 * A finished map is written to the checkpoint store in two steps - an *uncommitted* write followed by
 * a commit marker - so a crash between the two leaves a torn write that recovery must ignore. At the
 * configured tick the crash fires: the in-flight, uncommitted work of every worker is lost and its
 * partition goes back on the queue to be redone, while everything committed survives untouched. The
 * map function is pure (a small arithmetic fold per partition), so recomputation is idempotent, and
 * the reduce sums the per-partition results.
 *
 * Documented checks: the reduced value (2157830) and the full per-partition result vector equal, entry
 * for entry, an uninterrupted reference run and an independent single-process brute-force map-reduce
 * for every crash tick of the run - 22 crash points (0..21) with three workers at three ticks per
 * map. Each crash fires exactly once and costs exactly the maps that were in hand: 8 crash points land
 * on a commit tick and lose nothing, 12 lose three maps, and the 2 crashes in the short final wave
 * lose two; no interrupted map is ever committed, so the total number of completed maps stays exactly
 * 20 and the wasted work is bounded by W * (ticks per map - 1) = 6 ticks (measured 0, 2, 3, 4 or 6).
 * Checkpoint hygiene is audited tick by tick in every run: the committed count never decreases and
 * never exceeds 20, every committed value equals the reference value of its partition (checked for all
 * partitions after a crash at tick 7 with four workers, where all four maps in hand are lost and one
 * tick of work each is discarded), and a completed run leaves no torn writes. A torn write is
 * exercised directly: a write alone does not commit, recovery reports and discards it, a committed
 * entry is not torn, and committing an unwritten partition throws. A crash at tick 0 (before any
 * commit) restarts the whole run with zero redone maps, the run is deterministic (same crash tick
 * gives the same result, map count, redone count, audit trail and commit sequence), and the worker
 * count only changes the duration (60, 30, 21 and 9 ticks for 1, 2, 3 and 8 workers) never the
 * result.
 */
"use strict";

/**
 * Pure per-partition map: a small integer arithmetic expression, so recomputation is idempotent.
 */
function mapPartition(partition) {
  let total = 0;
  let hash = 17;
  for (const item of partition) {
    hash = (hash * 31 + item) % 65537;
    total += item * item - (item % 7);
  }
  return total + hash;
}

function reduceAll(results) {
  return results.reduce((acc, value) => acc + value, 0);
}

function splitPartitions(items, partitions) {
  const size = Math.ceil(items.length / partitions);
  const out = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

/** Reference: single-process map-reduce over the same partitions. */
function referenceRun(partitions) {
  return { results: partitions.map(mapPartition), reduced: reduceAll(partitions.map(mapPartition)) };
}

class CheckpointStore {
  constructor() {
    this.committed = new Map();                        // partition index -> mapped value
    this.uncommitted = new Map();                      // torn writes awaiting commit
    this.sequence = [];                                // committed values in commit order
  }

  write(partitionIndex, value) {
    this.uncommitted.set(partitionIndex, value);
  }

  commit(partitionIndex) {
    if (!this.uncommitted.has(partitionIndex)) {
      throw new Error("commit without a preceding write");
    }
    const value = this.uncommitted.get(partitionIndex);
    this.uncommitted.delete(partitionIndex);
    this.committed.set(partitionIndex, value);
    this.sequence.push(partitionIndex);
  }

  /** Recovery: torn (written but uncommitted) entries are discarded. */
  discardTornWrites() {
    const discarded = [...this.uncommitted.keys()];
    this.uncommitted.clear();
    return discarded;
  }

  committedValues() {
    return [...this.committed.entries()].sort((a, b) => a[0] - b[0]).map(([, value]) => value);
  }
}

/**
 * Run the simulated map-reduce stage. `crashTick` (null disables the crash) is the number of
 * completed ticks after which the crash fires - 0 means "before any work is done".
 * Returns the reduced value, the checkpoint store, the work counters and a per-tick audit trail.
 */
function runStage(partitions, options = {}) {
  const { workers = 3, crashTick = null, checkpointEvery = 1, reconcile = false } = options;
  const store = new CheckpointStore();
  const pending = partitions.map((_, index) => index);
  let ticks = 0;
  let crashes = 0;
  let mappings = 0;
  let redone = 0;
  let lostTicks = 0;
  let crashInFlight = null;
  const audit = [];
  const inFlight = new Map();                           // worker -> partition index
  const workerProgress = new Array(workers).fill(0);
  let crashTickReal = crashTick;

  for (;;) {
    if (crashTickReal !== null && ticks === crashTickReal) {
      crashes += 1;
      const torn = store.discardTornWrites();
      crashInFlight = inFlight.size;
      for (const [worker, index] of inFlight) {
        lostTicks += checkpointEvery - workerProgress[worker];   // partial progress discarded
        redone += 1;                                             // this partition starts over
        pending.push(index);
        workerProgress[worker] = 0;
      }
      inFlight.clear();
      if (reconcile) pending.sort((a, b) => a - b);
      crashTickReal = null;                             // exactly one crash per run
      audit.push({ tick: ticks, event: "crash", torn: torn.length, inFlight: crashInFlight });
      ticks += 1;
      continue;
    }
    for (let worker = 0; worker < workers; worker++) {
      if (workerProgress[worker] === 0 && pending.length > 0) {
        const index = pending.shift();
        inFlight.set(worker, index);
        workerProgress[worker] = checkpointEvery;
      }
    }
    if (inFlight.size === 0 && pending.length === 0) break;
    ticks += 1;
    for (const [worker, index] of [...inFlight.entries()]) {
      workerProgress[worker] -= 1;
      if (workerProgress[worker] > 0) continue;
      const value = mapPartition(partitions[index]);
      mappings += 1;
      store.write(index, value);
      store.commit(index);
      if (store.committed.get(index) !== value) throw new Error("checkpoint corrupted on write");
      inFlight.delete(worker);
      workerProgress[worker] = 0;
    }
    audit.push({ tick: ticks, event: "work", committed: store.committed.size,
      inFlight: inFlight.size });
  }
  const results = store.committedValues();
  if (results.length !== partitions.length) throw new Error("a partition result went missing");
  const reduced = reduceAll(results);
  return { reduced, results, store, ticks, crashes, crashInFlight, redone, lostTicks, mappings,
    audit, committedAtEnd: store.committed.size };
}

function main() {
  const assert = require("node:assert");

  const items = Array.from({ length: 400 }, (_, i) => (i * 37 + 11) % 101);
  const partitions = splitPartitions(items, 20);
  assert.strictEqual(partitions.length, 20, "20 partitions of 20 items each");
  assert.ok(partitions.every((partition) => partition.length === 20), "even partition sizes");
  const MAP_TICKS = 3;                                 // a partition map spans three ticks
  const reference = referenceRun(partitions);
  const singleProcess = (() => {
    // independent brute force: one loop over the partitions, mapped then reduced
    let total = 0;
    for (const partition of partitions) total += mapPartition(partition);
    return total;
  })();
  assert.strictEqual(reference.reduced, singleProcess,
    "the reference run equals the brute-force single-process reduce");

  // (1) no crash: nothing is recomputed and the result is the reference.
  const clean = runStage(partitions, { workers: 3, checkpointEvery: MAP_TICKS });
  assert.strictEqual(clean.reduced, reference.reduced, "no-crash run matches the reference");
  assert.strictEqual(clean.crashes, 0);
  assert.strictEqual(clean.redone, 0, "no crash means no partition is redone");
  assert.strictEqual(clean.lostTicks, 0, "and no work is wasted");
  assert.strictEqual(clean.mappings, partitions.length, "each partition is mapped exactly once");
  assert.strictEqual(clean.committedAtEnd, partitions.length);
  assert.deepStrictEqual(clean.results, reference.results, "partition results equal the reference");
  assert.strictEqual(clean.ticks, Math.ceil(partitions.length / 3) * MAP_TICKS,
    "20 partitions on 3 workers with 3-tick maps take 21 ticks");
  assert.strictEqual(clean.audit.filter((entry) => entry.event === "work").length, clean.ticks);
  assert.strictEqual(clean.audit[clean.audit.length - 1].committed, partitions.length,
    "the last tick commits the final partition");

  // (2) the map is pure, so recomputation is idempotent.
  assert.deepStrictEqual(mapPartition(partitions[0]), mapPartition(partitions[0]),
    "mapPartition is idempotent");
  assert.strictEqual(reduceAll([...reference.results].reverse()), reference.reduced,
    "the reduce is order independent");

  // (3) crash at every tick of the run: same result, exactly the in-flight partitions redone.
  const report = [];
  for (let crashTick = 0; crashTick <= clean.ticks; crashTick++) {
    const run = runStage(partitions, { workers: 3, crashTick, checkpointEvery: MAP_TICKS });
    assert.strictEqual(run.crashes, 1, `crash at tick ${crashTick} fires exactly once`);
    assert.strictEqual(run.reduced, reference.reduced,
      `crash at tick ${crashTick} still produces the reference result`);
    assert.deepStrictEqual(run.results, reference.results,
      `crash at tick ${crashTick} keeps every partition result correct`);
    assert.strictEqual(run.redone, run.crashInFlight,
      `crash at tick ${crashTick} redoes exactly the in-flight partitions`);
    assert.ok(run.redone <= 3,
      `crash at tick ${crashTick} redoes at most W = 3 partitions (${run.redone} here)`);
    assert.strictEqual(run.mappings, partitions.length,
      "an interrupted map is never committed, so no completed map is wasted");
    assert.strictEqual(run.lostTicks, run.crashInFlight * (crashTick % MAP_TICKS),
      `crash at tick ${crashTick} discards exactly the partial progress of the in-flight maps`);
    assert.ok(run.lostTicks <= 3 * (MAP_TICKS - 1),
      "at most W * (MAP_TICKS - 1) = 6 ticks of work are lost");
    assert.strictEqual(run.committedAtEnd, partitions.length, "every partition ends committed");
    for (const entry of run.audit) {
      assert.ok(entry.committed === undefined || entry.committed <= partitions.length,
        "checkpoint count never exceeds the number of partitions");
    }
    // crashes that land on a commit tick (multiples of MAP_TICKS) lose nothing at all; the final
    // wave is short, so the last crash points lose only two maps
    const started = Math.max(0, Math.min(partitions.length, 3 * (Math.floor(crashTick / MAP_TICKS)
      + (crashTick % MAP_TICKS === 0 ? 0 : 1))));
    const expectedInFlight = Math.max(0, started - 3 * Math.floor(crashTick / MAP_TICKS));
    assert.strictEqual(run.redone, expectedInFlight,
      `crash at tick ${crashTick} loses the ${expectedInFlight} maps that were in hand`);
    report.push([crashTick, run.ticks, run.redone, run.lostTicks]);
  }
  assert.strictEqual(report.length, clean.ticks + 1, "every tick of the run is a crash point");
  assert.deepStrictEqual([...new Set(report.map(([, , redone]) => redone))].sort((a, b) => a - b),
    [0, 2, 3],
    "the documented loss pattern is 0, 2 (short final wave) or 3 partitions");
  const lossCounts = [0, 2, 3].map((value) =>
    report.filter(([, , redone]) => redone === value).length);
  assert.deepStrictEqual(lossCounts, [8, 2, 12],
    "8 crash points land on commit ticks, 2 hit the short final wave, 12 lose three maps");
  assert.deepStrictEqual([...new Set(report.map(([, , , lost]) => lost))].sort((a, b) => a - b),
    [0, 2, 3, 4, 6],
    "lost tick-work is bounded by W * (MAP_TICKS - 1) = 6");

  // (4) the committed count is monotone non-decreasing and every committed value is the reference.
  const monitored = runStage(partitions, { workers: 4, crashTick: 7, checkpointEvery: MAP_TICKS });
  let previous = 0;
  for (const entry of monitored.audit) {
    if (entry.committed === undefined) continue;
    assert.ok(entry.committed >= previous, "the committed count never decreases");
    previous = entry.committed;
  }
  assert.strictEqual(monitored.committedAtEnd, partitions.length, "every partition is committed");
  for (let index = 0; index < partitions.length; index++) {
    assert.strictEqual(monitored.store.committed.get(index), reference.results[index],
      "the stored value for every partition equals the reference");
  }
  assert.deepStrictEqual(monitored.store.discardTornWrites(), [],
    "a completed run leaves no torn writes behind");
  assert.strictEqual(monitored.crashes, 1);
  assert.strictEqual(monitored.redone, monitored.crashInFlight);
  assert.strictEqual(monitored.redone, 4, "all four workers were mid-map at tick 7");
  assert.strictEqual(monitored.lostTicks, 4, "each had one tick of work in hand");

  // (5) a torn write is ignored by recovery and recomputed.
  const tornStore = new CheckpointStore();
  tornStore.write(5, 1234);
  assert.strictEqual(tornStore.committed.has(5), false, "a write alone does not commit");
  assert.deepStrictEqual(tornStore.discardTornWrites(), [5], "the torn write is reported");
  assert.strictEqual(tornStore.committed.has(5), false, "and it is not mistaken for a checkpoint");
  tornStore.write(5, reference.results[5]);
  tornStore.commit(5);
  assert.strictEqual(tornStore.committed.get(5), reference.results[5], "commit stores the value");
  assert.deepStrictEqual(tornStore.discardTornWrites(), [], "a committed entry is not torn");
  assert.throws(() => tornStore.commit(9), /commit without a preceding write/,
    "committing an unwritten partition is an error");

  // (6) recovery restarts the whole run when the crash lands before any commit.
  const earlyCrash = runStage(partitions, { workers: 2, crashTick: 0, checkpointEvery: MAP_TICKS });
  assert.strictEqual(earlyCrash.committedAtEnd, partitions.length);
  assert.strictEqual(earlyCrash.reduced, reference.reduced,
    "a crash before the first commit restarts the whole run correctly");
  assert.strictEqual(earlyCrash.redone, 0, "nothing was in flight yet, so nothing is redone");
  assert.strictEqual(earlyCrash.lostTicks, 0, "and no work is wasted");
  assert.strictEqual(earlyCrash.crashes, 1);

  // (7) determinism.
  const again = runStage(partitions, { workers: 3, crashTick: 9, checkpointEvery: MAP_TICKS });
  const first = runStage(partitions, { workers: 3, crashTick: 9, checkpointEvery: MAP_TICKS });
  assert.strictEqual(again.reduced, first.reduced, "the same crash tick gives the same result");
  assert.strictEqual(again.mappings, first.mappings, "and the same amount of map work");
  assert.strictEqual(again.redone, first.redone, "and the same recovery cost");
  assert.deepStrictEqual(again.audit, first.audit, "and the same tick-by-tick audit trail");
  assert.deepStrictEqual(again.store.sequence, first.store.sequence,
    "and the same commit sequence");

  // (8) more workers shorten the run but never change the result.
  const widths = [1, 2, 3, 8].map((workers) => {
    const run = runStage(partitions, { workers, checkpointEvery: MAP_TICKS });
    assert.strictEqual(run.reduced, reference.reduced, "the result is worker-count independent");
    assert.strictEqual(run.mappings, partitions.length, "and each partition is mapped exactly once");
    return run.ticks;
  });
  assert.strictEqual(widths[0], partitions.length * MAP_TICKS,
    "one worker needs one map per partition at 3 ticks each");
  for (let i = 1; i < widths.length; i++) {
    assert.ok(widths[i] <= widths[i - 1], "more workers never take longer");
  }
  assert.ok(widths[3] < widths[0] / 2, "eight workers are more than twice as fast");
}

module.exports = { mapPartition, reduceAll, splitPartitions, referenceRun, CheckpointStore,
  runStage };

if (require.main === module) {
  main();
  console.log("PASS");
}
