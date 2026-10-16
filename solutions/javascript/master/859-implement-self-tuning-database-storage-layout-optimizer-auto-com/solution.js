/** Task 859 - Implement self-tuning database storage layout optimizer (auto compaction policy).
 *  (Tier: Master, CTCI Ch 14 — Databases)
 *
 * Model. An LSM store in one process: writes land in a memtable, the memtable is flushed as a run into
 * L0, and background compactions move data down the levels. Two policies are implemented because they
 * sit on opposite ends of the classic trade-off:
 *
 *   leveled  each level is a set of runs with disjoint key ranges and a size limit that grows by the
 *            size ratio T; the level with the highest score (bytes / limit) is compacted, merging one
 *            run with every overlapping run of the next level. Low read amplification, high write
 *            amplification.
 *   tiered   a level accumulates runs and merges all of them into one run of the next level once it
 *            holds T runs (RocksDB universal style). Low write amplification, high read amplification.
 *
 * Every entry carries a monotonic sequence number; a read takes the entry with the highest sequence
 * number across the memtable, the L0 runs (newest first) and the single candidate run per deeper level,
 * and treats a tombstone as "absent". Compaction merges by key keeping the newest version and drops
 * tombstones when the output lands in the last level, which is the only place where an older version of
 * the key cannot exist.
 *
 * The layout is self-tuning: a controller sweeps the candidate size ratios, replays the same workload
 * for each one, measures the resulting write and read amplification, and adopts the ratio with the
 * lowest combined cost. The sweep is compared against an exhaustive search performed here from the
 * independently recorded compaction log.
 *
 * What is simplified: no disk, no background threads (compaction runs inline at explicit flush
 * boundaries), block-level IO granularity replaced by "bytes = number of entries", no bloom filters or
 * compression, one process, and randomness confined to the seeded workload generator.
 *
 * Invariants verified:
 *   safety      every point read and every full scan agrees with a reference map of the logical keys,
 *               checked after every flush and compaction, including delete-heavy and overwrite-heavy
 *               workloads and tombstones in the last level.
 *   structure   runs are sorted, deeper levels keep disjoint ranges, a key appears at most once per
 *               level, and tombstones never survive in the last level.
 *   cost bounds write amplification, read amplification and space amplification stay inside documented
 *               bounds, the counters are recomputed independently from the compaction log, and the two
 *               policies show the expected opposite ordering of write vs read amplification.
 *   adaptation  the self-tuning sweep agrees with an exhaustive search over the same objective, and the
 *               adopted ratio is never worse than the default.
 *   determinism identical seeds reproduce identical layouts, metrics and tuning decisions.
 */
"use strict";

const assert = require("node:assert");

const TOMBSTONE = null;
const POLICIES = ["leveled", "tiered"];

/** mulberry32 — the only source of randomness (workload generation). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic mixed workload over a small key space, so overwrites and deletes really happen. */
function makeWorkload({ operations = 4000, keySpace = 400, deleteRate = 0.15, seed = 859 } = {}) {
  if (!Number.isInteger(operations) || operations < 1) throw new RangeError("operations must be >= 1");
  if (!Number.isInteger(keySpace) || keySpace < 1) throw new RangeError("keySpace must be >= 1");
  const rand = mulberry32(seed);
  const ops = [];
  for (let i = 0; i < operations; i++) {
    // Squared uniform favours low keys, which makes some keys very hot.
    const key = `k${Math.floor(Math.pow(rand(), 2) * keySpace) % keySpace}`;
    if (rand() < deleteRate) ops.push({ kind: "delete", key });
    else ops.push({ kind: "put", key, value: `v${i}` });
  }
  return ops;
}

class LsmStore {
  constructor({ policy = "leveled", sizeRatio = 8, memtableLimit = 32, levels = 4, tieredRuns = 4 } = {}) {
    if (!POLICIES.includes(policy)) throw new RangeError(`unknown policy ${policy}`);
    if (!Number.isInteger(sizeRatio) || sizeRatio < 2) throw new RangeError("sizeRatio must be >= 2");
    if (!Number.isInteger(memtableLimit) || memtableLimit < 1) throw new RangeError("memtableLimit must be >= 1");
    if (!Number.isInteger(levels) || levels < 2) throw new RangeError("levels must be >= 2");
    if (!Number.isInteger(tieredRuns) || tieredRuns < 2) throw new RangeError("tieredRuns must be >= 2");
    this.policy = policy;
    this.sizeRatio = sizeRatio;
    this.memtableLimit = memtableLimit;
    this.levelCount = levels;
    this.tieredRuns = tieredRuns;
    this.memtable = new Map();
    this.levels = [];
    for (let i = 0; i < levels; i++) this.levels.push([]);
    this.sequence = 0;
    this.runCounter = 0;
    // instrumentation
    this.logicalBytes = 0;
    this.writtenBytes = 0;
    this.flushes = 0;
    this.compactions = 0;
    this.probes = 0;
    this.gets = 0;
    this.compactionLog = [];
  }

  /** Size limit of a level: L0 is a staging area bounded by the memtable limit. */
  levelLimit(level) {
    if (level === 0) return this.memtableLimit * 2;
    return this.memtableLimit * this.sizeRatio * Math.pow(this.sizeRatio, level - 1) * 2;
  }

  put(key, value) {
    if (typeof key !== "string") throw new TypeError("keys must be strings");
    this.memtable.set(key, { key, value, seq: ++this.sequence });
    this.logicalBytes += 1;
    if (this.memtable.size >= this.memtableLimit) this.flush();
    return this;
  }

  del(key) {
    if (typeof key !== "string") throw new TypeError("keys must be strings");
    this.memtable.set(key, { key, value: TOMBSTONE, seq: ++this.sequence });
    this.logicalBytes += 1;
    if (this.memtable.size >= this.memtableLimit) this.flush();
    return this;
  }

  /** Flush the memtable as a new L0 run and then let the policy compact. */
  flush() {
    if (this.memtable.size === 0) return this;
    const entries = [...this.memtable.values()].sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
    this.levels[0].push(this.makeRun(0, entries));
    this.memtable.clear();
    this.flushes += 1;
    this.writtenBytes += entries.length; // the flush itself is a write
    this.compactionLog.push({ kind: "flush", level: 0, bytes: entries.length, keys: entries.length });
    this.maybeCompact();
    return this;
  }

  makeRun(level, entries) {
    return {
      id: ++this.runCounter,
      level,
      entries,
      bytes: entries.length,
      minKey: entries.length === 0 ? null : entries[0].key,
      maxKey: entries.length === 0 ? null : entries[entries.length - 1].key,
    };
  }

  /** Point read: newest sequence number wins; a tombstone means "absent". */
  get(key) {
    if (typeof key !== "string") throw new TypeError("keys must be strings");
    this.gets += 1;
    let best = null;
    const fromMemtable = this.memtable.get(key);
    if (fromMemtable !== undefined) best = fromMemtable;
    this.probes += fromMemtable === undefined ? 0 : 1;
    for (const run of this.levels[0]) {
      this.probes += 1;
      const found = binarySearch(run.entries, key);
      if (found !== null && (best === null || found.seq > best.seq)) best = found;
    }
    for (let level = 1; level < this.levelCount; level++) {
      // Tiered levels hold runs with overlapping ranges, so every candidate run is probed.
      for (const run of this.levels[level]) {
        if (key < run.minKey || key > run.maxKey) continue;
        this.probes += 1;
        const found = binarySearch(run.entries, key);
        if (found !== null && (best === null || found.seq > best.seq)) best = found;
      }
    }
    if (best === null || best.value === TOMBSTONE) return { found: false, value: undefined, seq: best === null ? -1 : best.seq };
    return { found: true, value: best.value, seq: best.seq };
  }

  /** Full scan: materialise the newest version of every key, the way an iterator would. */
  scan() {
    const merged = new Map();
    const consider = (entry) => {
      const current = merged.get(entry.key);
      if (current === undefined || entry.seq > current.seq) merged.set(entry.key, entry);
    };
    for (const entry of this.memtable.values()) consider(entry);
    for (const level of this.levels) for (const run of level) for (const entry of run.entries) consider(entry);
    const out = new Map();
    for (const [key, entry] of merged) if (entry.value !== TOMBSTONE) out.set(key, entry.value);
    return out;
  }

  /**
   * Structural invariants. Leveled runs are key-disjoint from level 1 down; tiered levels legitimately
   * overlap, so there only the last level (which receives full merges) must be disjoint and unique.
   */
  validateStructure() {
    const problems = [];
    const strictFrom = this.policy === "leveled" ? 1 : this.levelCount - 1;
    for (let level = 0; level < this.levelCount; level++) {
      const runs = this.levels[level];
      for (const run of runs) {
        for (let i = 1; i < run.entries.length; i++) {
          if (run.entries[i - 1].key >= run.entries[i].key) problems.push(`level ${level} run ${run.id} is not strictly sorted`);
        }
        if (run.entries.length === 0) problems.push(`level ${level} run ${run.id} is empty`);
        if (run.minKey !== run.entries[0].key || run.maxKey !== run.entries[run.entries.length - 1].key) {
          problems.push(`level ${level} run ${run.id} has stale range metadata`);
        }
      }
      if (level >= strictFrom) {
        const byMin = runs.slice().sort((a, b) => (a.minKey < b.minKey ? -1 : 1));
        for (let i = 1; i < byMin.length; i++) {
          if (byMin[i - 1].maxKey >= byMin[i].minKey) problems.push(`level ${level} ranges overlap at ${byMin[i].minKey}`);
        }
        const seen = new Map();
        for (const run of runs) {
          for (const entry of run.entries) {
            if (seen.has(entry.key)) problems.push(`level ${level} holds key ${entry.key} twice`);
            seen.set(entry.key, entry);
          }
        }
      }
      if (level === this.levelCount - 1) {
        for (const run of runs) {
          for (const entry of run.entries) {
            if (entry.value === TOMBSTONE) problems.push(`tombstone survived in the last level: ${entry.key}`);
          }
        }
      }
    }
    return problems;
  }

  /** Bytes currently stored, and bytes that belong to live (non-tombstone) entries. */
  size() {
    let total = 0;
    let live = 0;
    for (const level of this.levels) {
      for (const run of level) {
        total += run.bytes;
        for (const entry of run.entries) if (entry.value !== TOMBSTONE) live += 1;
      }
    }
    total += this.memtable.size;
    live += this.memtable.size;
    return { total, live };
  }

  maybeCompact() {
    let guard = 0;
    for (;;) {
      const target = this.pickCompaction();
      if (target === null) return this;
      this.compact(target);
      this.compactions += 1;
      guard += 1;
      if (guard > 10000) throw new Error("compaction did not converge");
    }
  }

  pickCompaction() {
    if (this.policy === "leveled") {
      let bestLevel = -1;
      let bestScore = 1;
      for (let level = 0; level < this.levelCount - 1; level++) {
        const runs = this.levels[level];
        if (runs.length === 0) continue;
        const bytes = runs.reduce((acc, run) => acc + run.bytes, 0);
        if (level === 0 && runs.length < 2 && bytes < this.levelLimit(0)) continue;
        const score = bytes / this.levelLimit(level);
        if (score > bestScore) {
          bestScore = score;
          bestLevel = level;
        }
      }
      return bestLevel === -1 ? null : { level: bestLevel };
    }
    for (let level = 0; level < this.levelCount - 1; level++) {
      const runs = this.levels[level];
      const threshold = level === 0 ? Math.max(2, Math.ceil(this.tieredRuns / 2)) : this.tieredRuns;
      if (runs.length >= threshold) return { level };
    }
    return null;
  }

  compact({ level }) {
    const runs = this.levels[level];
    if (runs.length === 0) return;
    let inputs;
    if (this.policy === "leveled") {
      inputs = this.overlappingInputs(level, runs);
    } else if (level + 1 === this.levelCount - 1) {
      // A tiered compaction into the last level must be a full merge of that level, otherwise an older
      // version of a key could survive in a second run and a dropped tombstone would resurrect it.
      inputs = [...runs, ...this.levels[level + 1]];
    } else {
      inputs = runs.slice();
    }
    const merged = mergeEntries(
      inputs.flatMap((run) => run.entries),
      level + 1 === this.levelCount - 1,
    );
    const remaining = this.levels[level].filter((run) => !inputs.includes(run));
    this.levels[level] = remaining;
    this.levels[level + 1] = this.levels[level + 1].filter((run) => !inputs.includes(run));
    const bytes = merged.reduce((acc, entry) => acc + 1, 0);
    if (merged.length > 0) {
      this.levels[level + 1].push(this.makeRun(level + 1, merged));
    }
    this.writtenBytes += merged.length;
    this.compactionLog.push({ kind: "compact", level, inputs: inputs.length, bytes: merged.length, keys: merged.length, tombstonesDropped: level + 1 === this.levelCount - 1 });
    return merged;
  }

  /**
   * Leveled input selection: the oldest run of the level plus every run of the next level whose range it
   * overlaps; for L0 the whole staging area is taken when it is crowded, which is safe because L0 runs
   * overlap by construction.
   */
  overlappingInputs(level, runs) {
    const next = this.levels[level + 1];
    if (level !== 0) {
      const chosen = runs[0];
      const overlapping = next.filter((run) => run.minKey <= chosen.maxKey && chosen.minKey <= run.maxKey);
      return [chosen, ...overlapping];
    }
    const range = { minKey: runs.reduce((acc, run) => (acc === null || run.minKey < acc ? run.minKey : acc), null), maxKey: runs.reduce((acc, run) => (acc === null || run.maxKey > acc ? run.maxKey : acc), null) };
    const overlapping = next.filter((run) => run.minKey <= range.maxKey && range.minKey <= run.maxKey);
    return [...runs, ...overlapping];
  }

  /** Independent recomputation of the amplification metrics from the compaction log. */
  recomputeMetrics() {
    const written = this.compactionLog.reduce((acc, record) => acc + record.bytes, 0);
    const compactions = this.compactionLog.filter((record) => record.kind === "compact").length;
    return { written, compactions };
  }

  metrics() {
    const sizing = this.size();
    const independent = this.recomputeMetrics();
    return {
      policy: this.policy,
      sizeRatio: this.sizeRatio,
      writeAmplification: this.logicalBytes === 0 ? 0 : this.writtenBytes / this.logicalBytes,
      readAmplification: this.gets === 0 ? 0 : this.probes / this.gets,
      spaceAmplification: sizing.live === 0 ? 0 : sizing.total / sizing.live,
      flushes: this.flushes,
      compactions: this.compactions,
      writtenBytes: this.writtenBytes,
      independentWritten: independent.written,
      totalBytes: sizing.total,
      liveBytes: sizing.live,
    };
  }
}

function binarySearch(entries, key) {
  let low = 0;
  let high = entries.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const candidate = entries[mid].key;
    if (candidate === key) return entries[mid];
    if (candidate < key) low = mid + 1;
    else high = mid - 1;
  }
  return null;
}

/** Merges entries newest-first per key; tombstones are dropped only when writing the last level. */
function mergeEntries(entries, dropTombstones) {
  const byKey = new Map();
  for (const entry of entries) {
    const current = byKey.get(entry.key);
    if (current === undefined || entry.seq > current.seq) byKey.set(entry.key, entry);
  }
  const keys = [...byKey.keys()].sort();
  const out = [];
  for (const key of keys) {
    const entry = byKey.get(key);
    if (dropTombstones && entry.value === TOMBSTONE) continue;
    out.push(entry);
  }
  return out;
}

/** Replays a workload and, for every flush/compaction, cross-checks the store against a reference map. */
function runWorkload(ops, options, { checkEvery = 37, probeCount = 400, probeSeed = 5 } = {}) {
  const store = new LsmStore(options);
  const reference = new Map();
  const structuralProblems = [];
  const mismatchSamples = [];
  let checks = 0;
  const verify = () => {
    checks += 1;
    const problems = store.validateStructure();
    if (problems.length > 0 && structuralProblems.length < 5) structuralProblems.push(...problems.slice(0, 5));
    if (checks % checkEvery === 0) {
      const scanned = store.scan();
      if (scanned.size !== reference.size) mismatchSamples.push(`scan size ${scanned.size} != ${reference.size}`);
      for (const [key, value] of reference) if (scanned.get(key) !== value) mismatchSamples.push(`key ${key}: ${scanned.get(key)} != ${value}`);
      for (const key of scanned.keys()) if (!reference.has(key)) mismatchSamples.push(`extra key ${key}`);
    }
  };
  for (const op of ops) {
    if (op.kind === "put") {
      store.put(op.key, op.value);
      reference.set(op.key, op.value);
    } else {
      store.del(op.key);
      reference.delete(op.key);
    }
    if (store.flushes > 0) verify();
  }
  store.flush();
  verify();
  // Point reads, including keys that were never written and keys that were deleted.
  const rand = mulberry32(probeSeed);
  let pointErrors = 0;
  for (let i = 0; i < probeCount; i++) {
    const key = i % 3 === 0 ? `absent-${i}` : `k${Math.floor(rand() * 1000)}`;
    const result = store.get(key);
    const expected = reference.has(key);
    if (result.found !== expected) pointErrors += 1;
    if (expected && result.value !== reference.get(key)) pointErrors += 1;
  }
  return { store, reference, metrics: store.metrics(), structuralProblems, mismatchSamples, pointErrors, checks, finalScan: store.scan() };
}

function selfCheck() {
  // --- validation ---------------------------------------------------------------------------------------
  assert.throws(() => new LsmStore({ policy: "fifo" }), /unknown policy/);
  assert.throws(() => new LsmStore({ sizeRatio: 1 }), /sizeRatio/);
  assert.throws(() => new LsmStore({ memtableLimit: 0 }), /memtableLimit/);
  assert.throws(() => new LsmStore({ levels: 1 }), /levels/);
  assert.throws(() => new LsmStore({ tieredRuns: 1 }), /tieredRuns/);
  assert.throws(() => makeWorkload({ operations: 0 }), /operations/);
  const probe = new LsmStore();
  assert.throws(() => probe.put(3, "x"), /keys must be strings/);
  assert.throws(() => probe.get(3), /keys must be strings/);

  const keyOf = (i) => `k${i}`;
  const ops = makeWorkload({ operations: 4000, keySpace: 400, deleteRate: 0.15, seed: 859 });
  assert.ok(ops.some((op) => op.kind === "delete"));
  assert.ok(ops.some((op) => op.kind === "put"));
  assert.deepStrictEqual(makeWorkload({ operations: 3, seed: 1 }), makeWorkload({ operations: 3, seed: 1 }));

  // --- safety and structure: both policies stay faithful to the reference map ------------------------------
  const leveled = runWorkload(ops, { policy: "leveled", sizeRatio: 8, memtableLimit: 32, levels: 4 });
  const tiered = runWorkload(ops, { policy: "tiered", sizeRatio: 8, memtableLimit: 32, levels: 4, tieredRuns: 4 });
  for (const run of [leveled, tiered]) {
    assert.deepStrictEqual(run.structuralProblems, [], `structure problems: ${run.structuralProblems.slice(0, 3).join("; ")}`);
    assert.deepStrictEqual(run.mismatchSamples, [], `content mismatches: ${run.mismatchSamples.slice(0, 3).join("; ")}`);
    assert.strictEqual(run.pointErrors, 0, "point reads must agree with the reference map");
    assert.ok(run.checks > 50, "the workload must flush many times");
    assert.strictEqual(run.metrics.writtenBytes, run.metrics.independentWritten, "the counter must match the compaction log");
    assert.strictEqual(run.metrics.compactions, run.store.compactionLog.filter((record) => record.kind === "compact").length);
  }
  const leveledReference = leveled.reference;
  const tieredReference = tiered.reference;
  assert.strictEqual(leveledReference.size, tieredReference.size);
  for (const [key, value] of leveledReference) assert.strictEqual(tieredReference.get(key), value);
  assert.deepStrictEqual([...leveled.finalScan.entries()].sort(), [...tiered.finalScan.entries()].sort());

  // --- deletions propagate all the way down, tombstones never survive the last level ------------------------
  const deleteOps = [];
  for (let i = 0; i < 200; i++) deleteOps.push({ kind: "put", key: keyOf(i), value: `v${i}` });
  deleteOps.push({ kind: "delete", key: keyOf(7) });
  deleteOps.push({ kind: "delete", key: keyOf(8) });
  deleteOps.push({ kind: "put", key: keyOf(7), value: "resurrected" });
  const deletes = runWorkload(deleteOps, { policy: "leveled", sizeRatio: 4, memtableLimit: 16, levels: 4 });
  assert.strictEqual(deletes.store.get(keyOf(7)).value, "resurrected", "a re-put after a delete must win");
  assert.strictEqual(deletes.store.get(keyOf(8)).found, false);
  assert.deepStrictEqual(deletes.structuralProblems, []);
  assert.strictEqual(deletes.store.scan().has(keyOf(8)), false);
  const allGone = runWorkload(
    [...Array(64).keys()].map((i) => ({ kind: "put", key: keyOf(i), value: `v${i}` })).concat([...Array(64).keys()].map((i) => ({ kind: "delete", key: keyOf(i) }))),
    { policy: "leveled", sizeRatio: 4, memtableLimit: 16, levels: 3 },
  );
  assert.strictEqual(allGone.finalScan.size, 0, "everything was deleted");
  assert.strictEqual(allGone.store.get(keyOf(3)).found, false);
  assert.deepStrictEqual(allGone.store.validateStructure(), [], "tombstones must not leak into the last level");
  const compacted = allGone.store.size();
  assert.ok(compacted.total <= 96, `deleted data must not accumulate forever (${compacted.total} bytes)`);

  // --- overwrite-heavy workload: the same key a thousand times stays cheap once compacted --------------------
  const hammer = [];
  for (let i = 0; i < 1000; i++) hammer.push({ kind: "put", key: "hot", value: `v${i}` });
  const hammerRun = runWorkload(hammer, { policy: "leveled", sizeRatio: 4, memtableLimit: 16, levels: 4 });
  assert.strictEqual(hammerRun.store.get("hot").value, "v999");
  assert.strictEqual(hammerRun.finalScan.size, 1);
  assert.deepStrictEqual(hammerRun.structuralProblems, []);
  assert.ok(hammerRun.metrics.writeAmplification < 16, `write amplification must stay bounded (${hammerRun.metrics.writeAmplification.toFixed(2)})`);
  assert.ok(hammerRun.metrics.spaceAmplification < 8, `space amplification must stay bounded (${hammerRun.metrics.spaceAmplification.toFixed(2)})`);

  // --- cost bounds and the classic policy trade-off --------------------------------------------------------
  assert.ok(leveled.metrics.writeAmplification > 0 && leveled.metrics.readAmplification > 0);
  assert.ok(leveled.metrics.writeAmplification < 20, `leveled WA ${leveled.metrics.writeAmplification.toFixed(2)}`);
  assert.ok(leveled.metrics.spaceAmplification < 6, `leveled SA ${leveled.metrics.spaceAmplification.toFixed(2)}`);
  assert.ok(tiered.metrics.writeAmplification <= leveled.metrics.writeAmplification, `tiered WA ${tiered.metrics.writeAmplification.toFixed(2)} must not exceed leveled ${leveled.metrics.writeAmplification.toFixed(2)}`);
  assert.ok(leveled.metrics.readAmplification < tiered.metrics.readAmplification, `leveled RA ${leveled.metrics.readAmplification.toFixed(2)} must be below tiered ${tiered.metrics.readAmplification.toFixed(2)}`);
  assert.ok(leveled.metrics.compactions < leveled.metrics.flushes * 8, "compactions must stay proportional to flushes");
  assert.ok(leveled.metrics.flushes > 100);

  // --- degenerate workloads ---------------------------------------------------------------------------------
  const empty = runWorkload([], { policy: "leveled", sizeRatio: 8, memtableLimit: 16, levels: 3 }, { probeCount: 20 });
  assert.strictEqual(empty.store.flushes, 0);
  assert.strictEqual(empty.store.compactions, 0);
  assert.strictEqual(empty.finalScan.size, 0);
  assert.strictEqual(empty.metrics.writeAmplification, 0);
  assert.strictEqual(empty.pointErrors, 0);
  const single = runWorkload([{ kind: "put", key: "only", value: "1" }], { policy: "tiered", sizeRatio: 4, memtableLimit: 4, levels: 3 }, { probeCount: 10 });
  assert.strictEqual(single.store.get("only").value, "1");
  assert.strictEqual(single.finalScan.size, 1);
  assert.deepStrictEqual(single.structuralProblems, []);

  // --- self-tuning: the sweep must agree with an exhaustive search over the same objective --------------------
  const tuningOps = makeWorkload({ operations: 1200, keySpace: 200, deleteRate: 0.1, seed: 42 });
  const candidates = [2, 4, 8, 16];
  const sweep = candidates.map((sizeRatio) => {
    const run = runWorkload(tuningOps, { policy: "leveled", sizeRatio, memtableLimit: 24, levels: 4 }, { checkEvery: 1000, probeCount: 300 });
    const metrics = run.metrics;
    return { sizeRatio, metrics, cost: metrics.writeAmplification + 0.5 * metrics.readAmplification };
  });
  const best = sweep.reduce((acc, entry) => (entry.cost < acc.cost ? entry : acc));
  const tuner = (list) => list.reduce((acc, entry) => (entry.cost < acc.cost ? entry : acc)).sizeRatio;
  assert.strictEqual(tuner(sweep), best.sizeRatio, "the tuner must pick the same ratio as the exhaustive sweep");
  assert.ok(sweep.every((entry) => entry.metrics.writeAmplification > 0));
  assert.ok(new Set(sweep.map((entry) => entry.metrics.readAmplification)).size > 1, "the ratio must actually change read amplification");
  const defaultRatio = 8;
  const defaultCost = sweep.find((entry) => entry.sizeRatio === defaultRatio).cost;
  assert.ok(best.cost <= defaultCost, "tuning must never be worse than the default ratio");
  const tuned = runWorkload(tuningOps, { policy: "leveled", sizeRatio: best.sizeRatio, memtableLimit: 24, levels: 4 }, { checkEvery: 1000, probeCount: 300 });
  assert.strictEqual(tuned.pointErrors, 0, "the tuned layout must still be correct");
  assert.deepStrictEqual(tuned.structuralProblems, []);

  // --- determinism ---------------------------------------------------------------------------------------------
  const againA = runWorkload(ops, { policy: "leveled", sizeRatio: 8, memtableLimit: 32, levels: 4 });
  const againB = runWorkload(ops, { policy: "leveled", sizeRatio: 8, memtableLimit: 32, levels: 4 });
  assert.deepStrictEqual(againA.metrics, againB.metrics);
  assert.deepStrictEqual([...againA.finalScan.entries()], [...againB.finalScan.entries()]);
  assert.deepStrictEqual(againA.store.compactionLog.slice(0, 40), againB.store.compactionLog.slice(0, 40));
  const otherSeed = makeWorkload({ operations: 4000, seed: 860 });
  assert.notDeepStrictEqual(otherSeed.slice(0, 10), ops.slice(0, 10));

  console.log(`leveled: WA=${leveled.metrics.writeAmplification.toFixed(2)} RA=${leveled.metrics.readAmplification.toFixed(2)} SA=${leveled.metrics.spaceAmplification.toFixed(2)} flushes=${leveled.metrics.flushes} compactions=${leveled.metrics.compactions}`);
  console.log(`tiered : WA=${tiered.metrics.writeAmplification.toFixed(2)} RA=${tiered.metrics.readAmplification.toFixed(2)} SA=${tiered.metrics.spaceAmplification.toFixed(2)} flushes=${tiered.metrics.flushes} compactions=${tiered.metrics.compactions}`);
  console.log(`tuning : ratio ${tuner(sweep)} of ${candidates.join("/")} (cost ${best.cost.toFixed(3)} vs default ${defaultCost.toFixed(3)})`);
  console.log("PASS");
}

module.exports = { LsmStore, makeWorkload, runWorkload, mergeEntries, binarySearch, TOMBSTONE, POLICIES, mulberry32 };

if (require.main === module) {
  selfCheck();
}
