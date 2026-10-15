/** Task 479 - Implement compaction strategies for LSM (levelled vs size-tiered). (Tier: Advanced, CTCI Ch 14 — Databases) */
"use strict";

const KEYS = 256;
const OPS = 128;
const FLUSH_EVERY = 8;
const MERGE_FANOUT = 4;
const FACTOR = 2;

class SizeTieredEngine {
  constructor() {
    this.runs = []; // chronological; each run is a Map key -> value
    this.totalWritten = 0;
    this.mergeCount = 0;
    this.peakRuns = 0;
    this.peakEntries = 0;
  }

  flush(memtable) {
    this.runs.push(new Map(memtable));
    this.totalWritten += memtable.size;
    this.compact();
    this.peakRuns = Math.max(this.peakRuns, this.runs.length);
    this.peakEntries = Math.max(this.peakEntries, this.runs.reduce((sum, run) => sum + run.size, 0));
  }

  compact() {
    for (;;) {
      const sizes = this.runs.map((run, idx) => [run.size, idx]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      let target = null;
      for (let i = 0; i + 3 < sizes.length; i++) {
        if (sizes[i + 3][0] <= FACTOR * sizes[i][0]) {
          target = sizes.slice(i, i + MERGE_FANOUT).map(([, idx]) => idx).sort((a, b) => a - b);
          break;
        }
      }
      if (target === null) return;
      const merged = new Map();
      for (const idx of target) for (const [k, v] of this.runs[idx]) merged.set(k, v); // newer wins
      this.totalWritten += merged.size;
      this.mergeCount += 1;
      const newRuns = [];
      let inserted = false;
      this.runs.forEach((run, i) => {
        if (target.includes(i)) {
          if (!inserted) {
            newRuns.push(merged);
            inserted = true;
          }
        } else {
          newRuns.push(run);
        }
      });
      this.runs = newRuns;
    }
  }

  materialize() {
    const out = new Map();
    for (const run of this.runs) for (const [k, v] of run) out.set(k, v);
    return out;
  }
}

class LevelledEngine {
  constructor() {
    this.levels = []; // levels[i] is a chronological list of runs
    this.totalWritten = 0;
    this.mergeCount = 0;
    this.peakRuns = 0;
    this.peakEntries = 0;
  }

  ensureLevel(i) {
    while (this.levels.length <= i) this.levels.push([]);
  }

  flush(memtable) {
    this.ensureLevel(0);
    this.levels[0].push(new Map(memtable));
    this.totalWritten += memtable.size;
    this.compact();
    const runs = this.levels.reduce((sum, level) => sum + level.length, 0);
    const entries = this.levels.reduce((sum, level) => sum + level.reduce((s, run) => s + run.size, 0), 0);
    this.peakRuns = Math.max(this.peakRuns, runs);
    this.peakEntries = Math.max(this.peakEntries, entries);
  }

  compact() {
    for (;;) {
      let merged = false;
      for (let i = 0; i < this.levels.length; i++) {
        const level = this.levels[i];
        if (level.length >= 2) {
          const combined = new Map(level[0]); // older
          for (const [k, v] of level[1]) combined.set(k, v); // newer wins
          this.totalWritten += combined.size;
          this.mergeCount += 1;
          level.splice(0, 2);
          this.ensureLevel(i + 1);
          this.levels[i + 1].push(combined);
          merged = true;
          break;
        }
      }
      if (!merged) return;
    }
  }

  materialize() {
    const out = new Map();
    for (let i = this.levels.length - 1; i >= 0; i--) {
      for (const run of this.levels[i]) for (const [k, v] of run) out.set(k, v);
    }
    return out;
  }
}

function lcg(seed) {
  return (seed * 75 + 74) % 65537;
}

function genWrites(count, seed, keyspace) {
  const ops = [];
  for (let i = 0; i < count; i++) {
    seed = lcg(seed);
    const key = seed % keyspace;
    seed = lcg(seed);
    const value = seed % 1000;
    ops.push([key, value]);
  }
  return ops;
}

function mapsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const [k, v] of a) if (b.get(k) !== v) return false;
  return true;
}

module.exports = { SizeTieredEngine, LevelledEngine, lcg, genWrites };

if (require.main === module) {
  const assert = require("node:assert");
  const ops = genWrites(OPS, 24680, KEYS);
  const oracle = new Map();
  const st = new SizeTieredEngine();
  const lv = new LevelledEngine();
  for (let start = 0; start < OPS; start += FLUSH_EVERY) {
    const memtable = new Map();
    for (let i = start; i < start + FLUSH_EVERY; i++) {
      memtable.set(ops[i][0], ops[i][1]);
      oracle.set(ops[i][0], ops[i][1]);
    }
    st.flush(memtable);
    lv.flush(memtable);
    assert.ok(mapsEqual(st.materialize(), oracle), `size-tiered lost a key at ${start}`);
    assert.ok(mapsEqual(lv.materialize(), oracle), `levelled lost a key at ${start}`);
  }
  assert.strictEqual(oracle.size, 107, "live keys hand-counted to 107");
  assert.ok(mapsEqual(st.materialize(), oracle));
  assert.ok(mapsEqual(lv.materialize(), oracle));

  assert.strictEqual(st.totalWritten, 357);
  assert.strictEqual(st.mergeCount, 5);
  assert.strictEqual(st.peakRuns, 6);
  assert.strictEqual(st.peakEntries, 115);
  assert.strictEqual(st.runs.length, 1);
  assert.strictEqual(st.runs[0].size, 107);

  assert.strictEqual(lv.totalWritten, 600);
  assert.strictEqual(lv.mergeCount, 15, "8 + 4 + 2 + 1 compactions");
  assert.strictEqual(lv.peakRuns, 4);
  assert.strictEqual(lv.peakEntries, 112);
  assert.strictEqual(lv.levels.length, 5);
  for (let i = 0; i < lv.levels.length - 1; i++) assert.deepStrictEqual(lv.levels[i], []);
  assert.strictEqual(lv.levels[4].length, 1);
  assert.strictEqual(lv.levels[4][0].size, 107);

  // Documented qualitative outcomes for this parameterisation.
  assert.ok(lv.totalWritten > st.totalWritten, "levelled rewrites more entries (write amplification)");
  assert.ok(lv.mergeCount > st.mergeCount, "levelled compacts more often, in smaller steps");
  assert.ok(st.peakRuns > lv.peakRuns, "size-tiered keeps more runs alive (read amplification)");
  assert.ok(st.peakEntries >= lv.peakEntries, "size-tiered temporarily stores more copies");
  console.log("PASS");
}
