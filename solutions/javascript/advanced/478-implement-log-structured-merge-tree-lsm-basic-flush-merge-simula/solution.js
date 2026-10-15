/** Task 478 - Implement log-structured merge tree (LSM) basic flush/merge simulation. (Tier: Advanced, CTCI Ch 14 — Databases) */
"use strict";

const MERGE_FANOUT = 4;
const KEYS = 16;

class SSTable {
  constructor(tableId, entries) {
    this.id = tableId;
    this.entries = entries; // key -> {seq, value}; value null is a tombstone
  }
}

function mergeTables(tables) {
  const merged = new Map();
  for (const table of tables) {
    for (const [key, entry] of table.entries) {
      const current = merged.get(key);
      if (current === undefined || entry.seq > current.seq) merged.set(key, entry);
    }
  }
  return new SSTable(-1, merged);
}

class LSMStore {
  constructor(flushAfter) {
    this.memtable = new Map();
    this.sstables = [];
    this.seq = 0;
    this.opsSinceFlush = 0;
    this.flushAfter = flushAfter === undefined ? 4 : flushAfter;
    this.flushCount = 0;
    this.mergeCount = 0;
    this.nextTableId = 0;
  }

  write(key, value) {
    this.seq += 1;
    this.memtable.set(key, { seq: this.seq, value });
    this.opsSinceFlush += 1;
    if (this.opsSinceFlush >= this.flushAfter) this.flush();
  }

  delete(key) {
    this.write(key, null);
  }

  flush() {
    if (this.memtable.size === 0) return;
    this.nextTableId += 1;
    this.sstables.push(new SSTable(this.nextTableId, new Map(this.memtable)));
    this.memtable = new Map();
    this.opsSinceFlush = 0;
    this.flushCount += 1;
    while (this.sstables.length >= MERGE_FANOUT) {
      const merged = mergeTables(this.sstables.slice(0, MERGE_FANOUT));
      this.sstables = [merged, ...this.sstables.slice(MERGE_FANOUT)];
      this.mergeCount += 1;
    }
  }

  entryFor(key) {
    let best = this.memtable.get(key);
    for (const table of this.sstables) {
      const entry = table.entries.get(key);
      if (entry !== undefined && (best === undefined || entry.seq > best.seq)) best = entry;
    }
    return best;
  }

  get(key) {
    const entry = this.entryFor(key);
    if (entry === undefined) return null;
    return entry.value; // null means deleted
  }

  scan() {
    const merged = new Map();
    for (const table of this.sstables) {
      for (const [key, entry] of table.entries) {
        const current = merged.get(key);
        if (current === undefined || entry.seq > current.seq) merged.set(key, entry);
      }
    }
    for (const [key, entry] of this.memtable) {
      const current = merged.get(key);
      if (current === undefined || entry.seq > current.seq) merged.set(key, entry);
    }
    return [...merged.keys()]
      .sort((a, b) => a - b)
      .filter((key) => merged.get(key).value !== null)
      .map((key) => [key, merged.get(key).value]);
  }
}

function lcg(seed) {
  return (seed * 75 + 74) % 65537;
}

function genOps(count, seed) {
  const ops = [];
  for (let i = 0; i < count; i++) {
    seed = lcg(seed);
    const kind = seed % 5;
    seed = lcg(seed);
    const key = seed % KEYS;
    seed = lcg(seed);
    const value = seed % 100;
    ops.push([kind, key, value]);
  }
  return ops;
}

function runWorkload() {
  const assert = require("node:assert");
  const store = new LSMStore();
  const oracle = new Map();
  for (let k = 0; k < KEYS; k++) oracle.set(k, null);
  const ops = genOps(80, 31337);
  for (let i = 0; i < ops.length; i++) {
    const [kind, key, value] = ops[i];
    if (kind === 4) {
      store.delete(key);
      oracle.set(key, null);
    } else {
      store.write(key, value);
      oracle.set(key, value);
    }
    for (let k = 0; k < KEYS; k++) {
      assert.deepStrictEqual(store.get(k), oracle.get(k), `get ${i} key ${k}`);
    }
    const expected = [...oracle.keys()]
      .filter((k) => oracle.get(k) !== null)
      .sort((a, b) => a - b)
      .map((k) => [k, oracle.get(k)]);
    assert.deepStrictEqual(store.scan(), expected, `scan ${i}`);
  }
  assert.strictEqual(store.flushCount, 20, "one flush per four ops");
  assert.strictEqual(store.mergeCount, 6, "merge at flushes 4,7,10,13,16,19");
  assert.strictEqual(store.sstables.length, 2, "one merged table plus the last flush");
  assert.strictEqual(store.memtable.size, 0, "memtable drained by the last flush");
}

module.exports = { LSMStore, mergeTables, lcg, genOps };

if (require.main === module) {
  const assert = require("node:assert");
  const s = new LSMStore();
  s.write(0, 10);
  s.write(1, 11);
  s.write(2, 12);
  assert.strictEqual(s.sstables.length, 0);
  assert.strictEqual(s.get(0), 10);
  s.write(3, 13);
  assert.strictEqual(s.sstables.length, 1);
  assert.strictEqual(s.memtable.size, 0);
  assert.strictEqual(s.flushCount, 1);
  assert.deepStrictEqual(s.scan(), [[0, 10], [1, 11], [2, 12], [3, 13]]);
  s.delete(1);
  assert.strictEqual(s.get(1), null);
  s.write(1, 99);
  assert.strictEqual(s.get(1), 99);
  assert.deepStrictEqual(s.scan(), [[0, 10], [1, 99], [2, 12], [3, 13]]);

  const s2 = new LSMStore(2);
  s2.write(5, 1);
  s2.write(5, 2);
  s2.delete(5);
  s2.write(6, 3);
  assert.strictEqual(s2.get(5), null);
  assert.strictEqual(s2.get(6), 3);
  s2.sstables = [mergeTables(s2.sstables)];
  assert.strictEqual(s2.get(5), null, "merge must not resurrect the tombstone");
  assert.deepStrictEqual(s2.scan(), [[6, 3]]);
  s2.write(5, 7);
  s2.write(8, 8);
  assert.strictEqual(s2.get(5), 7);
  assert.strictEqual(s2.get(8), 8);
  assert.deepStrictEqual(s2.scan(), [[5, 7], [6, 3], [8, 8]]);

  runWorkload();
  console.log("PASS");
}
