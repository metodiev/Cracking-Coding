/** Task 484 - Implement simple key-value store with memtable + SSTable. (Tier: Advanced, CTCI Ch 14)
 *
 * LSM-style prototype with documented semantics:
 * - Writes land in an in-memory memtable (key -> value; deletes insert a null TOMBSTONE marker).
 * - When the memtable reaches `flush_threshold` entries it flushes to a new immutable SSTable:
 *   the records are written sorted by key and appended to the sstable list (oldest first).
 * - get(key) probes the memtable, then the SSTables from newest to oldest; the first record for the
 *   key wins. A tombstone means "deleted" (returns null) and must shadow older values in older
 *   SSTables - that is what makes deletes correct without rewriting data.
 * - compact() flushes the memtable, merges all SSTables newest-wins into a single one, and drops
 *   tombstones plus every record they shadow, so the result holds exactly the live keys.
 *
 * The test asserts get == oracle after every single operation of an LCG-driven 240-op script
 * (puts, gets, deletes) including tombstone shadowing across SSTables, then checks the
 * post-compaction layout and idempotence of a second compaction.
 */
"use strict";

class KVStore {
  constructor(flushThreshold) {
    this.flushThreshold = flushThreshold;
    this.memtable = new Map();
    this.sstables = [];
    this.flushes = 0;
    this.compactions = 0;
  }

  put(key, value) {
    this.memtable.set(key, value);
    this.maybeFlush();
  }

  delete(key) {
    this.memtable.set(key, null);
    this.maybeFlush();
  }

  maybeFlush() {
    if (this.memtable.size >= this.flushThreshold) this.flush();
  }

  flush() {
    if (this.memtable.size === 0) return;
    const records = Array.from(this.memtable.entries()).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    this.sstables.push(records);
    this.memtable.clear();
    this.flushes += 1;
  }

  get(key) {
    if (this.memtable.has(key)) return this.memtable.get(key);
    for (let i = this.sstables.length - 1; i >= 0; i--) {
      for (const [rkey, value] of this.sstables[i]) {
        if (rkey === key) return value;
      }
    }
    return null;
  }

  compact() {
    this.flush();
    const newest = new Map();
    for (let i = this.sstables.length - 1; i >= 0; i--) {
      for (const [key, value] of this.sstables[i]) {
        if (!newest.has(key)) newest.set(key, value);
      }
    }
    const live = Array.from(newest.entries())
      .filter(([, value]) => value !== null)
      .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    this.sstables = live.length > 0 ? [live] : [];
    this.compactions += 1;
  }
}

class Oracle {
  constructor(keys) {
    this.live = new Map();
    for (const key of keys) this.live.set(key, null);
  }

  put(key, value) {
    this.live.set(key, value);
  }

  delete(key) {
    this.live.delete(key);
  }

  get(key) {
    return this.live.has(key) ? this.live.get(key) : null;
  }
}

function lcgOps(count, seed, keyCount) {
  const ops = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    const kind = seed % 3;
    seed = (seed * 75 + 74) % 65537;
    const key = seed % keyCount;
    seed = (seed * 75 + 74) % 65537;
    const value = seed % 1000;
    ops.push([kind, key, value]);
  }
  return ops;
}

function recordValue(sstables, key) {
  for (const records of sstables) {
    for (const [rkey, value] of records) {
      if (rkey === key) return value;
    }
  }
  return "missing";
}

module.exports = { KVStore, Oracle };

if (require.main === module) {
  const assert = require("node:assert");

  let store = new KVStore(2);
  store.put("a", 1);
  store.put("b", 2);
  assert.strictEqual(store.flushes, 1);
  assert.strictEqual(store.memtable.size, 0);
  assert.strictEqual(store.get("a"), 1);
  assert.strictEqual(store.get("b"), 2);
  store.put("a", 5);
  store.flush();
  store.delete("b");
  assert.strictEqual(store.get("b"), null);
  store.flush();
  assert.strictEqual(store.get("a"), 5);
  assert.strictEqual(store.get("b"), null);
  assert.strictEqual(store.sstables.length, 3);
  assert.strictEqual(recordValue([store.sstables[2]], "b"), null);
  assert.strictEqual(recordValue([store.sstables[0]], "b"), 2);
  store.compact();
  assert.strictEqual(store.compactions, 1);
  assert.strictEqual(store.sstables.length, 1);
  assert.deepStrictEqual(store.sstables[0], [["a", 5]]);
  assert.strictEqual(store.get("a"), 5);
  assert.strictEqual(store.get("b"), null);
  store.compact();
  assert.deepStrictEqual(store.sstables[0], [["a", 5]]);

  const keys = [];
  for (let i = 0; i < 12; i++) keys.push(`k${i}`);
  store = new KVStore(4);
  const oracle = new Oracle(keys);
  let gets = 0;
  let tombstones = 0;
  const ops = lcgOps(240, 484, 12);
  for (let step = 0; step < ops.length; step++) {
    const [kind, keyIndex, value] = ops[step];
    const key = `k${keyIndex}`;
    if (kind === 0) {
      store.put(key, value);
      oracle.put(key, value);
    } else if (kind === 1) {
      assert.strictEqual(store.get(key), oracle.get(key), `direct get ${step}`);
      gets += 1;
    } else {
      store.delete(key);
      oracle.delete(key);
      tombstones += 1;
    }
    for (const probe of keys) {
      assert.strictEqual(store.get(probe), oracle.get(probe), `probe ${step} ${probe}`);
    }
    if (step % 60 === 59) {
      store.compact();
      for (const probe of keys) {
        assert.strictEqual(store.get(probe), oracle.get(probe), `post compact ${step} ${probe}`);
      }
    }
  }
  assert.strictEqual(gets, 84);
  assert.strictEqual(tombstones, 71);
  assert.strictEqual(store.flushes, 35);
  assert.strictEqual(store.compactions, 4);
  store.compact();
  assert.strictEqual(store.sstables.length, 1);
  const liveKeys = Array.from(oracle.live.keys())
    .filter((key) => oracle.live.get(key) !== null)
    .sort();
  const records = store.sstables[0];
  assert.strictEqual(records.length, liveKeys.length);
  for (let i = 0; i < records.length; i++) {
    assert.strictEqual(records[i][0], liveKeys[i]);
    assert.strictEqual(records[i][1], oracle.live.get(liveKeys[i]));
  }
  for (const probe of keys) {
    assert.strictEqual(store.get(probe), oracle.get(probe), `final probe ${probe}`);
  }
  console.log("PASS");
}
