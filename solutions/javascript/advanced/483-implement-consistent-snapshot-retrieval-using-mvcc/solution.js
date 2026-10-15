/** Task 483 - Implement consistent snapshot retrieval using MVCC. (Tier: Advanced, CTCI Ch 14)
 *
 * Multi-version store: every key holds the full list of (commit_ts, value) versions produced by
 * whole-transaction commits (all keys of one transaction share a single commit timestamp). A read
 * issued at snapshot timestamp s returns, per key, the newest version with commit_ts <= s, i.e. the
 * state as of a single point in time even if newer commits exist. Because one transaction stamps
 * all of its writes with the same ts, a snapshot never observes a torn transaction.
 *
 * The scripted history is seeded with base versions at ts = 0, then 12 LCG-driven transactions at
 * ts = 1..12. A slow reader opens a snapshot at ts = 3 after commits 1..8 exist, reads all keys,
 * waits while 9..12 commit, and re-reads: both reads must equal the brute-force oracle (a linear
 * scan of raw history), while fresh snapshots keep seeing the newest writers.
 */
"use strict";

class VersionedStore {
  constructor(keys) {
    this.versions = new Map();
    for (const key of keys) this.versions.set(key, []);
    this.commitCount = 0;
  }

  commit(ts, writes) {
    for (const [key, value] of writes) this.versions.get(key).push([ts, value]);
    this.commitCount += 1;
  }

  get(key, snapshotTs) {
    const entries = this.versions.get(key);
    let lo = 0;
    let hi = entries.length - 1;
    let best = null;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (entries[mid][0] <= snapshotTs) {
        best = entries[mid][1];
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return best;
  }

  snapshot(ts) {
    return new Snapshot(this, ts);
  }

  headTs() {
    let best = 0;
    for (const entries of this.versions.values()) {
      for (const [ts] of entries) best = Math.max(best, ts);
    }
    return best;
  }
}

class Snapshot {
  constructor(store, ts) {
    this.store = store;
    this.ts = ts;
  }

  get(key) {
    return this.store.get(key, this.ts);
  }
}

function oracleValue(history, key, snapshotTs) {
  let value = null;
  for (const [ts, hkey, hvalue] of history) {
    if (hkey === key && ts <= snapshotTs) value = hvalue;
  }
  return value;
}

function buildHistory(base, transactions, startSeed) {
  let seed = startSeed;
  const history = [];
  for (const key of Object.keys(base).sort()) history.push([0, key, base[key]]);
  for (let t = 1; t <= transactions; t++) {
    seed = (seed * 75 + 74) % 65537;
    const count = 1 + (t % 2);
    for (let w = 0; w < count; w++) {
      seed = (seed * 75 + 74) % 65537;
      const key = `k${seed % 5}`;
      seed = (seed * 75 + 74) % 65537;
      const value = 100 + (seed % 900);
      history.push([t, key, value]);
    }
  }
  return history;
}

function groupTransactions(history) {
  const grouped = [];
  for (const [ts, key, value] of history) {
    if (grouped.length === 0 || grouped[grouped.length - 1][0] !== ts) grouped.push([ts, []]);
    grouped[grouped.length - 1][1].push([key, value]);
  }
  return grouped;
}

module.exports = { VersionedStore, Snapshot };

if (require.main === module) {
  const assert = require("node:assert");
  const base = {};
  for (let i = 0; i < 5; i++) base[`k${i}`] = 10 * i + 1;
  const history = buildHistory(base, 12, 483);
  assert.strictEqual(history.length, 5 + 18);
  const transactions = groupTransactions(history);
  assert.strictEqual(transactions.length, 13);
  assert.strictEqual(transactions[0][0], 0);

  const store = new VersionedStore(Object.keys(base));
  for (let i = 0; i < 9; i++) store.commit(transactions[i][0], transactions[i][1]);
  assert.strictEqual(store.commitCount, 9);

  const reader = store.snapshot(3);
  const first = {};
  for (const key of Object.keys(base)) first[key] = reader.get(key);

  for (let i = 9; i < transactions.length; i++) store.commit(transactions[i][0], transactions[i][1]);
  for (const key of Object.keys(base)) {
    assert.deepStrictEqual(reader.get(key), first[key], `repeatable read ${key}`);
    assert.deepStrictEqual(first[key], oracleValue(history, key, 3), `snapshot oracle ${key}`);
    assert.notStrictEqual(first[key], null, `visible ${key}`);
  }

  assert.strictEqual(store.headTs(), 12);
  const fresh = store.snapshot(store.headTs());
  let moved = false;
  for (const key of Object.keys(base)) {
    assert.deepStrictEqual(fresh.get(key), oracleValue(history, key, 12), `head oracle ${key}`);
    if (first[key] !== fresh.get(key)) moved = true;
  }
  assert.ok(moved, "writers moved state");

  for (const key of Object.keys(base)) {
    for (let ts = -1; ts <= 13; ts++) {
      assert.deepStrictEqual(store.get(key, ts), oracleValue(history, key, ts), `isolation ${key} ${ts}`);
    }
  }

  for (let i = 1; i < transactions.length; i++) {
    const ts = transactions[i][0];
    const writes = transactions[i][1];
    const current = store.snapshot(ts);
    const previous = store.snapshot(ts - 1);
    const distinct = new Set();
    for (const [key, value] of writes) {
      distinct.add(key);
      assert.deepStrictEqual(current.get(key), value, `read your writes ${ts} ${key}`);
      assert.deepStrictEqual(previous.get(key), oracleValue(history, key, ts - 1), `prior ${ts} ${key}`);
    }
    assert.ok(distinct.size <= 2, `atomic ${ts}`);
  }

  const versions = {};
  for (const [key, entries] of store.versions) versions[key] = entries.length;
  assert.deepStrictEqual(versions, { k0: 3, k1: 6, k2: 4, k3: 3, k4: 7 });
  let total = 0;
  for (const count of Object.values(versions)) total += count;
  assert.strictEqual(total, 5 + 18);

  const countBefore = store.commitCount;
  store.commit(13, [["k0", 777]]);
  store.commit(14, [["k1", 888]]);
  assert.strictEqual(store.commitCount, countBefore + 2);
  assert.strictEqual(store.get("k0", 13), 777);
  assert.notStrictEqual(store.get("k0", 12), 777);
  assert.strictEqual(store.get("k1", 14), 888);
  assert.notStrictEqual(store.get("k1", 13), 888);
  console.log("PASS");
}
