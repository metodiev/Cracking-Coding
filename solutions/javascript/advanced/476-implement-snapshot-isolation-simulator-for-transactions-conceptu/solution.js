/** Task 476 - Implement snapshot isolation simulator for transactions (conceptual). (Tier: Advanced, CTCI Ch 14 — Databases) */
"use strict";

const KEYS = ["a", "b", "c", "d"];
const INITIAL = { a: 0, b: 1, c: 2, d: 3 };

class Txn {
  constructor(id, snapshot) {
    this.id = id;
    this.snapshot = snapshot;
    this.writes = {};
    this.state = "active";
  }
}

class MVCCStore {
  constructor(initial) {
    this.versions = {}; // key -> [[ts, value]]
    this.commitCounter = 0;
    this.nextTid = 0;
    for (const [key, value] of Object.entries(initial)) this.appendCommitted(key, value);
  }

  appendCommitted(key, value) {
    this.commitCounter += 1;
    if (!this.versions[key]) this.versions[key] = [];
    this.versions[key].push([this.commitCounter, value]);
  }

  begin() {
    this.nextTid += 1;
    return new Txn(`T${this.nextTid}`, this.commitCounter);
  }

  read(txn, key) {
    if (Object.prototype.hasOwnProperty.call(txn.writes, key)) return txn.writes[key];
    let best;
    for (const [ts, value] of this.versions[key] || []) {
      if (ts <= txn.snapshot) best = value; // only commit-time state as of BEGIN
    }
    return best;
  }

  write(txn, key, value) {
    txn.writes[key] = value;
  }

  commit(txn) {
    for (const key of Object.keys(txn.writes)) {
      for (const [ts] of this.versions[key] || []) {
        if (ts > txn.snapshot) {
          txn.state = "aborted";
          return false;
        }
      }
    }
    this.commitCounter += 1;
    const ts = this.commitCounter;
    for (const [key, value] of Object.entries(txn.writes)) {
      if (!this.versions[key]) this.versions[key] = [];
      this.versions[key].push([ts, value]);
    }
    txn.state = "committed";
    return true;
  }

  materialize() {
    const out = {};
    for (const [key, versions] of Object.entries(this.versions)) {
      let latest;
      for (const [, value] of versions) latest = value;
      out[key] = latest;
    }
    return out;
  }
}

function lcg(seed) {
  return (seed * 75 + 74) % 65537;
}

function randomScript(count, seed) {
  const script = [];
  for (let i = 0; i < count; i++) {
    seed = lcg(seed);
    const action = seed % 5;
    seed = lcg(seed);
    const pick = seed;
    seed = lcg(seed);
    const key = KEYS[seed % KEYS.length];
    seed = lcg(seed);
    const value = seed % 100;
    script.push([action, pick, key, value]);
  }
  return script;
}

function runRandomPhase() {
  const assert = require("node:assert");
  const store = new MVCCStore(INITIAL);
  const committed = { ...INITIAL };
  const versionTs = {};
  let ts = 1;
  for (const key of Object.keys(INITIAL)) versionTs[key] = ts++;
  let counter = store.commitCounter;
  assert.strictEqual(counter, 4);
  const live = new Map(); // tid -> {txn, snapshot copy, writes}
  let nextTid = 0;
  for (const [action, pick, key, value] of randomScript(60, 97531)) {
    if (action === 0 || live.size === 0) {
      nextTid += 1;
      const txn = store.begin();
      assert.strictEqual(txn.snapshot, counter, "snapshot pinned at BEGIN");
      live.set(txn.id, { txn, snapshot: { ...committed }, writes: {} });
    } else {
      const ids = [...live.keys()].sort();
      const tid = ids[pick % ids.length];
      const rec = live.get(tid);
      const txn = rec.txn;
      if (action === 1 || action === 4) {
        const got = store.read(txn, key);
        const expected = Object.prototype.hasOwnProperty.call(rec.writes, key) ? rec.writes[key] : rec.snapshot[key];
        assert.strictEqual(got, expected, `random read ${tid} ${key}`);
      } else if (action === 2) {
        store.write(txn, key, value);
        rec.writes[key] = value;
      } else {
        let success = true;
        for (const k of Object.keys(txn.writes)) {
          if ((versionTs[k] === undefined ? -1 : versionTs[k]) > txn.snapshot) success = false;
        }
        const got = store.commit(txn);
        assert.strictEqual(got, success, `random commit ${tid}`);
        if (got) {
          counter = store.commitCounter;
          for (const [k, v] of Object.entries(txn.writes)) {
            versionTs[k] = counter;
            committed[k] = v;
          }
        }
        live.delete(tid);
      }
    }
    assert.deepStrictEqual(store.materialize(), committed);
    assert.strictEqual(store.commitCounter, counter);
  }
  for (const rec of live.values()) {
    for (const key of KEYS) {
      const got = store.read(rec.txn, key);
      const expected = Object.prototype.hasOwnProperty.call(rec.writes, key) ? rec.writes[key] : rec.snapshot[key];
      assert.strictEqual(got, expected, "leftover txn read");
    }
  }
  assert.deepStrictEqual(store.materialize(), committed);
}

module.exports = { MVCCStore, Txn, randomScript, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  const store = new MVCCStore(INITIAL); // versions a@1, b@2, c@3, d@4
  const t1 = store.begin(); // snapshot 4
  store.write(t1, "a", 99);
  const t2 = store.begin(); // snapshot 4
  const t3 = store.begin(); // snapshot 4
  assert.strictEqual(store.read(t2, "a"), 0, "no dirty read of t1's uncommitted 99");
  store.write(t2, "a", 100);
  assert.ok(store.commit(t1), "first committer wins");
  assert.ok(!store.commit(t2), "write-write conflict must abort the loser");
  assert.strictEqual(store.read(t3, "a"), 0, "snapshot pinned before t1's commit");
  assert.strictEqual(store.read(t3, "a"), 0, "repeatable read");
  assert.ok(store.commit(t3), "read-only txn commits");
  const t4 = store.begin();
  assert.strictEqual(store.read(t4, "a"), 99, "fresh snapshot sees the commit");

  const t5 = store.begin();
  store.write(t5, "d", 6);
  assert.strictEqual(store.read(t5, "d"), 6, "read-your-writes");
  assert.ok(store.commit(t5));
  const t6 = store.begin();
  assert.strictEqual(store.read(t6, "d"), 6);

  const t7 = store.begin();
  const t8 = store.begin();
  assert.strictEqual(store.read(t7, "d"), 6);
  assert.strictEqual(store.read(t8, "d"), 6);
  store.write(t7, "d", 7);
  store.write(t8, "d", 8);
  assert.ok(store.commit(t7));
  assert.ok(!store.commit(t8), "lost update must be prevented");
  assert.strictEqual(store.read(store.begin(), "d"), 7);
  assert.deepStrictEqual(store.materialize(), { a: 99, b: 1, c: 2, d: 7 });

  // Write skew (documented SI anomaly): disjoint write sets both commit (0 = "on", 1 = "off").
  const docs = new MVCCStore({ x: 0, y: 0 });
  const t9 = docs.begin();
  const t10 = docs.begin();
  assert.strictEqual(docs.read(t9, "x"), 0);
  assert.strictEqual(docs.read(t9, "y"), 0);
  assert.strictEqual(docs.read(t10, "x"), 0);
  assert.strictEqual(docs.read(t10, "y"), 0);
  docs.write(t9, "x", 1);
  docs.write(t10, "y", 1);
  assert.ok(docs.commit(t9) && docs.commit(t10), "write skew is allowed under SI");
  assert.deepStrictEqual(docs.materialize(), { x: 1, y: 1 }, "the on-call invariant is broken");
  const fresh = docs.begin();
  assert.strictEqual(docs.read(fresh, "x"), 1);
  assert.strictEqual(docs.read(fresh, "y"), 1);

  runRandomPhase();
  console.log("PASS");
}
