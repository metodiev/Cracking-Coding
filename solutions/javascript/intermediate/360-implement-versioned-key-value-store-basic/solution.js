/**
 * Task 360 - Implement versioned key value store basic. (Tier: Intermediate, CTCI Ch 14)
 *
 * Versioned key-value store: every put or delete consumes exactly one monotonically
 * increasing version (1, 2, 3, ...). Per key a history of [version, value-or-TOMBSTONE]
 * entries is kept, so get(key, v) time-travels: it returns the value of the greatest
 * version <= v, or missing (null) when that entry is a tombstone. Tombstone policy
 * (documented): delete never erases history, it appends a TOMBSTONE; get(key) on a deleted
 * key reports missing while get(key, v) for v before the delete still returns the old
 * value; a later put resurrects the key at a fresh version. An explicit version must lie
 * in 1..current (RangeError otherwise); the automatic latest read is allowed on an empty
 * store. Reads never consume a version, so they are idempotent. A random LCG operation
 * script is cross-checked against a naive per-version snapshot list (snapshot[v-1] is the
 * whole map state after the v-th write).
 */
"use strict";

const SCRIPT_OPS = 200;
const KEYSPACE = 8;
const TOMBSTONE = Symbol("tombstone");

class VersionedKV {
  constructor() {
    this.version = 0;
    this.history = new Map();          // key -> array of [version, value|TOMBSTONE]
  }

  latestVersion() {
    return this.version;
  }

  put(key, value) {
    this.version += 1;
    this._entries(key).push([this.version, value]);
    return this.version;
  }

  delete(key) {
    this.version += 1;
    this._entries(key).push([this.version, TOMBSTONE]);
    return this.version;
  }

  /** Value of key as of `version`, else the latest; null means missing (incl. tombstoned). */
  get(key, version = null) {
    let atVersion = version;
    if (atVersion === null) {
      atVersion = this.version;
    } else if (!Number.isInteger(atVersion) || atVersion < 1 || atVersion > this.version) {
      throw new RangeError(`version must be in 1..${this.version}, got ${atVersion}`);
    }
    if (atVersion < 1) return null;
    let latestEntry = null;
    let found = false;
    for (const [entryVersion, value] of this.history.get(key) || []) {
      if (entryVersion <= atVersion) {
        latestEntry = value;
        found = true;
      } else {
        break;
      }
    }
    if (!found || latestEntry === TOMBSTONE) return null;
    return latestEntry;
  }

  _entries(key) {
    let entries = this.history.get(key);
    if (!entries) {
      entries = [];
      this.history.set(key, entries);
    }
    return entries;
  }
}

function main() {
  const assert = require("node:assert");

  // Canonical script from the task brief.
  const store = new VersionedKV();
  assert.strictEqual(store.get("a"), null, "empty store reports missing");
  const v1 = store.put("a", 1);
  const v2 = store.put("a", 2);
  assert.strictEqual(v1, 1, "first version");
  assert.strictEqual(v2, 2, "versions increase by exactly one");
  assert.strictEqual(store.get("a", v1), 1, "get(a, v1) == 1");
  assert.strictEqual(store.get("a", v2), 2, "get(a, v2) == 2");
  assert.strictEqual(store.get("a"), 2, "latest value");
  const v3 = store.delete("a");
  assert.strictEqual(v3, 3, "delete consumes a version");
  assert.strictEqual(store.get("a"), null, "get(a, latest) == missing");
  assert.strictEqual(store.get("a", 2), 2, "old value visible before the tombstone");
  assert.strictEqual(store.get("a", 3), null, "tombstone visible at its own version");
  const v4 = store.put("a", 5);
  assert.strictEqual(v4, 4, "put resurrects with a fresh version");
  assert.strictEqual(store.get("a"), 5, "resurrected value");
  assert.strictEqual(store.get("a", 3), null, "history unchanged by resurrection");
  assert.strictEqual(store.get("a", 1), 1, "history unchanged by resurrection (v1)");
  assert.strictEqual(store.get("missing"), null, "missing keys");
  assert.strictEqual(store.get("missing", 2), null, "missing keys at a version");

  // Idempotent reads: reads never consume a version and repeat identically.
  const before = store.latestVersion();
  for (let i = 0; i < 100; i++) {
    assert.strictEqual(store.get("a"), 5, "idempotent latest read");
    assert.strictEqual(store.get("a", 1), 1, "idempotent historical read");
    assert.strictEqual(store.get("a", 3), null, "idempotent tombstone read");
  }
  assert.strictEqual(store.latestVersion(), before, "no version consumed by reads");
  assert.strictEqual(before, 4, "hand-checked current version");

  // Versions outside 1..current are invalid for explicit reads.
  for (const bad of [0, -1, 5]) {
    assert.throws(() => store.get("a", bad), RangeError, `invalid version must raise: ${bad}`);
  }

  // Random LCG operation script cross-checked against a naive snapshot list.
  const scripted = new VersionedKV();
  const snapshots = [];                        // snapshots[v-1] == state after write v
  const reference = new Map();
  const allKeys = new Set();
  let seed = 20260;
  for (let step = 0; step < SCRIPT_OPS; step++) {
    seed = (seed * 75 + 74) % 65537;
    const key = `k${seed % KEYSPACE}`;
    seed = (seed * 75 + 74) % 65537;
    let version;
    if (seed % 4 === 0) {
      version = scripted.delete(key);
      reference.delete(key);
    } else {
      version = scripted.put(key, seed % 100);
      reference.set(key, seed % 100);
    }
    assert.strictEqual(version, step + 1, "one version per write");
    snapshots.push(new Map(reference));
    allKeys.add(key);
  }

  const current = scripted.latestVersion();
  assert.strictEqual(current, SCRIPT_OPS, "all writes counted");
  for (const key of [...allKeys, "absent"]) {
    const expectedLatest = reference.has(key) ? reference.get(key) : null;
    assert.strictEqual(scripted.get(key), expectedLatest, `latest view of ${key}`);
    for (let version = 1; version <= current; version++) {
      const snapshot = snapshots[version - 1];
      const expected = snapshot.has(key) ? snapshot.get(key) : null;
      assert.strictEqual(scripted.get(key, version), expected,
        `time travel of ${key} at version ${version}`);
    }
  }
  assert.strictEqual(scripted.latestVersion(), current, "the mass of reads consumed no version");
}

module.exports = { VersionedKV };

if (require.main === module) {
  try {
    main();
    console.log("PASS");
  } catch (err) {
    console.error(err);
    process.exit(1);
  }
}
