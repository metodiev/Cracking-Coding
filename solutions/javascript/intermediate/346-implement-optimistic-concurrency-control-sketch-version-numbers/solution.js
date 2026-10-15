/**
 * Task 346 - Optimistic concurrency control sketch with version numbers. (Tier: Intermediate, CTCI Ch 15)
 *
 * Versioned records: a write commits only if the expected version is still current, else the caller
 * re-reads and retries.  Like the other languages, this is a single-threaded, fully scripted
 * simulation of the interleaving - deterministic in every language - which makes the conflict
 * timing exact: two clients read version 1, A commits, B's stale commit is rejected, B re-reads
 * and wins.  A second script (three clients, round-robin scheduler) shows retry-until-success and
 * rules out lost updates.  Production OCC would compare-and-swap the version field.  (Node has no
 * shared-memory threads, so a simulation is also the only option here.)
 */
"use strict";

class VersionedStore {
  constructor() {
    this.records = new Map();            // key -> { value, version }
  }

  create(key, value) {
    if (this.records.has(key)) throw new Error("key already exists");
    this.records.set(key, { value, version: 1 });   // new records start at version 1
  }

  read(key) {
    if (!this.records.has(key)) throw new Error(`no such key: ${key}`);
    const { value, version } = this.records.get(key);
    return { value, version };
  }

  commit(key, newValue, expectedVersion) {
    if (!this.records.has(key)) return false;
    const record = this.records.get(key);
    if (record.version !== expectedVersion) return false;   // stale read - the caller retries
    record.value = newValue;
    record.version = expectedVersion + 1;
    return true;
  }
}

/** A client with a read snapshot; attempt() commits or reports a conflict. */
class Client {
  constructor(name, store, key) {
    this.name = name;
    this.store = store;
    this.key = key;
    this.snapshot = null;
    this.retries = 0;
  }

  read() {
    this.snapshot = this.store.read(this.key);
  }

  attempt(newValue) {
    const ok = this.store.commit(this.key, newValue, this.snapshot.version);
    if (!ok) this.retries++;
    return ok;
  }
}

module.exports = { VersionedStore, Client };

if (require.main === module) {
  const assert = require("node:assert");

  const store = new VersionedStore();
  store.create("counter", 10);

  // scripted interleaving: A and B both read version 1 before anyone writes
  const a = new Client("A", store, "counter");
  const b = new Client("B", store, "counter");
  a.read();
  b.read();
  assert.deepStrictEqual(a.snapshot, { value: 10, version: 1 }, "A read (10, 1)");
  assert.deepStrictEqual(b.snapshot, { value: 10, version: 1 }, "B read (10, 1)");

  assert.strictEqual(a.attempt(11), true, "A commits first");
  assert.deepStrictEqual(store.read("counter"), { value: 11, version: 2 });
  assert.strictEqual(b.attempt(11), false, "B's stale commit is rejected");
  assert.strictEqual(b.retries, 1, "B counted one retry");
  assert.deepStrictEqual(store.read("counter"), { value: 11, version: 2 }, "rejected write changed nothing");

  b.read();
  assert.deepStrictEqual(b.snapshot, { value: 11, version: 2 }, "B re-read");
  assert.strictEqual(b.attempt(12), true, "B retries and wins");
  assert.strictEqual(a.retries, 0);
  assert.strictEqual(b.retries, 1);
  assert.deepStrictEqual(store.read("counter"), { value: 12, version: 3 }, "final state: 10 + 1 + 1");

  assert.strictEqual(store.commit("counter", 99, 1), false, "stale commit long after is rejected");
  assert.strictEqual(store.read("counter").value, 12, "value unchanged");
  assert.strictEqual(store.commit("missing", 1, 1), false, "unknown key never commits");
  assert.throws(() => store.create("counter", 0), Error, "duplicate create throws");

  // three clients, deterministic round-robin scheduler, retry until success
  store.create("total", 0);
  const clients = ["X", "Y", "Z"].map((name) => new Client(name, store, "total"));
  for (const client of clients) client.read();     // all read version 1 before anyone writes
  let pending = [...clients];
  let rounds = 0;
  while (pending.length > 0) {
    rounds++;
    const stillPending = [];
    for (const client of pending) {
      if (!client.attempt(client.snapshot.value + 1)) {
        client.read();                             // conflict -> refresh, retry later
        stillPending.push(client);
      }
    }
    pending = stillPending;
  }
  assert.strictEqual(rounds, 3, "X, then Y, then Z");
  assert.deepStrictEqual(clients.map((client) => client.retries), [0, 1, 2], "round-robin retry counts");
  assert.deepStrictEqual(store.read("total"), { value: 3, version: 4 }, "no lost update: 0 + 3 increments");
  console.log("PASS");
}
