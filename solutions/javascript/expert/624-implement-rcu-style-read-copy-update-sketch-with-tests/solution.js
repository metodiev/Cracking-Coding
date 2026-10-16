/** Task 624 - Implement RCU-style read-copy-update sketch with tests. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * RCU separates "publish" from "reclaim". A writer builds a new *immutable* version, swaps the
 * pointer (rcu_assign_pointer) and defers freeing the old one (call_rcu) until a *grace period*
 * has passed - that is, until every reader that could still hold the old version has left its
 * read-side critical section. Readers then pay no atomic at all: they copy the pointer once and
 * walk a structure nobody ever mutates in place.
 *
 * Simplifications (the task asks for a sketch): no threads, timers or preemption. Grace periods
 * are explicit - synchronize() snapshots the active readers and the deferred callbacks run when
 * the last of them exits; a reader entering after synchronize() must not delay reclamation. The
 * published structure is a persistent (path-copying) treap keyed by FNV-1a priority, so versions
 * share untouched nodes and a snapshot stays intact after any number of updates.
 *
 * Verified: (1) a reader in a section sees a stable snapshot while versions are published;
 * (2) no version is reclaimed while a reader could hold it, and the eager (no grace period)
 * control violates exactly that on the same schedule; (3) late readers do not delay reclamation;
 * (4) after the final period the structure matches the sequential model and every replaced
 * version was reclaimed exactly once.
 */
"use strict";

const FNV_PRIME = 0x01000193;
function fnv1a(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), FNV_PRIME) >>> 0;
  return hash >>> 0;
}

const mk = (key, value, left, right, prio) => Object.freeze({ key, value, left, right, prio });
const nodeFor = (key, value) => mk(key, value, null, null, fnv1a(String(key)));

/** Split by a key predicate, copying only the nodes on the two search paths. */
function splitBy(tree, pred) {
  if (tree === null) return [null, null];
  if (pred(tree.key)) {
    const [left, right] = splitBy(tree.right, pred);
    return [mk(tree.key, tree.value, tree.left, left, tree.prio), right];
  }
  const [left, right] = splitBy(tree.left, pred);
  return [left, mk(tree.key, tree.value, right, tree.right, tree.prio)];
}

function merge(a, b) {
  if (a === null) return b;
  if (b === null) return a;
  if (a.prio > b.prio) return mk(a.key, a.value, a.left, merge(a.right, b), a.prio);
  return mk(b.key, b.value, merge(a, b.left), b.right, b.prio);
}

/** Cut at a key boundary so the equal key (if present) is returned separately, not copied. */
function cut(tree, key) {
  const text = String(key);
  const [lt, ge] = splitBy(tree, (k) => k < text);
  const [eq, gt] = splitBy(ge, (k) => k <= text);
  return [lt, eq, gt];
}

function insert(tree, key, value) {
  const [lt, , gt] = cut(tree, key);
  return merge(merge(lt, nodeFor(key, value)), gt);
}

function remove(tree, key) {
  const [lt, , gt] = cut(tree, key);
  return merge(lt, gt);
}

function get(tree, key) {
  const text = String(key);
  let node = tree;
  while (node !== null) {
    if (text === node.key) return node.value;
    node = text < node.key ? node.left : node.right;
  }
  return undefined;
}

function keysInOrder(tree, out = []) {
  if (tree === null) return out;
  keysInOrder(tree.left, out);
  out.push(tree.key);
  keysInOrder(tree.right, out);
  return out;
}

function toObject(tree) {
  const out = {};
  for (const key of keysInOrder(tree)) out[key] = get(tree, key);
  return out;
}

/** Simulated grace period. It completes when every reader that was active when it started has left
 *  its section; the callbacks queued before it then run. Later readers cannot delay it. */
class RcuDomain {
  constructor() {
    this.seq = 0;
    this.readers = new Set();
    this.queue = [];
    this.periods = [];
    this.completedPeriods = 0;
    this.freeLog = [];
  }
  enterRead(name) {
    const id = `${name}#${++this.seq}`;
    this.readers.add(id);
    return id;
  }
  exitRead(id) {
    if (!this.readers.delete(id)) throw new Error(`read section ${id} is not active`);
    this.pump();
  }
  callRcu(fn, tag) {
    this.queue.push({ fn, tag });
  }
  synchronize() {
    if (this.queue.length === 0) return 0;
    const period = { wait: new Set(this.readers), batch: this.queue, done: false };
    this.queue = [];
    this.periods.push(period);
    this.pump();
    return period.done ? period.batch.length : 0;
  }
  pump() {
    for (const period of this.periods) {
      if (period.done) continue;
      let blocked = false;
      for (const reader of period.wait) if (this.readers.has(reader)) blocked = true;
      if (blocked) continue;
      period.done = true;
      this.completedPeriods++;
      for (const { fn, tag } of period.batch) {
        this.freeLog.push(tag);
        fn();
      }
    }
  }
  pendingPeriods() {
    return this.periods.filter((period) => !period.done).length;
  }
  activeReaders() {
    return this.readers.size;
  }
}

/** An RCU-protected map. `eager` selects the broken variant that skips the grace period; it exists
 *  only so a check can show that reclamation without RCU corrupts a concurrent reader. */
class RcuMap {
  constructor(domain, entries = [], { eager = false } = {}) {
    this.domain = domain;
    this.eager = eager;
    this.version = 0;
    this.ptr = null;
    for (const [key, value] of entries) this.ptr = insert(this.ptr, key, value);
    this.inUse = new Map();
    this.freedVersions = [];
    this.violations = 0;
    this.writing = false;
  }
  readBegin(name) {
    const handle = { id: this.domain.enterRead(name), version: this.version, snapshot: this.ptr };
    this.inUse.set(handle.version, (this.inUse.get(handle.version) || 0) + 1);
    return handle;
  }
  readEnd(handle) {
    const left = this.inUse.get(handle.version) - 1;
    if (left === 0) this.inUse.delete(handle.version);
    else this.inUse.set(handle.version, left);
    this.domain.exitRead(handle.id);
    return left;
  }
  readSection(name, fn) {
    const handle = this.readBegin(name);
    try {
      return fn(handle.snapshot, handle.version);
    } finally {
      this.readEnd(handle);
    }
  }
  update(key, value) {
    if (this.writing) throw new Error("RCU permits a single writer at a time");
    this.writing = true;
    const oldVersion = this.version;
    const next = value === undefined ? remove(this.ptr, key) : insert(this.ptr, key, value);
    this.ptr = next; // rcu_assign_pointer: readers keep walking the version they already loaded
    this.version++;
    if (this.eager) this.reclaim(oldVersion);
    else this.domain.callRcu(() => this.reclaim(oldVersion), `v${oldVersion}`);
    this.writing = false;
    return this.version;
  }
  reclaim(version) {
    if (this.inUse.has(version)) this.violations++; // freed while a reader could still hold it
    this.freedVersions.push(version);
  }
  synchronize() {
    return this.domain.synchronize();
  }
}

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

function ensure(ok, message) {
  if (!ok) throw new Error(`check failed: ${message}`);
}

/** Interleaved writer/reader schedule behind the convergence and violation checks. A reader takes
 *  its snapshot, a writer then publishes between the reader's two looks, and the two must agree. */
function runRcuSchedule(seed, steps, { eager = false } = {}) {
  const domain = new RcuDomain();
  const map = new RcuMap(domain, [["a", 0], ["b", 0]], { eager });
  const model = new Map([["a", 0], ["b", 0]]);
  const keys = ["a", "b", "c", "d"];
  let s = seed;
  let readers = 0;
  let tornReads = 0;
  for (let i = 0; i < steps; i++) {
    s = lcg(s);
    const choice = s % 100;
    const key = keys[s % keys.length];
    if (choice < 35) {
      const value = (s >> 3) % 50;
      map.update(key, value);
      model.set(key, value);
      continue;
    }
    if (choice < 45) {
      map.update(key, undefined);
      model.delete(key);
      continue;
    }
    if (choice >= 80) {
      map.synchronize();
      continue;
    }
    const handle = map.readBegin(`r${readers}`);
    const before = JSON.stringify(toObject(handle.snapshot));
    if (readers % 3 === 0) {
      s = lcg(s);
      const key2 = keys[s % keys.length];
      const value2 = (s >> 5) % 50;
      map.update(key2, value2);
      model.set(key2, value2);
    }
    if (before !== JSON.stringify(toObject(handle.snapshot))) tornReads++;
    readers++;
    map.readEnd(handle);
    ensure(domain.activeReaders() === 0 && map.inUse.size === 0, "reader bookkeeping leaked");
  }
  map.synchronize();
  return { domain, map, model, tornReads, sections: readers };
}

module.exports = {
  fnv1a, splitBy, merge, insert, remove, get, keysInOrder, toObject,
  RcuDomain, RcuMap, runRcuSchedule, lcg,
};

if (require.main === module) {
  const assert = require("node:assert");
  // --- persistent treap: old versions survive every update --------------------------------
  let tree = null;
  for (const [key, value] of [["b", 2], ["a", 1], ["d", 4], ["c", 3], ["a", 9]]) tree = insert(tree, key, value);
  assert.deepStrictEqual(keysInOrder(tree), ["a", "b", "c", "d"]);
  assert.strictEqual(get(tree, "a"), 9, "re-inserting a key replaces its value");
  assert.strictEqual(get(insert(tree, "a", 100), "a"), 100);
  assert.strictEqual(get(tree, "a"), 9, "the old version is untouched (path copying)");
  assert.deepStrictEqual(keysInOrder(remove(remove(tree, "b"), "a")), ["c", "d"]);
  assert.deepStrictEqual(keysInOrder(null), []);
  assert.strictEqual(get(tree, "zz"), undefined);
  assert.throws(() => { tree.value = 5; }, TypeError, "published versions are frozen");
  assert.throws(() => { tree.left.right = null; }, TypeError);
  // --- a reader mid-section is stable while three versions are published -------------------
  const map = new RcuMap(new RcuDomain(), [["x", 0], ["y", 0]]);
  const observed = map.readSection("slow", (snapshot, version) => {
    const before = [get(snapshot, "x"), get(snapshot, "y")];
    map.update("x", 1);
    map.update("y", 2);
    map.update("x", 3);
    return { version, before, after: [get(snapshot, "x"), get(snapshot, "y")] };
  });
  assert.deepStrictEqual(observed.before, observed.after, "no torn read inside a read section");
  assert.deepStrictEqual(observed.before, [0, 0], "the reader keeps the version it loaded");
  assert.strictEqual(observed.version, 0); assert.strictEqual(map.version, 3, "three versions published");
  assert.deepStrictEqual(toObject(map.ptr), { x: 3, y: 2 }, "the latest version has the new values");
  // --- a grace period waits for the readers that were active when it started --------------
  const domain = new RcuDomain();
  const graceMap = new RcuMap(domain, [["x", 0]]);
  const slow = graceMap.readBegin("slow");
  graceMap.update("x", 1); // queues the deferred free of v0
  assert.strictEqual(graceMap.synchronize(), 0, "the period cannot complete while 'slow' is inside");
  assert.strictEqual(domain.pendingPeriods(), 1);
  assert.deepStrictEqual(graceMap.freedVersions, [], "nothing is reclaimed before the period ends");
  graceMap.update("x", 2); // queues v1 behind the running period
  assert.deepStrictEqual(graceMap.freedVersions, [], "a later writer waits for the same period");
  graceMap.readEnd(slow);
  assert.deepStrictEqual(graceMap.freedVersions, [0], "the deferred free runs the moment the reader exits");
  assert.strictEqual(domain.pendingPeriods(), 0);
  assert.strictEqual(domain.freeLog[0], "v0");
  // --- a reader that enters after synchronize() does not delay reclamation ----------------
  graceMap.update("x", 3);
  assert.strictEqual(graceMap.synchronize(), 2, "with no active reader the period completes at once");
  assert.deepStrictEqual(graceMap.freedVersions, [0, 1, 2]);
  const late = graceMap.readBegin("late");
  assert.strictEqual(late.version, 3, "the late reader reads the current version");
  assert.deepStrictEqual(graceMap.freedVersions, [0, 1, 2], "the late reader does not reopen the period");
  graceMap.update("x", 4);
  assert.strictEqual(graceMap.synchronize(), 0, "a new period does wait for the now-active reader");
  assert.strictEqual(domain.pendingPeriods(), 1);
  assert.deepStrictEqual(graceMap.freedVersions, [0, 1, 2]); // v3 is still held by no one
  graceMap.readEnd(late);
  assert.deepStrictEqual(graceMap.freedVersions, [0, 1, 2, 3], "the second period drains on exit");
  assert.strictEqual(graceMap.synchronize(), 0, "nothing left to synchronize");
  assert.strictEqual(graceMap.violations, 0, "no version may be reclaimed in use");
  assert.strictEqual(graceMap.freedVersions.length, graceMap.version, "all versions reclaimed");
  // --- randomized schedules: convergence, no violations, no leaks -------------------------
  let seed = 987654321;
  let sections = 0;
  for (let trial = 0; trial < 10; trial++) {
    seed = lcg(seed);
    const run = runRcuSchedule(seed, 120);
    assert.strictEqual(run.tornReads, 0, `trial ${trial}: RCU reader observed a torn version`);
    assert.strictEqual(run.map.violations, 0, `trial ${trial}: version reclaimed while in use`);
    assert.strictEqual(run.domain.pendingPeriods(), 0, `trial ${trial}: pending grace period left`);
    assert.strictEqual(run.map.inUse.size, 0, `trial ${trial}: version still referenced`);    assert.deepStrictEqual(toObject(run.map.ptr), Object.fromEntries(run.model), `trial ${trial}: divergence`);
    assert.deepStrictEqual(run.map.freedVersions, Array.from({ length: run.map.version }, (_, i) => i),
      `trial ${trial}: every replaced version is reclaimed exactly once`);
    assert.ok(run.sections > 20, `trial ${trial}: schedule must exercise many read sections`);
    sections += run.sections;
  }
  // --- the same schedule without a grace period corrupts readers --------------------------
  seed = lcg(seed);
  const eagerRun = runRcuSchedule(seed, 120, { eager: true });
  assert.ok(eagerRun.map.violations > 0, "eager reclamation must free a version that is still in use");
  assert.strictEqual(eagerRun.domain.completedPeriods, 0, "the eager map never runs a grace period");
  // --- writer and reader misuse is rejected ------------------------------------------------
  const strictMap = new RcuMap(new RcuDomain(), [["k", 1]]);
  assert.throws(() => strictMap.readEnd({ id: "nope", version: 0 }), Error, "unknown read section");
  const nested = strictMap.readBegin("outer");
  assert.throws(() => { strictMap.writing = true; strictMap.update("k", 3); }, Error, "second writer rejected");
  strictMap.writing = false;
  assert.strictEqual(strictMap.update("k", 2), 1);
  strictMap.readEnd(nested);
  assert.strictEqual(strictMap.readSection("again", (snapshot) => get(snapshot, "k")), 2);

  console.log(`rcu sections=${sections} torn-reads=0 eager-violations=${eagerRun.map.violations} grace-periods=${domain.completedPeriods}`);
  console.log("PASS");
}
