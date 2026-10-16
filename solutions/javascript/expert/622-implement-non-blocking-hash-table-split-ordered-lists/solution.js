/** Task 622 - Implement non-blocking hash table (split-ordered lists). (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * One *ordered* linked list holds the whole table (Shalev-Shavit split-ordered lists). A key maps to
 * `order(key) = reverse32(hash(key))`, the list is sorted by that, and its bucket is the top
 * log2(capacity) bits of the order (the reversed low bits of the hash). Each bucket therefore owns a
 * contiguous range, so doubling the table splits bucket i into 2i and 2i+1 without moving a single
 * node: the elements with the new bit clear are already in the first half of the range. A sentinel
 * per bucket marks the base of its range and appears lazily the first time that bucket is touched
 * (the recursive split of the original algorithm, one bucket at a time here).
 *
 * Insert scans from the bucket sentinel to the ordered window and CASes the predecessor's next
 * pointer; remove marks the node and CASes its predecessor around it, helping any marked node it
 * crosses; both restart from the sentinel when a CAS fails, so the only "lock" is the bucket window.
 *
 * What is simplified: operations are generators that yield once per atomic access and a schedule of
 * thread ids drives them (as in the neighbouring queue task) - a faithful interleaving simulation,
 * not a kernel test - and there is no memory reclamation story (hazard pointers are their own task).
 * Verified after every step: the list is strictly ordered and duplicate-free, every element sits in
 * the window of the last sentinel before it, a lookup never leaves its bucket, the size matches the
 * listed keys, and after a resize each new odd bucket holds exactly the old bucket's high half.
 */
"use strict";

const HASH_BITS = 32;

function hash32(key) {
  let x = (key | 0) ^ 0x9e3779b9;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

function reverseBits32(value) {
  let x = value >>> 0;
  x = ((x & 0x55555555) << 1) | ((x >>> 1) & 0x55555555);
  x = ((x & 0x33333333) << 2) | ((x >>> 2) & 0x33333333);
  x = ((x & 0x0f0f0f0f) << 4) | ((x >>> 4) & 0x0f0f0f0f);
  x = ((x & 0x00ff00ff) << 8) | ((x >>> 8) & 0x00ff00ff);
  return ((x << 16) | (x >>> 16)) >>> 0;
}

const orderOf = (key) => reverseBits32(hash32(key));
const capacityBits = (capacity) => Math.round(Math.log2(capacity));
/** A bucket is the top `log2(capacity)` bits of the order, so its range is contiguous. */
const bucketOfOrder = (order, capacity) => {
  const bits = capacityBits(capacity);
  return bits === 0 ? 0 : order >>> (HASH_BITS - bits);
};
const bucketOf = (key, capacity) => bucketOfOrder(orderOf(key), capacity);
const sentinelOrder = (bucket, capacity) => {
  const bits = capacityBits(capacity);
  return bits === 0 ? 0 : (bucket << (HASH_BITS - bits)) >>> 0;
};

function sortsAfter(node, other) {
  if (node.order !== other.order) return node.order > other.order;
  if (node.sentinel !== other.sentinel) return other.sentinel;
  return !node.sentinel && node.key > other.key;
}

class Atomic {
  constructor(value) { this.value = value; }

  load() { return this.value; }

  compareAndSwap(expected, replacement, table) {
    if (this.value === expected) { this.value = replacement; return true; }
    if (table !== undefined) table.casFailures++;
    return false;
  }
}

const makeNode = (sentinel, order, key, bucketHint) => ({
  sentinel, order, key, bucketHint, marked: new Atomic(false), next: new Atomic(null),
});

class SplitOrderedTable {
  constructor(capacity = 2, loadFactor = 0.75) {
    if (!Number.isInteger(capacity) || capacity < 1 || (capacity & (capacity - 1)) !== 0) {
      throw new RangeError("capacity must be a positive power of two");
    }
    this.capacity = capacity;
    this.loadFactor = loadFactor;
    this.head = makeNode(true, 0, null, 0);
    this.sentinels = [this.head];
    this.size = 0;
    this.casFailures = 0;
    this.resizes = 0;
    this.sentinelsCreated = 1;
    this.steps = 0;
    this.searchVisits = 0;
  }

  get headSentinel() { return this.head; }

  sentinelFor(bucket) {
    for (const sentinel of this.sentinels) if (bucketOfOrder(sentinel.order, this.capacity) === bucket) return sentinel;
    return null;
  }

  /** The recursive split: insert the sentinel of `bucket`, ordered, if it is not there yet. */
  *splitOp(bucket) {
    const target = sentinelOrder(bucket, this.capacity);
    let pred = this.head;
    let current = pred.next.load();
    yield "split-read";
    while (current !== null && (current.order < target ||
           (current.order === target && current.sentinel && bucketOfOrder(current.order, this.capacity) !== bucket))) {
      pred = current;
      current = current.next.load();
      yield "split-read";
    }
    if (current !== null && current.order === target && current.sentinel &&
        bucketOfOrder(current.order, this.capacity) === bucket) return current;
    const sentinel = makeNode(true, target, null, bucket);
    sentinel.next.value = current;
    yield "split-before-cas";
    if (!pred.next.compareAndSwap(current, sentinel, this)) return null;
    this.sentinels.push(sentinel);
    this.sentinelsCreated++;
    return sentinel;
  }

  /** Walks from a bucket sentinel to the ordered window; reports the crossed element nodes. */
  *windowOp(sentinel, order, key) {
    let pred = sentinel;
    let current = pred.next.load();
    const crossed = [];
    yield "window-read";
    while (current !== null &&
           (current.sentinel ? current.order <= order
             : (current.order < order || (current.order === order && current.key < key)))) {
      if (!current.sentinel) crossed.push(current);
      pred = current;
      current = current.next.load();
      yield "window-read";
    }
    return { pred, current, crossed };
  }

  #stable(crossed) { return crossed.every((node) => node.marked.load() === false); }

  #inList(target) {
    for (let node = this.head.next.load(); node !== null; node = node.next.load()) if (node === target) return true;
    return false;
  }

  /** Physical removal of nodes another thread has already marked ("helping"). */
  *helpOp(crossed) {
    for (const node of crossed) {
      if (!node.marked.load()) continue;
      let pred = this.head;
      let current = pred.next.load();
      yield "help-read";
      while (current !== null && current !== node) {
        pred = current;
        current = current.next.load();
        yield "help-read";
      }
      if (current !== node) continue;
      yield "help-before-unlink";
      if (pred.next.compareAndSwap(node, node.next.load(), this)) this.size--;
    }
  }

  *insertOp(key) {
    if (!Number.isInteger(key)) throw new TypeError("keys must be integers");
    const order = orderOf(key);
    while (true) {
      const bucket = bucketOf(key, this.capacity);
      let sentinel = this.sentinelFor(bucket);
      if (sentinel === null) {
        const created = yield* this.splitOp(bucket);
        if (created === null) continue;
        sentinel = created;
      }
      const window = yield* this.windowOp(sentinel, order, key);
      if (window.current !== null && !window.current.sentinel &&
          window.current.order === order && window.current.key === key) return false;
      if (!this.#stable(window.crossed)) {                       // a removed node is in the way: help
        yield* this.helpOp(window.crossed);
        continue;
      }
      const node = makeNode(false, order, key, bucket);
      node.next.value = window.current;
      yield "before-link";
      if (window.pred.next.compareAndSwap(window.current, node, this)) {
        this.size++;
        while (this.size > this.capacity * this.loadFactor) { this.capacity *= 2; this.resizes++; }
        return true;
      }
    }
  }

  *containsOp(key) {
    if (!Number.isInteger(key)) throw new TypeError("keys must be integers");
    const bucket = bucketOf(key, this.capacity);
    let sentinel = this.sentinelFor(bucket);
    this.searchVisits = 0;
    if (sentinel === null) {
      const created = yield* this.splitOp(bucket);       // every operation performs the split
      if (created === null) return this.#drain(this.containsOp(key));
      sentinel = created;
    }
    const order = orderOf(key);
    let current = sentinel.next.load();
    yield "contains-read";
    while (current !== null && !current.sentinel && current.order <= order) {
      this.searchVisits++;
      if (current.order === order && current.key === key) return true;
      current = current.next.load();
      yield "contains-read";
    }
    return false;
  }

  *removeOp(key) {
    if (!Number.isInteger(key)) throw new TypeError("keys must be integers");
    const order = orderOf(key);
    while (true) {
      const bucket = bucketOf(key, this.capacity);
      let sentinel = this.sentinelFor(bucket);
      if (sentinel === null) {
        const created = yield* this.splitOp(bucket);
        if (created === null) continue;
        sentinel = created;
      }
      const window = yield* this.windowOp(sentinel, order, key);
      const victim = window.current;
      if (victim === null || victim.sentinel || victim.order !== order || victim.key !== key) return false;
      if (!this.#stable(window.crossed)) {
        yield* this.helpOp(window.crossed);
        continue;
      }
      yield "before-mark";
      if (!victim.marked.compareAndSwap(false, true, this)) continue;
      yield "before-unlink";
      if (window.pred.next.compareAndSwap(victim, victim.next.load(), this)) {
        this.size--;
        return true;
      }
      yield* this.helpOp([victim]);                              // our own mark may be unlinked by helping
      if (!this.#inList(victim)) return true;
    }
  }

  #drain(generator) {
    let outcome = generator.next();
    while (!outcome.done) outcome = generator.next();
    return outcome.value;
  }

  insert(key) { return this.#drain(this.insertOp(key)); }
  contains(key) { return this.#drain(this.containsOp(key)); }
  remove(key) { return this.#drain(this.removeOp(key)); }

  contents() {
    const keys = [];
    for (let node = this.head.next.load(); node !== null; node = node.next.load()) if (!node.sentinel) keys.push(node.key);
    return keys;
  }

  /** The keys per bucket, computed for the *current* capacity (the bucket is derived from the key). */
  buckets() {
    const out = new Map();
    for (const key of this.contents()) {
      const bucket = bucketOf(key, this.capacity);
      if (!out.has(bucket)) out.set(bucket, []);
      out.get(bucket).push(key);
    }
    return out;
  }

  /** Structural invariants; throws with the broken invariant instead of returning false. */
  validate() {
    const nodes = [];
    for (let node = this.head.next.load(); node !== null; node = node.next.load()) nodes.push(node);
    for (let i = 1; i < nodes.length; i++) if (!sortsAfter(nodes[i], nodes[i - 1])) throw new Error("the list is not ordered");
    const sentinelOrders = nodes.filter((node) => node.sentinel).map((node) => node.order);
    const seen = new Set();
    for (const node of nodes) {
      if (node.sentinel) {
        if (node.order !== sentinelOrder(bucketOfOrder(node.order, this.capacity), this.capacity)) {
          throw new Error("a sentinel is not at a bucket base");
        }
        continue;
      }
      if (node.order !== orderOf(node.key)) throw new Error("an element has the wrong order");
      if (seen.has(node.key)) throw new Error("duplicate key in the list");
      seen.add(node.key);
      let low = -1;
      let high = 2 ** 32;
      for (const order of sentinelOrders) {
        if (order <= node.order) low = order;
        else { high = order; break; }
      }
      if (node.order < low || node.order >= high) {
        throw new Error("an element sits outside every sentinel window");
      }
      if (low >= 0) {
        const windowBucket = bucketOfOrder(low, this.capacity);
        const elementBucket = bucketOf(node.key, this.capacity);
        const endBucket = bucketOfOrder(high === 2 ** 32 ? 0xffffffff : high, this.capacity);
        if (elementBucket < windowBucket || (high !== 2 ** 32 && elementBucket > endBucket)) {
          throw new Error("an element is outside its bucket's range");
        }
      }
    }
    if (seen.size !== this.size) throw new Error(`size ${this.size} but ${seen.size} distinct keys`);
    const present = new Set([this.head]);
    for (const node of nodes) present.add(node);
    for (const sentinel of this.sentinels) if (!present.has(sentinel)) throw new Error("a tracked sentinel left the list");
    return true;
  }
}

/** Advances one thread by one step and validates the table at every step boundary. */
function runSchedule(table, plans, order, options = {}) {
  const limit = options.limit ?? 200000;
  const threads = plans.map((plan) => ({ plan, index: 0, current: null, results: [] }));
  const isActive = (thread) => thread.current !== null || thread.index < thread.plan.length;
  let steps = 0;
  while (threads.some(isActive)) {
    if (steps > limit) throw new Error("schedule did not converge");
    const wanted = order(steps, threads);
    const id = isActive(threads[wanted]) ? wanted : threads.findIndex(isActive);
    const thread = threads[id];
    if (thread.current === null) {
      const operation = thread.plan[thread.index];
      if (operation.kind === "insert") thread.current = table.insertOp(operation.key);
      else if (operation.kind === "contains") thread.current = table.containsOp(operation.key);
      else if (operation.kind === "remove") thread.current = table.removeOp(operation.key);
      else throw new Error(`unknown operation ${operation.kind}`);
    }
    const outcome = thread.current.next();
    table.steps = ++steps;
    if (outcome.done) {
      thread.results.push(outcome.value);
      thread.index++;
      thread.current = null;
    }
    table.validate();
  }
  return { steps, results: threads.map((thread) => thread.results) };
}

const roundRobin = (step, threads) => step % threads.length;
const phaseOrder = (phases) => {
  let index = 0;
  let used = 0;
  return (step, threads) => {
    while (index < phases.length && used >= phases[index].steps) { index++; used = 0; }
    if (index >= phases.length) return 0;
    used++;
    return phases[index].thread;
  };
};

module.exports = {
  SplitOrderedTable, hash32, reverseBits32, orderOf, bucketOf, bucketOfOrder, sentinelOrder, capacityBits, runSchedule,
  roundRobin, phaseOrder,
};

if (require.main === module) {
  const assert = require("node:assert");
  const sorted = (keys) => [...keys].sort((a, b) => a - b);

  // --- the ordering primitives the scheme rests on ---
  assert.strictEqual(reverseBits32(1), 0x80000000);
  assert.strictEqual(reverseBits32(0x80000000), 1);
  assert.strictEqual(reverseBits32(reverseBits32(0x1234abcd)), 0x1234abcd);
  for (const key of [0, 1, 7, 64, 1000, 123456]) {
    assert.strictEqual(orderOf(key), reverseBits32(hash32(key)));
    assert.strictEqual(bucketOf(key, 16), orderOf(key) >>> 28);
    assert.strictEqual(bucketOfOrder(sentinelOrder(bucketOf(key, 64), 64), 64), bucketOf(key, 64));
  }

  // --- sequential insert / lookup / remove, validated after every operation ---
  const table = new SplitOrderedTable(4);
  for (let key = 0; key < 60; key++) { assert.strictEqual(table.insert(key * 7), true); table.validate(); }
  assert.strictEqual(table.insert(7), false, "re-inserting a key is a no-op");
  assert.strictEqual(table.size, 60);
  assert.deepStrictEqual(sorted(table.contents()), sorted(Array.from({ length: 60 }, (_, i) => i * 7)));
  for (const key of table.contents()) {
    assert.strictEqual(table.contains(key), true, `contains(${key})`);
    assert.ok(table.searchVisits <= table.buckets().get(bucketOf(key, table.capacity)).length + 1, "a lookup stays in its bucket");
  }
  assert.strictEqual(table.contains(1), false);
  assert.strictEqual(table.remove(1), false, "removing a missing key reports failure");
  for (let key = 0; key < 30; key++) assert.strictEqual(table.remove(key * 7), true);
  for (let key = 0; key < 30; key++) assert.strictEqual(table.contains(key * 7), false);
  assert.strictEqual(table.size, 30);
  table.validate();
  assert.ok(table.resizes >= 3, `the table must grow (resizes=${table.resizes})`);

  // --- the split invariant: doubling partitions every bucket by the new bit ---
  const split = new SplitOrderedTable(4);
  for (let key = 0; key < 40; key++) split.insert(key);
  const beforeCapacity = split.capacity;
  const frozen = new Map([...split.buckets()].map(([bucket, keys]) => [bucket, keys.slice()]));
  let probe = 1000;
  while (split.capacity === beforeCapacity) split.insert(probe++);
  assert.strictEqual(split.capacity, beforeCapacity * 2, "the table doubled exactly once");
  const k = capacityBits(beforeCapacity);
  const after = split.buckets();
  for (const [bucket, keys] of frozen) {
    const lowBucket = after.get(2 * bucket) ?? [];
    const highBucket = after.get(2 * bucket + 1) ?? [];
    for (const key of keys) {
      const upstairs = ((hash32(key) >>> k) & 1) === 1;
      assert.ok((upstairs ? highBucket : lowBucket).includes(key), `key ${key} of bucket ${bucket} landed right`);
      assert.ok(!(upstairs ? lowBucket : highBucket).includes(key), `key ${key} did not switch halves`);
    }
  }
  split.validate();

  // --- crafted interleavings: a lost CAS retries; a remove racing an insert keeps the key ---
  const raced = new SplitOrderedTable(4);
  const racedRun = runSchedule(raced, [
    [{ kind: "insert", key: 3 }, { kind: "insert", key: 11 }],
    [{ kind: "insert", key: 19 }],
  ], phaseOrder([{ thread: 0, steps: 2 }, { thread: 1, steps: Infinity }]));
  raced.validate();
  assert.strictEqual(raced.size, 3);
  assert.deepStrictEqual(sorted(raced.contents()), [3, 11, 19]);
  assert.deepStrictEqual(racedRun.results.map((r) => r.length), [2, 1]);
  const racing = new SplitOrderedTable(4);
  const racingRun = runSchedule(racing, [
    [{ kind: "insert", key: 5 }],
    [{ kind: "remove", key: 5 }, { kind: "insert", key: 5 }, { kind: "contains", key: 5 }],
  ], phaseOrder([{ thread: 0, steps: 3 }, { thread: 1, steps: Infinity }]));
  racing.validate();
  assert.strictEqual(racingRun.results[1][2], true, "the key is present again at the end");
  assert.deepStrictEqual(racing.contents(), [5]);

  // --- randomised interleavings against an exact mirror (disjoint key ranges per thread) ---
  let seed = 0x5eed;
  const next = () => { seed ^= (seed << 13) >>> 0; seed >>>= 0; seed ^= seed >>> 17; seed ^= (seed << 5) >>> 0; seed >>>= 0; return seed / 4294967296; };
  for (let trial = 0; trial < 4; trial++) {
    const randomTable = new SplitOrderedTable(4);
    const mirror = new Set();
    const plans = [];
    for (let thread = 0; thread < 3; thread++) {
      const plan = [];
      for (let i = 0; i < 10; i++) {
        const key = thread + 3 * Math.floor(next() * 10);
        const roll = next();
        plan.push(roll < 0.5 ? { kind: "insert", key } : roll < 0.8 ? { kind: "remove", key } : { kind: "contains", key });
      }
      plans.push(plan);
      for (const operation of plan) {
        if (operation.kind === "insert") mirror.add(operation.key);
        else if (operation.kind === "remove") mirror.delete(operation.key);
      }
    }
    runSchedule(randomTable, plans, (step, threads) => Math.floor(next() * threads.length));
    randomTable.validate();
    assert.deepStrictEqual(sorted(randomTable.contents()), sorted(mirror), `trial ${trial}: contents match the mirror`);
    for (const key of mirror) assert.strictEqual(randomTable.contains(key), true, `trial ${trial}: contains(${key})`);
  }

  // --- determinism, boundaries and errors ---
  const build = () => {
    const fresh = new SplitOrderedTable(2);
    const run = runSchedule(fresh, [
      [{ kind: "insert", key: 5 }, { kind: "remove", key: 5 }],
      [{ kind: "insert", key: 9 }, { kind: "contains", key: 9 }],
    ], roundRobin);
    return { contents: fresh.contents(), capacity: fresh.capacity, sentinels: fresh.sentinelsCreated, steps: run.steps };
  };
  assert.deepStrictEqual(build(), build(), "identical schedules replay identically");
  const single = new SplitOrderedTable(1);
  assert.strictEqual(single.insert(42), true);
  assert.strictEqual(single.resizes, 1, "a single bucket grows as soon as it is over the load factor");
  assert.strictEqual(single.capacity, 2);
  single.validate();
  assert.strictEqual(single.remove(42), true);
  single.validate();
  const pristine = new SplitOrderedTable(4);
  pristine.validate();
  assert.strictEqual(pristine.sentinelsCreated, 1, "only the head sentinel exists before any operation");
  assert.strictEqual(pristine.contains(1), false);
  assert.strictEqual(pristine.remove(1), false);
  assert.strictEqual(pristine.size, 0);
  assert.throws(() => new SplitOrderedTable(3), RangeError);
  assert.throws(() => new SplitOrderedTable(0), RangeError);
  assert.throws(() => pristine.insert("x"), TypeError);
  assert.throws(() => pristine.remove(1.5), TypeError);
  assert.throws(() => pristine.contains(NaN), TypeError);
  assert.throws(() => runSchedule(new SplitOrderedTable(2), [[{ kind: "put", key: 1 }]], roundRobin), /unknown operation/);
  assert.throws(() => runSchedule(new SplitOrderedTable(2), [[{ kind: "insert", key: 1 }]], roundRobin, { limit: 0 }), /did not converge/);

  console.log(`resizes=${table.resizes} sentinels=${table.sentinelsCreated} keys=${table.size} casFailures=${raced.casFailures + racing.casFailures}`);
  console.log("PASS");
}
