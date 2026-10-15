/** Task 159 - Implement function memoizer decorator (or equivalent). (Tier: Beginner, CTCI Ch 8) */
"use strict";

/**
 * Memoizer for single-argument functions, distinct from the task 128 wrapper:
 * records hits / misses, exposes the cache size, and supports an optional
 * capacity with FIFO eviction - when full, the OLDEST inserted key leaves first
 * and a hit never changes insertion order (FIFO, deliberately not LRU; see
 * task 160 for an LRU cache).
 */
class Memoizer {
  constructor(fn, capacity = null) {
    if (capacity !== null && capacity <= 0) throw new RangeError("capacity must be >= 1");
    this.fn = fn;
    this.capacity = capacity;
    this.cache = new Map();
    this.order = [];          // FIFO insertion order of keys
    this.hits = 0;
    this.misses = 0;
  }

  call(n) {
    if (this.cache.has(n)) {
      this.hits++;
      return this.cache.get(n);
    }
    this.misses++;
    const value = this.fn(n);
    if (this.capacity !== null && this.cache.size >= this.capacity) {
      const evicted = this.order.shift();        // FIFO: oldest insertion goes first
      this.cache.delete(evicted);
    }
    this.cache.set(n, value);
    this.order.push(n);
    return value;
  }

  get size() {
    return this.cache.size;
  }

  /** True iff n is currently cached (used to assert eviction). */
  cached(n) {
    return this.cache.has(n);
  }
}

/** Steps for n to reach 1; collatzSteps(27) === 111. */
function collatzSteps(n) {
  if (n < 1) throw new RangeError("n must be >= 1");
  let steps = 0;
  while (n !== 1) {
    n = n % 2 === 0 ? n / 2 : 3 * n + 1;
    steps++;
  }
  return steps;
}

module.exports = { Memoizer, collatzSteps };

if (require.main === module) {
  const assert = require("node:assert");

  // 1) no capacity: everything stays cached
  const memo = new Memoizer(collatzSteps);
  assert.strictEqual(memo.call(27), 111);
  assert.strictEqual(memo.call(6), 8);
  assert.strictEqual(memo.call(7), 16);
  assert.deepStrictEqual([memo.hits, memo.misses, memo.size], [0, 3, 3]);
  assert.strictEqual(memo.call(27), 111);
  assert.strictEqual(memo.call(6), 8);
  assert.deepStrictEqual([memo.hits, memo.misses], [2, 3]);

  // 2) capacity 3 with FIFO eviction
  const small = new Memoizer(collatzSteps, 3);
  for (const n of [27, 6, 7]) small.call(n);
  assert.strictEqual(small.size, 3);
  assert.deepStrictEqual([small.hits, small.misses], [0, 3]);
  assert.strictEqual(small.call(27), 111);                 // hit keeps FIFO order
  assert.strictEqual(small.cached(27), true);
  assert.strictEqual(small.call(9), 19);                   // evicts the oldest (27)
  assert.strictEqual(small.size, 3);
  assert.strictEqual(small.cached(27), false);
  assert.strictEqual(small.cached(9), true);
  assert.strictEqual(small.call(6), 8);                    // still cached
  assert.deepStrictEqual([small.hits, small.misses], [2, 4]);

  // 3) capacity 1 behaves like a one-slot cache
  const tiny = new Memoizer((n) => n * n, 1);
  assert.strictEqual(tiny.call(3), 9);
  assert.strictEqual(tiny.call(4), 16);
  assert.strictEqual(tiny.size, 1);
  assert.strictEqual(tiny.cached(3), false);
  assert.strictEqual(tiny.cached(4), true);
  assert.strictEqual(tiny.call(4), 16);
  assert.deepStrictEqual([tiny.hits, tiny.misses], [1, 2]);

  // capacity must be positive
  assert.throws(() => new Memoizer((n) => n, 0), RangeError);

  // stats are per-instance, not shared
  const other = new Memoizer((n) => n + 1);
  assert.strictEqual(other.call(1), 2);
  assert.deepStrictEqual([other.hits, other.misses], [0, 1]);
  assert.deepStrictEqual([memo.hits, memo.misses], [2, 3]);
  console.log("PASS");
}
