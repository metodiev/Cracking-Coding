/** Task 194 - Implement caching of expensive function results with TTL. (Tier: Beginner, CTCI Ch 7) */
"use strict";

class TTLCache {
  constructor() {
    this.entries = new Map();        // key -> { value, expiresAtMs }
    this.hits = 0;
    this.misses = 0;
    this.evictions = 0;
  }

  put(key, value, nowMs, ttlMs) {
    if (ttlMs <= 0) throw new Error("ttl_ms must be positive");
    this.entries.set(key, { value, expiresAtMs: nowMs + ttlMs });
  }

  get(key, nowMs) {
    const entry = this.entries.get(key);
    if (entry === undefined) {
      this.misses++;
      return undefined;
    }
    if (nowMs >= entry.expiresAtMs) {           // exactly at TTL -> expired
      this.entries.delete(key);
      this.evictions++;
      this.misses++;
      return undefined;
    }
    this.hits++;
    return entry.value;
  }

  memoize(fn, key, nowMs, ttlMs) {
    const cached = this.get(key, nowMs);
    if (cached !== undefined) return cached;
    const value = fn(key);
    this.put(key, value, nowMs, ttlMs);
    return value;
  }

  size() {
    return this.entries.size;
  }
}

module.exports = { TTLCache };

if (require.main === module) {
  const assert = require("node:assert");

  const cache = new TTLCache();
  cache.put("a", 10, 1000, 100);                // valid while now < 1100
  assert.strictEqual(cache.get("a", 1099), 10);
  assert.strictEqual(cache.get("a", 1100), undefined);   // exactly at TTL -> expired
  assert.deepStrictEqual([cache.hits, cache.misses, cache.evictions], [1, 1, 1]);
  assert.strictEqual(cache.size(), 0);                   // expired entry dropped lazily

  cache.put("b", 20, 0, 50);
  assert.strictEqual(cache.get("b", 25), 20);
  cache.put("b", 21, 30, 50);                   // refresh -> expires at 80
  assert.strictEqual(cache.get("b", 79), 21);
  assert.strictEqual(cache.get("b", 80), undefined);

  const missing = new TTLCache();
  assert.strictEqual(missing.get("nope", 0), undefined);
  assert.deepStrictEqual([missing.hits, missing.misses], [0, 1]);
  assert.throws(() => missing.put("x", 1, 0, 0), /ttl_ms must be positive/);

  const calls = [];
  const slow = (n) => { calls.push(n); return n * n; };
  const memo = new TTLCache();
  assert.strictEqual(memo.memoize(slow, 7, 0, 100), 49);    // computes (miss)
  assert.strictEqual(memo.memoize(slow, 7, 99, 100), 49);   // cache hit
  assert.deepStrictEqual(calls, [7]);
  assert.strictEqual(memo.memoize(slow, 7, 100, 100), 49);  // expired -> recompute
  assert.deepStrictEqual(calls, [7, 7]);
  assert.deepStrictEqual([memo.hits, memo.misses, memo.evictions], [1, 2, 1]);
  console.log("PASS");
}
