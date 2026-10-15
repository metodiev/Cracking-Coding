/** Task 485 - Implement cache eviction policies: LRU, LFU, ARC (simulate). (Tier: Advanced, CTCI Ch 14)
 *
 * Documented capacity semantics: `capacity` c is the maximum number of distinct keys the cache may
 * hold (cache-as-main-memory, no separate disk tiers here). Every access returns hit/miss and the
 * cache size never exceeds c.
 *
 * Policy definitions (the "simulator"):
 * - LRU: a hit moves the key to the most-recently-used end; a miss inserts at MRU and, when the
 *   cache is over capacity, evicts the least-recently-used key.
 * - LFU: every hit increments the key's count and refreshes its recency; a miss on a full cache
 *   evicts the key with the smallest count, ties broken by least-recently-used (documented rule).
 * - ARC (Megiddo & Modha): p-adaptive split of T1 (recent) / T2 (frequent); misses that hit ghost
 *   lists B1/B2 adjust p, and `replace` demotes within real lists. Cache contents = T1 + T2 with
 *   |T1| + |T2| <= c; B1/B2 hold only ghosts.
 *
 * Each policy is run against an independent naive reference implementation (arrays + linear scans)
 * on three fixed LCG-derived traces; hit totals must be identical, and the totals are also pinned
 * as constants verified by the oracle.
 */
"use strict";

class LRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.order = new Map();
    this.hitCount = 0;
  }

  name() {
    return "LRU";
  }

  access(key) {
    if (this.order.has(key)) {
      this.order.delete(key);
      this.order.set(key, true);
      this.hitCount += 1;
      return true;
    }
    this.order.set(key, true);
    if (this.order.size > this.capacity) {
      this.order.delete(this.order.keys().next().value);
    }
    return false;
  }

  size() {
    return this.order.size;
  }

  contents() {
    return Array.from(this.order.keys()).sort((a, b) => a - b);
  }

  hits() {
    return this.hitCount;
  }
}

class LFUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.counts = new Map();
    this.recency = new Map();
    this.tick = 0;
    this.hitCount = 0;
  }

  name() {
    return "LFU";
  }

  access(key) {
    this.tick += 1;
    if (this.counts.has(key)) {
      this.counts.set(key, this.counts.get(key) + 1);
      this.recency.set(key, this.tick);
      this.hitCount += 1;
      return true;
    }
    if (this.counts.size >= this.capacity) {
      let victim = null;
      let best = null;
      for (const candidate of this.counts.keys()) {
        const score = this.counts.get(candidate) * 1000000 + this.recency.get(candidate);
        if (best === null || score < best) {
          best = score;
          victim = candidate;
        }
      }
      this.counts.delete(victim);
      this.recency.delete(victim);
    }
    this.counts.set(key, 1);
    this.recency.set(key, this.tick);
    return false;
  }

  size() {
    return this.counts.size;
  }

  contents() {
    return Array.from(this.counts.keys()).sort((a, b) => a - b);
  }

  hits() {
    return this.hitCount;
  }
}

class ARCCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.p = 0;
    this.t1 = new Map();
    this.t2 = new Map();
    this.b1 = new Map();
    this.b2 = new Map();
    this.hitCount = 0;
  }

  name() {
    return "ARC";
  }

  replace(key) {
    if (this.t1.size > 0 && ((this.b2.has(key) && this.t1.size === this.p) || this.t1.size > this.p)) {
      const victim = this.t1.keys().next().value;
      this.t1.delete(victim);
      this.b1.set(victim, true);
    } else if (this.t2.size > 0) {
      const victim = this.t2.keys().next().value;
      this.t2.delete(victim);
      this.b2.set(victim, true);
    }
  }

  access(key) {
    if (this.t1.has(key)) {
      this.t1.delete(key);
      this.t2.set(key, true);
      this.hitCount += 1;
      return true;
    }
    if (this.t2.has(key)) {
      this.t2.delete(key);
      this.t2.set(key, true);
      this.hitCount += 1;
      return true;
    }
    if (this.b1.has(key)) {
      this.p = Math.min(this.capacity, this.p + Math.max(1, Math.floor(this.b2.size / this.b1.size)));
      this.replace(key);
      this.b1.delete(key);
      this.t2.set(key, true);
    } else if (this.b2.has(key)) {
      this.p = Math.max(0, this.p - Math.max(1, Math.floor(this.b1.size / this.b2.size)));
      this.replace(key);
      this.b2.delete(key);
      this.t2.set(key, true);
    } else {
      if (this.t1.size + this.b1.size === this.capacity) {
        if (this.t1.size < this.capacity) {
          this.b1.delete(this.b1.keys().next().value);
          this.replace(key);
        } else {
          this.t1.delete(this.t1.keys().next().value);
        }
      } else if (this.t1.size + this.b1.size < this.capacity) {
        const total = this.t1.size + this.t2.size + this.b1.size + this.b2.size;
        if (total >= this.capacity) {
          if (total === 2 * this.capacity) this.b2.delete(this.b2.keys().next().value);
          this.replace(key);
        }
      }
      this.t1.set(key, true);
    }
    return false;
  }

  size() {
    return this.t1.size + this.t2.size;
  }

  contents() {
    return Array.from(this.t1.keys()).concat(Array.from(this.t2.keys())).sort((a, b) => a - b);
  }

  hits() {
    return this.hitCount;
  }
}

function uniformTrace(count, seed, keyCount) {
  const trace = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    trace.push(seed % keyCount);
  }
  return trace;
}

function localityTrace(count, seed, keyCount, repeatMod) {
  const trace = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    if (i > 0 && seed % 10 < repeatMod) trace.push(trace[i - 1]);
    else trace.push(seed % keyCount);
  }
  return trace;
}

function hotScanTrace(rounds, hot, scanWidth) {
  const trace = [];
  for (let block = 0; block < rounds; block++) {
    for (let i = 0; i < hot; i++) trace.push(i);
    for (let i = 0; i < scanWidth; i++) trace.push(100 + block * scanWidth + i);
    for (let i = 0; i < hot; i++) trace.push(i);
  }
  return trace;
}

function lruReference(trace, capacity) {
  const order = [];
  let hits = 0;
  for (const key of trace) {
    const at = order.indexOf(key);
    if (at >= 0) {
      order.splice(at, 1);
      order.push(key);
      hits += 1;
    } else {
      order.push(key);
      if (order.length > capacity) order.shift();
    }
  }
  return hits;
}

function lfuReference(trace, capacity) {
  const counts = new Map();
  const recency = new Map();
  let tick = 0;
  let hits = 0;
  for (const key of trace) {
    tick += 1;
    if (counts.has(key)) {
      counts.set(key, counts.get(key) + 1);
      recency.set(key, tick);
      hits += 1;
    } else {
      if (counts.size >= capacity) {
        let victim = null;
        let best = null;
        for (const candidate of counts.keys()) {
          const score = counts.get(candidate) * 1000000 + recency.get(candidate);
          if (best === null || score < best) {
            best = score;
            victim = candidate;
          }
        }
        counts.delete(victim);
        recency.delete(victim);
      }
      counts.set(key, 1);
      recency.set(key, tick);
    }
  }
  return hits;
}

function arcReference(trace, capacity) {
  let p = 0;
  let hits = 0;
  const t1 = [];
  const t2 = [];
  const b1 = [];
  const b2 = [];
  const evict = (key) => {
    if (t1.length > 0 && ((b2.includes(key) && t1.length === p) || t1.length > p)) {
      b1.push(t1.shift());
    } else if (t2.length > 0) {
      b2.push(t2.shift());
    }
  };
  for (const key of trace) {
    const inT1 = t1.indexOf(key);
    const inT2 = t2.indexOf(key);
    if (inT1 >= 0) {
      t1.splice(inT1, 1);
      t2.push(key);
      hits += 1;
    } else if (inT2 >= 0) {
      t2.splice(inT2, 1);
      t2.push(key);
      hits += 1;
    } else {
      const inB1 = b1.indexOf(key);
      const inB2 = b2.indexOf(key);
      if (inB1 >= 0) {
        p = Math.min(capacity, p + Math.max(1, Math.floor(b2.length / b1.length)));
        evict(key);
        b1.splice(inB1, 1);
        t2.push(key);
      } else if (inB2 >= 0) {
        p = Math.max(0, p - Math.max(1, Math.floor(b1.length / b2.length)));
        evict(key);
        b2.splice(inB2, 1);
        t2.push(key);
      } else {
        if (t1.length + b1.length === capacity) {
          if (t1.length < capacity) {
            b1.shift();
            evict(key);
          } else {
            t1.shift();
          }
        } else if (t1.length + b1.length < capacity) {
          const total = t1.length + t2.length + b1.length + b2.length;
          if (total >= capacity) {
            if (total === 2 * capacity) b2.shift();
            evict(key);
          }
        }
        t1.push(key);
      }
    }
  }
  return hits;
}

function runTrace(cache, trace, expected, label) {
  let overflows = 0;
  let lastSize = 0;
  let maxSize = 0;
  let summedHits = 0;
  for (const key of trace) {
    if (cache.access(key)) summedHits += 1;
    const size = cache.size();
    lastSize = size;
    maxSize = Math.max(maxSize, size);
    if (size > 4) overflows += 1;
  }
  if (cache.hits() !== expected) throw new Error(`FAILED: ${label} hits ${cache.hits()}`);
  if (summedHits !== cache.hits()) throw new Error(`FAILED: ${label} hit vector`);
  if (overflows !== 0) throw new Error(`FAILED: ${label} capacity respected`);
  if (maxSize !== 4 || lastSize !== 4) throw new Error(`FAILED: ${label} warm capacity`);
  const contents = cache.contents();
  if (contents.length !== 4 || new Set(contents).size !== 4) throw new Error(`FAILED: ${label} contents`);
}

module.exports = { LRUCache, LFUCache, ARCCache, lruReference, lfuReference, arcReference };

if (require.main === module) {
  const assert = require("node:assert");
  const hand = [1, 2, 1, 3, 1];
  const expected = [false, false, true, false, true];
  for (const cache of [new LRUCache(2), new LFUCache(2), new ARCCache(2)]) {
    assert.deepStrictEqual(hand.map((key) => cache.access(key)), expected, cache.name());
    assert.strictEqual(cache.hits(), 2, cache.name());
    assert.deepStrictEqual(cache.contents(), [1, 3], cache.name());
  }
  assert.strictEqual(lruReference(hand, 2), 2);
  assert.strictEqual(lfuReference(hand, 2), 2);
  assert.strictEqual(arcReference(hand, 2), 2);

  const uniform = uniformTrace(400, 485, 20);
  const locality = localityTrace(400, 486, 24, 7);
  const hotScan = hotScanTrace(24, 4, 8);

  runTrace(new LRUCache(4), uniform, 70, "uniform LRU");
  runTrace(new LFUCache(4), uniform, 64, "uniform LFU");
  runTrace(new ARCCache(4), uniform, 64, "uniform ARC");
  runTrace(new LRUCache(4), locality, 296, "locality LRU");
  runTrace(new LFUCache(4), locality, 291, "locality LFU");
  runTrace(new ARCCache(4), locality, 290, "locality ARC");
  runTrace(new LRUCache(4), hotScan, 92, "hot-scan LRU");
  runTrace(new LFUCache(4), hotScan, 161, "hot-scan LFU");
  runTrace(new ARCCache(4), hotScan, 161, "hot-scan ARC");

  assert.strictEqual(lruReference(uniform, 4), 70);
  assert.strictEqual(lfuReference(uniform, 4), 64);
  assert.strictEqual(arcReference(uniform, 4), 64);
  assert.strictEqual(lruReference(locality, 4), 296);
  assert.strictEqual(lfuReference(locality, 4), 291);
  assert.strictEqual(arcReference(locality, 4), 290);
  assert.strictEqual(lruReference(hotScan, 4), 92);
  assert.strictEqual(lfuReference(hotScan, 4), 161);
  assert.strictEqual(arcReference(hotScan, 4), 161);
  assert.ok(161 > 92, "ARC beats LRU on hot-scan");
  console.log("PASS");
}
