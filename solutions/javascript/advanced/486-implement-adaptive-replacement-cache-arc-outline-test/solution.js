/** Task 486 - Implement adaptive replacement cache (ARC) outline & test. (Tier: Advanced, CTCI Ch 14)
 *
 * ARC outline (Megiddo & Modha 2003), capacity c:
 * - Real lists: T1 (keys seen once recently), T2 (keys seen at least twice). Cache contents = T1+T2,
 *   with |T1| + |T2| <= c.
 * - Ghost lists: B1 (recently evicted from T1), B2 (evicted from T2); |T1| + |B1| <= c and
 *   |T1|+|T2|+|B1|+|B2| <= 2c.
 * - A request found in T1 is promoted to T2; in T2 it is refreshed (MRU). A request found in a
 *   ghost list adapts the target size p for T1: p grows on a B1 hit, shrinks on a B2 hit, always
 *   clamped to [0, c]; the hit key is admitted to T2. A brand-new request is admitted to T1, first
 *   discarding a ghost or demoting the LRU of T1/T2 via `replace` (T1 when |T1| > p, else T2).
 *
 * Adaptivity test:
 * - Trace S ("scan-resistance", 20 rounds of [hot 0..3, 8-key scan, hot 0..3], scan keys cycling
 *   mod 8): LRU thrashes on the scans; pinned ARC 133 vs LRU 76 hits.
 * - Trace A ("adaptivity", same rounds but 4 rotating scan keys): ghost hits drive p up to c = 4
 *   and back down; pinned p_max = 4, 4 B1-hits, 141 B2-hits, ARC 79 vs LRU 76.
 *
 * Correctness: on every access the requested key is in the cache, the four lists are pairwise
 * disjoint with the documented size bounds honoured, and per-step cache contents/hits match an
 * independent list-based ARC reference implementation.
 */
"use strict";

class ARCCache {
  constructor(capacity) {
    this.c = capacity;
    this.p = 0;
    this.t1 = new Map();
    this.t2 = new Map();
    this.b1 = new Map();
    this.b2 = new Map();
    this.hits = 0;
    this.pMax = 0;
    this.pMin = 0;
    this.b1Hits = 0;
    this.b2Hits = 0;
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
      this.hits += 1;
      return true;
    }
    if (this.t2.has(key)) {
      this.t2.delete(key);
      this.t2.set(key, true);
      this.hits += 1;
      return true;
    }
    if (this.b1.has(key)) {
      this.p = Math.min(this.c, this.p + Math.max(1, Math.floor(this.b2.size / this.b1.size)));
      this.b1Hits += 1;
      this.replace(key);
      this.b1.delete(key);
      this.t2.set(key, true);
    } else if (this.b2.has(key)) {
      this.p = Math.max(0, this.p - Math.max(1, Math.floor(this.b1.size / this.b2.size)));
      this.b2Hits += 1;
      this.replace(key);
      this.b2.delete(key);
      this.t2.set(key, true);
    } else {
      if (this.t1.size + this.b1.size === this.c) {
        if (this.t1.size < this.c) {
          this.b1.delete(this.b1.keys().next().value);
          this.replace(key);
        } else {
          this.t1.delete(this.t1.keys().next().value);
        }
      } else if (this.t1.size + this.b1.size < this.c) {
        const total = this.t1.size + this.t2.size + this.b1.size + this.b2.size;
        if (total >= this.c) {
          if (total === 2 * this.c) this.b2.delete(this.b2.keys().next().value);
          this.replace(key);
        }
      }
      this.t1.set(key, true);
    }
    this.pMax = Math.max(this.pMax, this.p);
    this.pMin = Math.min(this.pMin, this.p);
    return false;
  }

  contents() {
    return Array.from(this.t1.keys()).concat(Array.from(this.t2.keys())).sort((a, b) => a - b);
  }

  checkInvariants(key, step) {
    if (!this.t1.has(key) && !this.t2.has(key)) throw new Error(`FAILED: key absent ${step}`);
    if (this.t1.size + this.t2.size > this.c) throw new Error(`FAILED: cache bound ${step}`);
    if (this.t1.size + this.b1.size > this.c) throw new Error(`FAILED: t1+b1 bound ${step}`);
    if (this.t1.size + this.t2.size + this.b1.size + this.b2.size > 2 * this.c) {
      throw new Error(`FAILED: total bound ${step}`);
    }
    if (this.p < 0 || this.p > this.c) throw new Error(`FAILED: p range ${step}`);
    const lists = [this.t1, this.t2, this.b1, this.b2];
    for (let i = 0; i < 4; i++) {
      for (let j = i + 1; j < 4; j++) {
        for (const key2 of lists[i].keys()) {
          if (lists[j].has(key2)) throw new Error(`FAILED: lists overlap ${step}`);
        }
      }
    }
  }
}

function arcReference(trace, capacity) {
  let p = 0;
  let hits = 0;
  const t1 = [];
  const t2 = [];
  const b1 = [];
  const b2 = [];
  const contents = [];
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
    contents.push(t1.concat(t2).sort((a, b) => a - b));
  }
  return { hits, contents };
}

function lruHits(trace, capacity) {
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

function scanTrace(rounds, hot, scanWidth, cold) {
  const trace = [];
  for (let r = 0; r < rounds; r++) {
    for (let i = 0; i < hot; i++) trace.push(i);
    for (let j = 0; j < scanWidth; j++) trace.push(100 + ((r * scanWidth + j) % cold));
    for (let i = 0; i < hot; i++) trace.push(i);
  }
  return trace;
}

function verifyTrace(trace, capacity, expectedHits, expectedLru, label) {
  const cache = new ARCCache(capacity);
  const contentsVector = [];
  let summedHits = 0;
  for (let step = 0; step < trace.length; step++) {
    if (cache.access(trace[step])) summedHits += 1;
    cache.checkInvariants(trace[step], step);
    contentsVector.push(cache.contents());
  }
  const reference = arcReference(trace, capacity);
  if (cache.hits !== reference.hits || cache.hits !== expectedHits) {
    throw new Error(`FAILED: ${label} hits ${cache.hits} ref ${reference.hits}`);
  }
  if (summedHits !== cache.hits) throw new Error(`FAILED: ${label} hit vector`);
  if (JSON.stringify(contentsVector) !== JSON.stringify(reference.contents)) {
    throw new Error(`FAILED: ${label} contents vector`);
  }
  const lru = lruHits(trace, capacity);
  if (lru !== expectedLru) throw new Error(`FAILED: ${label} lru ${lru}`);
  if (cache.hits < lru) throw new Error(`FAILED: ${label} ARC < LRU`);
  return cache;
}

module.exports = { ARCCache, arcReference };

if (require.main === module) {
  const assert = require("node:assert");

  const cache = new ARCCache(2);
  const observed = [1, 2, 1, 3, 1].map((key) => cache.access(key));
  assert.deepStrictEqual(observed, [false, false, true, false, true]);
  assert.strictEqual(cache.hits, 2);
  assert.deepStrictEqual(cache.contents(), [1, 3]);

  const scan = scanTrace(20, 4, 8, 8);
  assert.strictEqual(scan.length, 320);
  assert.deepStrictEqual(scan.slice(0, 4), [0, 1, 2, 3]);
  const scanCache = verifyTrace(scan, 4, 133, 76, "scan-resistance");
  assert.deepStrictEqual(scanCache.contents(), [0, 1, 2, 3]);
  assert.ok(scanCache.hits > lruHits(scan, 4));

  const adaptive = scanTrace(20, 4, 4, 4);
  assert.strictEqual(adaptive.length, 240);
  const adaptiveCache = verifyTrace(adaptive, 4, 79, 76, "adaptivity");
  assert.strictEqual(adaptiveCache.pMax, 4);
  assert.strictEqual(adaptiveCache.pMin, 0);
  assert.strictEqual(adaptiveCache.p, 0);
  assert.strictEqual(adaptiveCache.b1Hits, 4);
  assert.strictEqual(adaptiveCache.b2Hits, 141);
  assert.deepStrictEqual(adaptiveCache.contents(), [0, 1, 2, 3]);
  console.log("PASS");
}
