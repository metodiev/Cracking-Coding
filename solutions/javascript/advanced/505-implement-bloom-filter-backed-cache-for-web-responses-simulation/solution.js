/**
 * Task 505 - Implement bloom-filter-backed cache for web responses simulation. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * A deterministic web-response cache simulation with a Bloom filter in front of the origin store.
 * The filter uses enhanced double hashing - k probes at fmix32(h1(key) + i * h2(key)) mod m, where
 * h1 is a 32-bit FNV-1a mix, h2 a different multiplicative mix and fmix32 the Murmur3 finalizer -
 * all pure integer arithmetic, so results are identical on every run and every platform. Sizing
 * follows the textbook formulas: m = ceil(-n * ln(p) / ln(2)^2) bits for n keys and a target
 * false-positive rate p, and k = max(1, round((m / n) * ln 2)) hash functions. The cache itself is
 * an LRU response map for 64 bodies plus a bounded exact set of URLs already confirmed absent; the
 * Bloom filter is the small front filter for that set, so a "definitely absent" answer skips the
 * exact-set probe entirely and a "maybe" answer falls through to the exact set and, on a false
 * positive, to the origin - never to a wrong answer. The simulated origin answers from a
 * ground-truth URL table where only /hot/ and /cold/ URLs exist and /dead/ and /miss/ are real 404s.
 *
 * Documented checks: the filter has *no* false negatives (every inserted key is reported present,
 * verified key by key, and again over every 404 URL of the live trace) and adding a key twice
 * leaves the bit vector unchanged (idempotence, checked byte-for-byte). The empirical
 * false-positive rate over 20000 never-inserted LCG keys must sit within 0.005 of the theory
 * (1 - e^(-kn/m))^k = 0.0100 for the documented n = 1000, m = 9586, k = 7 configuration (the
 * finalizer is what makes the measured 0.0103 line up), the measured rate must grow with load (1x
 * vs 4x insertions), an over-full filter must report > 0.5, an empty filter must reject every probe
 * and an all-ones bit vector must accept everything. The simulation feeds one deterministic
 * 4000-request trace (hot, cold, a pool of 40 repeated dead links and fresh 404s) to a plain cache
 * and to the Bloom-backed one: both must return *exactly* the same (status, body) pairs, and every
 * status must match the ground truth, so the 404s are real 404s. The Bloom-backed run performs
 * 1379 origin lookups against the plain cache's 2196 (documented >= 1.3x; measured 1.59x), 817 of
 * them replaced by negative-cache hits, with 0 false positives in a properly sized configuration.
 * Degradations are measured explicitly and never change an answer: shrinking the negative set to 64
 * entries evicts 296 dead links, which come back as 176 false-positive probes costing *exactly* one
 * extra origin lookup each (1379 + 176 = 1555, still 1.41x better than no negative caching), while
 * a deliberately tiny 64-bit 4-hash filter is useless as a front filter (1251 false positives, the
 * exact set consulted 2068 times instead of 817) but leaves the origin-lookup count untouched,
 * because a false positive on an unknown URL lands on exactly the lookup the filter would have
 * made. Re-running the whole simulation reproduces identical responses and statistics.
 */
"use strict";

const DEFAULT_N = 1000;
const DEFAULT_P = 0.01;

function fnv1a(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function secondHash(text) {
  let hash = 0x9e3779b1;
  for (let i = 0; i < text.length; i++) {
    hash = (Math.imul(hash ^ text.charCodeAt(i), 0x85ebca6b) + i) >>> 0;
  }
  return (hash | 1) >>> 0;                              // odd, so the probe sequence is not trivial
}

/** Murmur3-style 32-bit finalizer: mixes the whole double-hashing word before the modulo, which
 *  is what makes the observed false-positive rate match the (1 - e^(-kn/m))^k prediction. */
function fmix32(value) {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return x >>> 0;
}

function optimalBits(n, p) {
  return Math.max(1, Math.ceil(-n * Math.log(p) / (Math.LN2 * Math.LN2)));
}

function optimalHashes(m, n) {
  return Math.max(1, Math.round((m / n) * Math.LN2));
}

class BloomFilter {
  constructor(bits, hashes) {
    this.m = bits;
    this.k = hashes;
    this.words = new Uint8Array(bits);
    this.insertions = 0;
  }

  probes(key) {
    const h1 = fnv1a(key);
    const h2 = secondHash(key);
    const out = new Array(this.k);
    for (let i = 0; i < this.k; i++) {
      out[i] = fmix32((h1 + Math.imul(i, h2)) >>> 0) % this.m;
    }
    return out;
  }

  add(key) {
    for (const index of this.probes(key)) this.words[index] = 1;
    this.insertions += 1;
    return this;
  }

  maybeContains(key) {
    for (const index of this.probes(key)) {
      if (this.words[index] === 0) return false;
    }
    return true;
  }

  /** Number of set bits: the load factor that drives the false-positive rate. */
  bitCount() {
    let total = 0;
    for (const value of this.words) total += value;
    return total;
  }

  theoreticalFp() {
    const fraction = 1.0 - Math.exp(-this.k * this.insertions / this.m);
    return fraction ** this.k;
  }

  static forTarget(n = DEFAULT_N, p = DEFAULT_P) {
    const m = optimalBits(n, p);
    return new BloomFilter(m, optimalHashes(m, n));
  }
}

/** Deterministic request trace: hot URLs, cold URLs, a pool of repeated dead links and new 404s. */
function requestTrace(count, seed) {
  const trace = [];
  let s = seed;
  for (let i = 0; i < count; i++) {
    s = (s * 75 + 74) % 65537;
    const roll = s % 100;
    s = (s * 75 + 74) % 65537;
    if (roll < 45) {
      trace.push(`/hot/${s % 20}`);                   // frequently requested, always in origin
    } else if (roll < 65) {
      trace.push(`/cold/${s % 400}`);                 // in origin, requested rarely
    } else if (roll < 85) {
      trace.push(`/dead/${s % 40}`);                  // a pool of 40 dead links, hit again and again
    } else if (roll < 95) {
      trace.push(`/cold/${(s % 400) + 400}`);         // exists in origin, no cache history
    } else {
      trace.push(`/miss/${s % 300}`);                 // never exists: a fresh 404 each time
    }
  }
  return trace;
}

/** Ground truth origin: /hot/ and /cold/ URLs exist, /dead/ and /miss/ are genuine 404s. */
function originTable(trace) {
  const store = new Map();
  for (const url of trace) {
    if (!url.startsWith("/hot/") && !url.startsWith("/cold/")) continue;
    if (!store.has(url)) store.set(url, `body:${url}`);
  }
  return store;
}

function plainCacheSimulation(trace, origin, capacity = 64) {
  const cache = new Map();
  const responses = [];
  const stats = { hits: 0, misses: 0, originLookups: 0, evictions: 0 };
  for (const url of trace) {
    if (cache.has(url)) {
      stats.hits += 1;
      responses.push([200, cache.get(url)]);
      cache.delete(url);
      cache.set(url, responses[responses.length - 1][1]);
      continue;
    }
    stats.misses += 1;
    stats.originLookups += 1;
    const body = origin.get(url);
    if (body === undefined) {
      responses.push([404, null]);
      continue;
    }
    responses.push([200, body]);
    cache.set(url, body);
    if (cache.size > capacity) {
      const oldest = cache.keys().next().value;
      cache.delete(oldest);
      stats.evictions += 1;
    }
  }
  return { responses, stats };
}

/**
 * Bloom-backed cache. The Bloom filter is a small front filter over a (large, hash-resident) exact
 * set of URLs already confirmed absent: a filter answer of "definitely absent" skips that exact
 * lookup entirely, and a "maybe" answer falls through to the exact set and, on a false positive, to
 * the origin - so answers stay exact while the exact-set probes (and, through the negative set, the
 * origin lookups) are reduced. Everything is counted: hits, misses, origin lookups, negative-cache
 * hits, exact-set probes, exact-set probes skipped and false-positive probes.
 */
function bloomCacheSimulation(trace, origin, filter, capacity = 64, negativeCapacity = 256) {
  const cache = new Map();
  const missing = new Set();
  const missingOrder = [];
  const responses = [];
  const stats = { hits: 0, misses: 0, originLookups: 0, negativeHits: 0, exactProbes: 0,
    exactProbesSaved: 0, falsePositiveProbes: 0, evictions: 0, negativeEvictions: 0 };
  const rememberMissing = (url) => {
    if (missing.has(url)) return;
    missing.add(url);
    missingOrder.push(url);
    filter.add(url);
    if (missing.size > negativeCapacity) {
      const oldest = missingOrder.shift();
      missing.delete(oldest);
      stats.negativeEvictions += 1;
    }
  };
  for (const url of trace) {
    if (cache.has(url)) {
      stats.hits += 1;
      const body = cache.get(url);
      responses.push([200, body]);
      cache.delete(url);
      cache.set(url, body);
      continue;
    }
    stats.misses += 1;
    if (!filter.maybeContains(url)) {
      // the exact negative set is guaranteed not to contain this URL: skip that probe
      stats.exactProbesSaved += 1;
    } else {
      stats.exactProbes += 1;
      if (missing.has(url)) {
        stats.negativeHits += 1;
        responses.push([404, null]);               // a true positive: no origin lookup needed
        continue;
      }
      stats.falsePositiveProbes += 1;              // fall through: exactness is preserved
    }
    stats.originLookups += 1;
    const body = origin.get(url);
    if (body === undefined) {
      responses.push([404, null]);
      rememberMissing(url);
      continue;
    }
    responses.push([200, body]);
    cache.set(url, body);
    if (cache.size > capacity) {
      const oldest = cache.keys().next().value;
      cache.delete(oldest);
      stats.evictions += 1;
    }
  }
  return { responses, stats, missingSize: missing.size };
}

/** The invariant behind "no false negatives": every remembered absent URL is in the filter. */
function checkNoFalseNegatives(filter, missing) {
  for (const url of missing) {
    if (!filter.maybeContains(url)) return false;
  }
  return true;
}

function measureFalsePositiveRate(filter, probes, seed) {
  let falsePositives = 0;
  let s = seed;
  for (let i = 0; i < probes; i++) {
    s = (s * 75 + 74) % 65537;
    const key = `/probe/${i}/${s}`;
    if (filter.maybeContains(key)) falsePositives += 1;
  }
  return falsePositives / probes;
}

function main() {
  const assert = require("node:assert");

  // (1) sizing formulas and their documented values.
  const n = 1000;
  const m = optimalBits(n, 0.01);
  const k = optimalHashes(m, n);
  assert.strictEqual(m, 9586, "m = ceil(-n ln p / ln(2)^2) for n = 1000, p = 0.01");
  assert.strictEqual(k, 7, "k = round((m / n) ln 2)");
  assert.strictEqual(optimalHashes(9586, 1000), 7);
  assert.ok(optimalBits(100, 0.01) < m, "a smaller target needs fewer bits");
  assert.ok(optimalBits(1000, 0.001) > m, "a tighter target needs more bits");

  const filter = new BloomFilter(m, k);
  const inserted = [];
  for (let i = 0; i < n; i++) inserted.push(`/cold/${i}`);

  // (2) no false negatives, ever.
  for (const key of inserted) filter.add(key);
  for (const key of inserted) {
    assert.ok(filter.maybeContains(key), `no false negative for ${key}`);
  }
  const before = Uint8Array.from(filter.words);
  filter.add(inserted[0]);
  filter.add(inserted[0]);
  assert.deepStrictEqual(Array.from(filter.words), Array.from(before),
    "re-adding a key leaves the bit vector unchanged");
  assert.strictEqual(filter.bitCount(), filter.words.reduce((a, b) => a + b, 0));

  // (3) empirical false-positive rate matches (1 - e^(-kn/m))^k.
  const fpRate = measureFalsePositiveRate(filter, 20000, 777);
  const theory = filter.theoreticalFp();
  assert.ok(Math.abs(theory - 0.01) <= 0.001, "theory for n = 1000, m = 9586, k = 7 is ~0.010");
  assert.ok(Math.abs(fpRate - theory) <= 0.005,
    `measured fp ${fpRate} within 0.005 of theory ${theory}`);
  assert.ok(fpRate > 0.0, "some false positives do occur (the filter is probabilistic)");

  // (4) the rate grows with the load, and saturates.
  const light = BloomFilter.forTarget(n, 0.01);
  for (const key of inserted) light.add(key);
  const heavy = BloomFilter.forTarget(n, 0.01);
  for (let i = 0; i < 4 * n; i++) heavy.add(`/heavy/${i}`);
  const lightRate = measureFalsePositiveRate(light, 20000, 12345);
  const heavyRate = measureFalsePositiveRate(heavy, 20000, 12345);
  assert.ok(heavyRate > lightRate, `4x the insertions raise the fp rate (${heavyRate})`);
  const saturated = BloomFilter.forTarget(n, 0.01);
  for (let i = 0; i < 40 * n; i++) saturated.add(`/dense/${i}`);
  const saturatedRate = measureFalsePositiveRate(saturated, 5000, 999);
  assert.ok(saturatedRate > 0.5, `an over-full filter reports almost everything (${saturatedRate})`);

  // (5) an empty filter rejects every probe; a full bit vector accepts everything.
  const empty = new BloomFilter(512, 4);
  assert.ok(!empty.maybeContains("/anything"), "an empty filter has no false positives");
  const full = new BloomFilter(512, 4);
  for (let i = 0; i < 512; i++) full.words[i] = 1;
  assert.ok(full.maybeContains("/anything"), "a saturated bit vector accepts everything");

  // (6) the two cache simulations must agree response-for-response, with the sane filter.
  const trace = requestTrace(4000, 20260116);
  const origin = originTable(trace);
  const plain = plainCacheSimulation(trace, origin);
  const bloomFilter = BloomFilter.forTarget(2000, 0.01);
  const negativeCapacity = 512;
  const bloomed = bloomCacheSimulation(trace, origin, bloomFilter, 64, negativeCapacity);
  assert.strictEqual(plain.responses.length, trace.length);
  assert.deepStrictEqual(bloomed.responses, plain.responses,
    "the Bloom-backed cache returns exactly the same statuses and bodies");
  for (let i = 0; i < trace.length; i++) {
    const [status, body] = plain.responses[i];
    const exists = origin.has(trace[i]);
    assert.strictEqual(status, exists ? 200 : 404, "status matches the ground-truth origin");
    assert.strictEqual(body, exists ? origin.get(trace[i]) : null);
  }

  // (7) fewer origin lookups, and the filter proving absence cheaply.
  assert.ok(bloomed.stats.originLookups < plain.stats.originLookups,
    `negative caching saves origin work (${bloomed.stats.originLookups} < `
    + `${plain.stats.originLookups})`);
  assert.ok(plain.stats.originLookups / bloomed.stats.originLookups >= 1.3,
    `documented >= 1.3x fewer origin lookups `
    + `(${(plain.stats.originLookups / bloomed.stats.originLookups).toFixed(2)}x)`);
  assert.ok(bloomed.stats.negativeHits > 0, "repeated 404s are answered from the negative set");
  assert.ok(bloomed.stats.originLookups <= bloomed.stats.misses,
    "the negative cache never adds extra origin lookups");
  assert.strictEqual(plain.stats.originLookups, plain.stats.misses,
    "the plain cache misses are exactly its origin lookups");
  assert.strictEqual(bloomed.stats.hits, plain.stats.hits, "hit counts are identical");
  assert.strictEqual(bloomed.stats.misses, plain.stats.misses, "miss counts are identical");
  assert.strictEqual(bloomed.stats.evictions, plain.stats.evictions,
    "the bounded response cache evicts identically in both runs");
  assert.ok(bloomed.stats.exactProbesSaved > bloomed.stats.exactProbes,
    `the filter answers most memberships without the exact set (${bloomed.stats.exactProbesSaved} `
    + `skipped vs ${bloomed.stats.exactProbes} probes)`);
  assert.strictEqual(bloomed.stats.falsePositiveProbes, 0,
    "with a properly sized filter and a 512-entry negative set nothing is a false positive");
  assert.ok(bloomed.missingSize > 0 && bloomed.missingSize <= negativeCapacity,
    "the negative set is bounded and non-empty");
  const deadUrls = new Set(trace.filter((url) => !origin.has(url)));
  assert.strictEqual(checkNoFalseNegatives(bloomFilter, deadUrls), true,
    "no false negative over every 404 URL of the trace");
  const savedShare = bloomed.stats.exactProbesSaved
    / (bloomed.stats.exactProbesSaved + bloomed.stats.exactProbes);
  assert.ok(savedShare >= 0.5, `the filter skips >= 50% of exact probes (${savedShare})`);

  // (8) degraded configurations keep answers exact while costing extra probes/lookups.
  const weakFilter = new BloomFilter(64, 4);
  const weak = bloomCacheSimulation(trace, origin, weakFilter, 64, negativeCapacity);
  assert.deepStrictEqual(weak.responses, bloomed.responses,
    "a saturated filter still returns exactly the same responses");
  assert.deepStrictEqual(weak.responses, plain.responses, "and matches the plain cache too");
  assert.ok(weak.stats.falsePositiveProbes > bloomed.stats.falsePositiveProbes,
    `the tiny filter is a useless front filter (${weak.stats.falsePositiveProbes} false positives)`);
  assert.ok(weak.stats.exactProbes >= 2 * bloomed.stats.exactProbes,
    `the exact set is consulted at least twice as often (${weak.stats.exactProbes} vs `
    + `${bloomed.stats.exactProbes})`);
  assert.strictEqual(weak.stats.originLookups, bloomed.stats.originLookups,
    "a false positive on an unknown URL lands on exactly the lookup the filter would have made");

  const smallNegative = bloomCacheSimulation(trace, origin, BloomFilter.forTarget(2000, 0.01), 64, 64);
  assert.deepStrictEqual(smallNegative.responses, plain.responses,
    "shrinking the negative set changes no answer");
  assert.ok(smallNegative.stats.negativeEvictions > 0, "the 64-entry negative set does evict");
  assert.ok(smallNegative.stats.falsePositiveProbes > 0,
    "evicted dead links come back as filter false positives");
  assert.strictEqual(smallNegative.stats.originLookups,
    bloomed.stats.originLookups + smallNegative.stats.falsePositiveProbes,
    "every evicted-then-requested dead link costs exactly one extra origin lookup");
  assert.ok(plain.stats.originLookups / smallNegative.stats.originLookups >= 1.3,
    "the smaller negative set is still a >= 1.3x win over no negative caching");

  // (9) determinism of the whole simulation.
  const filterAgain = BloomFilter.forTarget(2000, 0.01);
  const bloomedAgain = bloomCacheSimulation(trace, origin, filterAgain, 64, negativeCapacity);
  assert.deepStrictEqual(bloomedAgain.responses, bloomed.responses, "deterministic responses");
  assert.deepStrictEqual(bloomedAgain.stats, bloomed.stats, "deterministic statistics");
}

module.exports = { DEFAULT_N, DEFAULT_P, fnv1a, secondHash, optimalBits, optimalHashes,
  BloomFilter, requestTrace, originTable, plainCacheSimulation, bloomCacheSimulation,
  measureFalsePositiveRate, checkNoFalseNegatives };

if (require.main === module) {
  main();
  console.log("PASS");
}
