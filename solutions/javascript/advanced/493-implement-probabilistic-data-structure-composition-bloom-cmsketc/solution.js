/**
 * Task 493 - Implement probabilistic data structure composition (bloom + cmsketch). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A SketchPipeline composes two sketches over the same item stream: a Bloom filter (relaxed
 * membership) and a Count-Min sketch (frequency estimation with d rows and w counters).
 * membership(item) = bloom.maybeContains(item) AND cms.estimate(item) >= 1 - so a positive
 * answer always implies an estimated frequency of at least 1, and a Bloom false positive whose
 * every CMS counter is still empty is rejected (the pipeline tightens membership without ever
 * introducing a false negative). Hashing is the stable task-254 scheme: h1 = fmix32(fnv1a(item)),
 * h2 = fmix32(fnv1a(item + '#')) | 1 for the Bloom probes, and per-row fmix32(fnv1a(item + '|r'))
 * for the CMS; all arithmetic is 32-bit integer work (Math.imul keeps the wrap exact), identical
 * in all four languages. Documented guarantees: CMS estimates never underestimate the true
 * count and overestimate by at most epsilon*N with epsilon = e/w and delta = e^(-d) (w = 256,
 * d = 4, N = 754 updates); on 5000 absent probes the Bloom rate (m = 1024, k = 7, n = 250
 * items, theory ~25.6%) stays under the documented 28% tolerance, and the pipeline rejects
 * most of those Bloom hits because their CMS counters are still empty - with the same Bloom
 * filter and a width-1024 CMS almost all are rejected. The exact popcounts and false-positive
 * counts are asserted as cross-language constants.
 */
"use strict";

/** FNV-1a 32-bit (Math.imul keeps the 32-bit wrap exact). */
function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h = (h ^ text.charCodeAt(i)) >>> 0;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

/** MurmurHash3 avalanche: raw FNV-1a low bits are poorly mixed. */
function fmix32(x) {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  x = (x ^ (x >>> 13)) >>> 0;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x;
}

class BloomFilter {
  constructor(bits, hashes) {
    this.m = bits;
    this.k = hashes;
    this.words = new Uint8Array(bits);
  }

  positions(item) {
    const h1 = fmix32(fnv1a(item));
    const h2 = (fmix32(fnv1a(item + "#")) | 1) >>> 0;
    const out = [];
    for (let i = 0; i < this.k; i++) out.push((h1 + i * h2) % this.m);
    return out;
  }

  add(item) {
    for (const p of this.positions(item)) this.words[p] = 1;
  }

  maybeContains(item) {
    for (const p of this.positions(item)) {
      if (this.words[p] === 0) return false;
    }
    return true;
  }

  popcount() {
    let total = 0;
    for (let i = 0; i < this.words.length; i++) total += this.words[i];
    return total;
  }
}

class CountMinSketch {
  constructor(width, rows) {
    this.w = width;
    this.d = rows;
    this.counters = [];
    for (let r = 0; r < rows; r++) this.counters.push(new Int32Array(width));
  }

  column(row, item) {
    return fmix32(fnv1a(item + "|" + row)) % this.w;
  }

  add(item, delta = 1) {
    for (let row = 0; row < this.d; row++) {
      this.counters[row][this.column(row, item)] += delta;
    }
  }

  estimate(item) {
    let best = Infinity;
    for (let row = 0; row < this.d; row++) {
      best = Math.min(best, this.counters[row][this.column(row, item)]);
    }
    return best;
  }

  total() {
    let total = 0;
    for (const row of this.counters) {
      for (let c = 0; c < row.length; c++) total += row[c];
    }
    return total;
  }
}

class SketchPipeline {
  constructor(bloomBits = 1024, bloomHashes = 7, cmsWidth = 256, cmsRows = 4) {
    this.bloom = new BloomFilter(bloomBits, bloomHashes);
    this.cms = new CountMinSketch(cmsWidth, cmsRows);
  }

  add(item, count = 1) {
    this.bloom.add(item);
    this.cms.add(item, count);
  }

  estimate(item) {
    return this.cms.estimate(item);
  }

  /** True only when the Bloom filter accepts AND the CMS sees a nonzero count. */
  membership(item) {
    return this.bloom.maybeContains(item) && this.cms.estimate(item) >= 1;
  }
}

/** Deterministic insertion stream: item key-<j> gets 1 + seed % 5 copies. */
function lcgCounts(itemCount, seed) {
  const counts = new Map();
  for (let j = 0; j < itemCount; j++) {
    seed = (seed * 75 + 74) % 65537;
    counts.set(`key-${j}`, 1 + (seed % 5));
  }
  return counts;
}

function main() {
  const assert = require("node:assert");

  const counts = lcgCounts(250, 123);
  let totalUpdates = 0;
  for (const count of counts.values()) totalUpdates += count;
  const pipeline = new SketchPipeline();
  for (const [item, count] of counts) pipeline.add(item, count);

  const slack = (Math.E / pipeline.cms.w) * totalUpdates;
  for (const [item, trueCount] of counts) {
    assert.strictEqual(pipeline.membership(item), true, "no false negatives");
    const estimate = pipeline.estimate(item);
    assert.ok(estimate >= trueCount, "CMS must never underestimate");
    assert.ok(estimate <= trueCount + slack, "CMS additive error bound");
    assert.strictEqual(pipeline.bloom.maybeContains(item), true);
  }

  for (const item of counts.keys()) {
    if (pipeline.membership(item)) assert.ok(pipeline.estimate(item) >= 1);
  }

  assert.strictEqual(pipeline.estimate("never-seen"), 0);
  assert.strictEqual(pipeline.membership("never-seen"), false);

  const probes = [];
  for (let j = 0; j < 5000; j++) probes.push(`probe-${j}`);
  const bloomFp = probes.filter((item) => pipeline.bloom.maybeContains(item)).length;
  const pipelineFp = probes.filter((item) => pipeline.membership(item)).length;
  assert.ok(bloomFp <= 1400, "documented Bloom tolerance");
  assert.ok(pipelineFp <= bloomFp - 1000, "the pipeline must reject most Bloom false hits");
  assert.ok(pipelineFp <= 200, "documented pipeline tolerance");

  const tight = new SketchPipeline(1024, 7, 1024, 4);
  for (const [item, count] of counts) tight.add(item, count);
  const tightBloomFp = probes.filter((item) => tight.bloom.maybeContains(item)).length;
  const tightPipelineFp = probes.filter((item) => tight.membership(item)).length;
  assert.ok(tightPipelineFp < tightBloomFp, "cold counters must reject some Bloom hits");
  assert.ok(tightPipelineFp <= tightBloomFp - 1000);
  for (const [item, trueCount] of counts) {
    assert.strictEqual(tight.membership(item), true);
    assert.ok(tight.estimate(item) >= trueCount);
    assert.ok(tight.estimate(item) <= trueCount + 1, "wider sketch: error <= e*N/w");
  }

  assert.strictEqual(totalUpdates, 754);
  assert.strictEqual(pipeline.bloom.popcount(), 842);
  assert.strictEqual(tight.bloom.popcount(), 842);
  assert.strictEqual(bloomFp, 1296);
  assert.strictEqual(pipelineFp, 194);
  assert.strictEqual(tightBloomFp, 1296);
  assert.strictEqual(tightPipelineFp, 5);

  const solo = new SketchPipeline();
  solo.add("solo", 3);
  assert.strictEqual(solo.membership("solo"), true);
  assert.strictEqual(solo.estimate("solo"), 3);
  for (const probe of ["solo#", "solo1", "sol", "proxy-7"]) {
    if (solo.membership(probe)) assert.ok(solo.estimate(probe) >= 1);
  }
  assert.strictEqual(solo.cms.total(), 12, "4 rows x 3 copies");
}

module.exports = { fnv1a, fmix32, BloomFilter, CountMinSketch, SketchPipeline, lcgCounts };

if (require.main === module) {
  main();
  console.log("PASS");
}
