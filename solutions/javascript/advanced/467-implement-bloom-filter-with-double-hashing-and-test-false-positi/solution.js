/** Task 467 - Implement bloom filter with double hashing and test false positive. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Bloom filter with m = 2^16 bits and k = 7 probes via double hashing:
 * h1 = mix32(key), h2 = mix32(key ^ 0x9E3779B9) | 1 (forced odd, hence coprime to
 * the power-of-two m, so the 7 probes are pairwise distinct),
 * probe i = (h1 + i*h2) mod m.
 *
 * Checks: a hand-verifiable small filter (single insert sets exactly 7 distinct bits
 * and is idempotent), then insert 10000 distinct member keys (LCG values in
 * [0, 65536)), verify all 10000 are reported present (zero false negatives), and
 * test 10000 disjoint non-member keys (200000 + i) counting false positives. The
 * documented bound is FP_MAX = 800; the theoretical rate (1 - e^(-k*n/m))^k ~ 0.0524
 * gives ~524 expected and the measured count on this fixed key set is 488 (asserted
 * <= 800 with a >= 444 sanity floor).
 */
"use strict";

const M_BITS = 1 << 16;
const PROBES = 7;
const FP_MAX = 800;

function mix32(x) {
  x = x >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x45d9f3b) >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x45d9f3b) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

class BloomFilter {
  constructor() {
    this.data = new Uint8Array(M_BITS / 8);
  }

  positions(key) {
    const h1 = mix32(key);
    const h2 = (mix32(key ^ 0x9e3779b9) | 1) >>> 0;
    const out = [];
    for (let i = 0; i < PROBES; i++) out.push((h1 + i * h2) % M_BITS);
    return out;
  }

  add(key) {
    for (const pos of this.positions(key)) {
      this.data[pos >> 3] |= 1 << (pos & 7);
    }
  }

  contains(key) {
    for (const pos of this.positions(key)) {
      if (((this.data[pos >> 3] >> (pos & 7)) & 1) === 0) return false;
    }
    return true;
  }

  popcount() {
    let total = 0;
    for (const b of this.data) {
      let x = b;
      while (x) {
        total += x & 1;
        x >>= 1;
      }
    }
    return total;
  }
}

function makeLcg(state) {
  let seed = state;
  return () => (seed = (seed * 75 + 74) % 65537);
}

module.exports = { BloomFilter };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-verifiable small filter: 7 distinct probes, idempotent insert.
  const tiny = new BloomFilter();
  assert.strictEqual(new Set(tiny.positions(5)).size, PROBES);
  tiny.add(5);
  assert.strictEqual(tiny.popcount(), PROBES);
  assert.ok(tiny.contains(5));
  tiny.add(5);
  assert.strictEqual(tiny.popcount(), PROBES);
  assert.ok(!tiny.contains(6));

  // Workload: 10000 distinct members from the LCG, 10000 disjoint non-members.
  const bloom = new BloomFilter();
  const lcg = makeLcg(3141592);
  const members = [];
  const seen = new Set();
  while (members.length < 10000) {
    const value = lcg();
    if (!seen.has(value)) {
      seen.add(value);
      members.push(value);
    }
  }
  for (const key of members) bloom.add(key);
  for (const key of members) assert.ok(bloom.contains(key)); // zero false negatives
  let falsePositives = 0;
  for (let i = 0; i < 10000; i++) if (bloom.contains(200000 + i)) falsePositives++;
  assert.ok(falsePositives <= FP_MAX, `fp bound ${falsePositives}`); // measured 488
  assert.ok(falsePositives >= 444, `fp sanity floor ${falsePositives}`);
  console.log("PASS");
}
