/** Task 254 - Implement bloom filter insert/check (probabilistic). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

const M_BITS = 1024;
const K_HASHES = 7;
// Deterministic values for the exact test set below (cross-language check).
const EXPECTED_BITS = 503;
const EXPECTED_FP = 74;

/** FNV-1a 32-bit (Math.imul keeps the 32-bit wrap exact). */
function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    h = (h ^ text.charCodeAt(i)) >>> 0;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

/** MurmurHash3 avalanche: raw FNV-1a low bits are poorly mixed for double hashing. */
function fmix32(x) {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  x = (x ^ (x >>> 13)) >>> 0;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x;
}

/** Probe i is (h1 + i * h2) mod m with h2 odd: double hashing with distinct probes. */
function positions(item, bits, hashes) {
  const h1 = fmix32(fnv1a(item));
  const h2 = (fmix32(fnv1a(item + "#")) | 1) >>> 0;
  const out = [];
  for (let i = 0; i < hashes; i += 1) out.push((h1 + i * h2) % bits);
  return out;
}

class BloomFilter {
  constructor(bits = M_BITS, hashes = K_HASHES) {
    this.m = bits;
    this.k = hashes;
    this.words = new Uint8Array(bits);
  }

  add(item) {
    for (const p of positions(item, this.m, this.k)) this.words[p] = 1;
  }

  maybeContains(item) {
    return positions(item, this.m, this.k).every((p) => this.words[p] === 1);
  }

  popcount() {
    let count = 0;
    for (let i = 0; i < this.words.length; i += 1) count += this.words[i];
    return count;
  }
}

module.exports = { M_BITS, K_HASHES, fnv1a, fmix32, positions, BloomFilter };

if (require.main === module) {
  const assert = require("node:assert");

  const f = new BloomFilter();
  assert.strictEqual(f.popcount(), 0);
  for (const probe of ["item-0", "x", "hello"]) assert.strictEqual(f.maybeContains(probe), false);

  for (let i = 0; i < 100; i += 1) f.add(`item-${i}`);
  for (let i = 0; i < 100; i += 1) assert.strictEqual(f.maybeContains(`item-${i}`), true); // no false negatives
  const filled = f.popcount();
  f.add("item-0");
  assert.strictEqual(f.popcount(), filled);              // inserting again is a no-op

  let falsePositives = 0;
  for (let i = 0; i < 10000; i += 1) if (f.maybeContains(`other-${i}`)) falsePositives += 1;
  assert.strictEqual(f.popcount(), EXPECTED_BITS);
  assert.strictEqual(falsePositives, EXPECTED_FP);
  // Documented bound: theory gives (1 - e^(-k*n/m))^k ~ 0.7% for n=100, m=1024, k=7;
  // the measured rate must stay under 2% (200 out of 10000 queries).
  assert.ok(falsePositives <= 200);

  const small = new BloomFilter(256, 4);
  for (const item of ["a", "b", "c"]) small.add(item);
  for (const item of ["a", "b", "c"]) assert.strictEqual(small.maybeContains(item), true);
  console.log("PASS");
}
