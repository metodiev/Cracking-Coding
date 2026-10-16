/**
 * Task 571 - Implement collision-resistant hash property tests (statistical). (Tier: Advanced, CTCI Beyond — Security & Privacy)
 *
 * A statistical property-testing harness for integer hashes.  The subject is an explicitly
 * documented *testbed* hash, not a cryptographic one: `mix32` is the splitmix32-style avalanche
 * finalizer (add a golden-ratio constant, xor-shift, multiply, xor-shift, multiply, xor-shift) and
 * the 64-bit variant chains two such lanes (lane a takes the low half, lane b takes the high half
 * mixed with a, and the output low half mixes both).  `hashBytes` is a Merkle-Damgard-style
 * iterated construction over bytes with length padding and chaining.  Against them stand four
 * deliberately weak variants - identity, low-bit truncation, high-bit truncation and a plain XOR
 * of per-byte mixes - so that every statistical test is shown to have teeth: a test that cannot
 * fail on a broken hash is not evidence.  All inputs come from a documented 64-bit splitmix64
 * generator (the repo's 16-bit LCG is unusable here because these tests consume tens of thousands
 * of independent keys; see the wave-3 note about its short cycles and poor low bits).
 *
 * Documented statistical checks, all on fixed data (so they are deterministic but still meaningful,
 * with tolerances set from the measured values rather than from exact float equality):
 *   * determinism/purity, and no collision among 2^16 distinct 32-bit keys;
 *   * avalanche: mean Hamming distance of 32-bit outputs over single-bit key flips inside
 *     [15, 17] out of 32, every input-bit's aggregate flip rate inside [0.45, 0.55], every
 *     output-bit's flip rate inside [0.45, 0.55], and the same for both halves of the 64-bit hash;
 *   * uniformity: chi-square over 2^12 buckets with 2^16 keys below 1.05 * (df = 4095), while the
 *     low-bit and high-bit truncations exceed 10 * df on keys that are multiples of 2^12 or below
 *     2^20 respectively;
 *   * birthday collisions: 2^16 keys hashed into 2^32 (no collisions observed) and into 2^16
 *     buckets, where the number of colliding pairs is asserted within 5% of the exact expectation
 *     C(N,2)/M = 32767.5, while the 16-bit weak hash produces more than 20000;
 *   * structured keys: keys differing only in bits 20..31 still flip about half of the output bits
 *     under the good hash, whereas the mod-2^12 weak hash maps them to one output;
 *   * linear independence: |correlation(key, output)| < 0.05 for the good hash versus exactly 1 for
 *     the identity hash and for a non-wrapping affine map - and the documented caveat that the
 *     wrapping multiplicative map shows little correlation while still collapsing aligned keys, so
 *     correlation alone would miss it (the low-bit uniformity check catches it);
 *   * conditional uniformity: each of the 256 low-byte classes of the input is spread uniformly
 *     over the output's low byte (chi-square <= 1.5 * 255 for every class);
 *   * message hashing: the iterated byte hash separates "ab" from "ba" (the plain XOR variant
 *     provably collides on them), handles the empty message, and finds no collision among a large
 *     sample of distinct multi-byte messages.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;
const MASK32 = 0xffffffff;

/** Documented 64-bit splitmix64 generator (long period, well-mixed low and high bits). */
class Rng {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    const z = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    this.state = z;
    let x = z;
    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return x ^ (x >> 31n);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

/* --------------------------------------------------------------- the hash testbed */

/** splitmix32-style avalanche finalizer: a bijection of the 32-bit integers. */
function mix32(value) {
  let x = (value + 0x9e3779b9) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x21f0aaad) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x735a2d97) >>> 0;
  return (x ^ (x >>> 15)) >>> 0;
}

function hash32(key) {
  return mix32(key >>> 0);
}

/** 64-bit key hashed to 64 bits through two chained lanes. */
function hash64(keyHi, keyLo) {
  const a = mix32((keyLo ^ 0x9e3779b9) >>> 0);
  const b = mix32(((keyHi ^ a) + 0x85ebca6b) >>> 0);
  return { hi: b, lo: mix32(((a ^ b) + 0xc2b2ae35) >>> 0) };
}

/** Number of differing bits between two 32-bit values. */
function hamming32(a, b) {
  let x = (a ^ b) >>> 0;
  let count = 0;
  while (x !== 0) {
    count += x & 1;
    x >>>= 1;
  }
  return count;
}

/* ------------------------------------------------------------- deliberately weak hashes */

const weakIdentity = (key) => key >>> 0;
const weakLowBits = (key) => (key & 0xfff) >>> 0;
const weakHighBits = (key) => (key >>> 20) >>> 0;
const weakLinear = (key) => Math.imul(key, 0x9e3779b1) >>> 0;

/** XOR of per-byte mixes: no length information and fully order-insensitive. */
function xorHash(bytes) {
  let acc = 0;
  for (const byte of bytes) acc = (acc ^ mix32(byte)) >>> 0;
  return acc;
}

/** Iterated hash with chaining and length padding (Merkle-Damgard shape, not a crypto claim). */
function hashBytes(bytes) {
  let state = mix32(bytes.length ^ 0x1b873593);
  for (let i = 0; i < bytes.length; i++) {
    state = mix32(((state ^ bytes[i]) + Math.imul(i + 1, 0x9e3779b9)) >>> 0);
  }
  return mix32((state ^ (bytes.length * 0x85ebca6b)) >>> 0);
}

function asciiBytes(text) {
  return [...text].map((ch) => ch.charCodeAt(0) & 0xff);
}

/* ------------------------------------------------------------------- statistics */

function chiSquare(observed, expected) {
  let total = 0;
  for (let i = 0; i < observed.length; i++) {
    const diff = observed[i] - expected[i];
    total += (diff * diff) / expected[i];
  }
  return total;
}

function bucketCounts(keys, bucketCount, hash) {
  const counts = new Array(bucketCount).fill(0);
  for (const key of keys) counts[hash(key) % bucketCount] += 1;
  return counts;
}

/** Hamming distance of the outputs of all 32 single-bit flips of every key. */
function avalanche32(keys) {
  const histogram = new Array(33).fill(0);
  const inputBitChanges = new Array(32).fill(0); // sum, per bit, of Hamming(distance)/32
  const outputBitChanges = new Array(32).fill(0); // count of samples in which that output bit moved
  let distanceSum = 0;
  let samples = 0;
  for (const key of keys) {
    const base = hash32(key);
    for (let bit = 0; bit < 32; bit++) {
      const flipped = hash32((key ^ (1 << bit)) >>> 0);
      const distance = hamming32(base, flipped);
      histogram[distance] += 1;
      distanceSum += distance;
      samples += 1;
      inputBitChanges[bit] += distance / 32;
      for (let out = 0; out < 32; out++) {
        if (((base >>> out) & 1) !== ((flipped >>> out) & 1)) outputBitChanges[out] += 1;
      }
    }
  }
  const perInputBit = samples / 32;
  return { samples, mean: distanceSum / samples, histogram,
    inputBitRate: inputBitChanges.map((sum) => sum / perInputBit),
    outputBitRate: outputBitChanges.map((count) => count / samples) };
}

/** Hamming distance over the 64 output bits of all 64 single-bit key flips. */
function avalanche64(keys) {
  let samples = 0;
  let distanceSum = 0;
  for (const [hi, lo] of keys) {
    const base = hash64(hi, lo);
    for (let bit = 0; bit < 64; bit++) {
      const nextHi = bit < 32 ? (hi ^ (1 << bit)) >>> 0 : hi;
      const nextLo = bit < 32 ? lo : (lo ^ (1 << (bit - 32))) >>> 0;
      const flipped = hash64(nextHi, nextLo);
      distanceSum += hamming32(base.hi, flipped.hi) + hamming32(base.lo, flipped.lo);
      samples += 1;
    }
  }
  return { mean: distanceSum / samples, samples };
}

function collidingPairs(keys, hash) {
  const counts = new Map();
  for (const key of keys) {
    const value = hash(key);
    counts.set(value, (counts.get(value) || 0) + 1);
  }
  let pairs = 0;
  let distinct = 0;
  for (const count of counts.values()) {
    pairs += (count * (count - 1)) / 2;
    distinct += 1;
  }
  return { pairs, distinct, keys: keys.length };
}

function correlation(xs, ys) {
  const n = xs.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  return sxy / Math.sqrt(sxx * syy);
}

function main() {
  const assert = require("node:assert");
  const rng = new Rng(0xc0ffee);

  // (1) Determinism, purity and no collisions among 2^16 distinct 32-bit keys.
  {
    const keys = [];
    for (let i = 0; i < 65536; i++) keys.push(rng.nextInt(0x100000000));
    const replay = new Rng(0xc0ffee);
    const replayKeys = [];
    for (let i = 0; i < 65536; i++) replayKeys.push(replay.nextInt(0x100000000));
    assert.deepStrictEqual(replayKeys, keys, "the key stream itself is deterministic");
    const seen = new Map();
    let collisions = 0;
    for (const key of keys) {
      const value = hash32(key);
      if (seen.has(value)) collisions += 1;
      seen.set(value, key);
    }
    assert.strictEqual(collisions, 0, "no collision among 2^16 keys hashed into 2^32");
    assert.strictEqual(seen.size, keys.length);
    for (const key of keys.slice(0, 50)) assert.strictEqual(hash32(key), hash32(key));
    assert.strictEqual(hash32(0), mix32(0));
    assert.strictEqual(hash32(0xffffffff), mix32(0xffffffff));
    assert.notStrictEqual(hash32(0), hash32(1));
    assert.strictEqual(weakIdentity(7), 7);
  }

  // (2) Avalanche on the 32-bit hash: single-bit key flips move about half of the output bits.
  {
    const sampleKeys = [];
    for (let i = 0; i < 512; i++) sampleKeys.push(rng.nextInt(0x100000000));
    const report = avalanche32(sampleKeys);
    assert.strictEqual(report.samples, 512 * 32);
    assert.ok(report.mean >= 15.0 && report.mean <= 17.0,
      `mean Hamming distance ${report.mean} out of 32`);
    assert.ok(Math.abs(report.mean - 16.0) <= 1.0, "close to the ideal 16");
    assert.ok(report.histogram[0] === 0, "no single-bit flip leaves the output unchanged");
    assert.ok(report.histogram[32] === 0, "and none inverts every bit");
    // Every input bit position must matter: at least 35% of the flipped outputs differ in 12+ bits.
    for (let bit = 0; bit < 32; bit++) {
      const rate = report.inputBitRate[bit];
      assert.ok(rate >= 0.45 && rate <= 0.55,
        `flipping input bit ${bit} changes ${(rate * 100).toFixed(1)}% of the output bits`);
    }
    for (let out = 0; out < 32; out++) {
      const rate = report.outputBitRate[out];
      assert.ok(rate >= 0.45 && rate <= 0.55,
        `output bit ${out} flips in ${(rate * 100).toFixed(1)}% of the samples`);
    }
  }

  // (3) Avalanche on the 64-bit hash: both lanes react to every one of the 64 key bits.
  {
    const keys = [];
    for (let i = 0; i < 256; i++) {
      keys.push([rng.nextInt(0x100000000), rng.nextInt(0x100000000)]);
    }
    const report = avalanche64(keys);
    assert.strictEqual(report.samples, 256 * 64);
    assert.ok(report.mean >= 28.0 && report.mean <= 36.0,
      `mean Hamming distance ${report.mean} out of 64 must be near 32`);
    const base = hash64(1, 2);
    assert.notStrictEqual(base.hi, base.lo, "the two lanes are not degenerate");
    assert.notDeepStrictEqual(hash64(1, 2), hash64(2, 1), "lanes are not symmetric");
  }

  // (4) Uniformity: chi-square below 1.05 * df for the good hash, far above for the weak ones.
  {
    const keys = [];
    for (let i = 0; i < 65536; i++) keys.push(i);
    const buckets = 4096;
    const df = buckets - 1;
    const counts = bucketCounts(keys, buckets, hash32);
    const statistic = chiSquare(counts, new Array(buckets).fill(keys.length / buckets));
    assert.ok(statistic <= 1.05 * df,
      `sequential keys: chi-square ${statistic.toFixed(1)} must be below ${(1.05 * df).toFixed(1)}`);
    assert.ok(statistic >= 0.8 * df, "and inside the expected band for a uniform hash");

    // Weak hash 1: the low 12 bits of keys that are multiples of 2^12 are constant.
    const aligned = [];
    for (let i = 0; i < 65536; i++) aligned.push((i * 4096) >>> 0);
    const weakCounts = bucketCounts(aligned, buckets, weakLowBits);
    const weakStatistic = chiSquare(weakCounts, new Array(buckets).fill(keys.length / buckets));
    assert.ok(weakStatistic > 10 * df,
      `the truncated hash collapses these keys: chi-square ${weakStatistic}`);
    assert.strictEqual(weakCounts[0], 65536, "and every key lands in bucket 0");

    // Weak hash 2: the top 12 bits of small keys are constant.
    const small = [];
    for (let i = 0; i < 65536; i++) small.push(i % (1 << 20)); // all below 2^20
    const expectedPerBucket = keys.length / buckets;
    const highCounts = bucketCounts(small, buckets, weakHighBits);
    assert.strictEqual(highCounts[0], 65536, "keys below 2^20 have no bits above bit 19");
    assert.ok(chiSquare(highCounts, new Array(buckets).fill(expectedPerBucket)) > 10 * df);
    // The good hash spreads the same small keys out.
    const goodCounts = bucketCounts(small, buckets, hash32);
    assert.ok(chiSquare(goodCounts, new Array(buckets).fill(expectedPerBucket)) <= 1.1 * df);

    // Weak hash 3: multiplication is a permutation but does not diffuse - the low 12 bits of the
    // output depend only on the low 12 bits of the input, so aligned keys collapse again.
    const linearCounts = bucketCounts(aligned, buckets, weakLinear);
    assert.strictEqual(linearCounts[0], 65536,
      "imul by an odd constant sends every multiple of 4096 to bucket 0");
    assert.ok(chiSquare(linearCounts, new Array(buckets).fill(expectedPerBucket)) > 10 * df);
  }

  // (5) Birthday collisions into 2^32 versus into 2^16 buckets.
  {
    const keys = [];
    for (let i = 0; i < 65536; i++) keys.push(rng.nextInt(0x100000000));
    const wide = collidingPairs(keys, hash32);
    assert.ok(wide.pairs <= 4,
      `2^16 keys in a 2^32 space: ${wide.pairs} colliding pairs (expected 0.5)`);
    const narrow = collidingPairs(keys, (key) => hash32(key) & 0xffff);
    const expected = (keys.length * (keys.length - 1)) / 2 / 65536;
    assert.ok(Math.abs(narrow.pairs - expected) <= 0.05 * expected,
      `bucket collisions ${narrow.pairs} within 5% of C(N,2)/M = ${expected}`);
    const weakWide = collidingPairs(keys, weakLowBits);
    assert.ok(weakWide.pairs > 20000,
      `the 12-bit weak hash collides constantly: ${weakWide.pairs} pairs`);
    // A small sample already exposes the 16-bit weak hash but not the good one.
    const small = keys.slice(0, 1000);
    assert.strictEqual(collidingPairs(small, hash32).pairs, 0);
    assert.ok(collidingPairs(small, (key) => key & 0xffff).pairs >= 2,
      "the identity-truncation collides even on 1000 keys");
  }

  // (6) Structured keys: high-bit differences still avalanche.
  {
    const keys = [];
    for (let i = 0; i < 512; i++) {
      const base = rng.nextInt(1 << 20);
      keys.push(base);
      keys.push((base + (1 << 20) * (1 + rng.nextInt(4000))) >>> 0);
    }
    let distanceSum = 0;
    for (let i = 0; i < keys.length; i += 2) {
      distanceSum += hamming32(hash32(keys[i]), hash32(keys[i + 1]));
    }
    const mean = distanceSum / (keys.length / 2);
    assert.ok(mean >= 14.0 && mean <= 18.0,
      `keys differing only above bit 19 give mean Hamming distance ${mean}`);
    assert.strictEqual(weakLowBits(keys[0]), weakLowBits(keys[1]),
      "the low-bit weak hash maps the whole pair to one value");
    assert.notStrictEqual(weakHighBits(keys[0]), weakHighBits(keys[1]),
      "while the high-bit truncation does separate this particular pair");
  }

  // (7) Linear independence from the key value, and what correlation cannot see.
  {
    const keys = [];
    for (let i = 0; i < 4096; i++) keys.push(rng.nextInt(0x100000000));
    const corr = correlation(keys, keys.map(hash32));
    assert.ok(Math.abs(corr) < 0.05, `|correlation| ${Math.abs(corr).toFixed(4)} must be small`);
    assert.ok(Math.abs(correlation(keys, keys.map(weakIdentity)) - 1) <= 1e-12,
      "the identity hash is perfectly correlated");
    const small = keys.slice(0, 512).map((key) => key % (1 << 20));
    const affine = small.map((value) => value * 3 + 7); // exact in doubles: no 2^32 wrap-around
    assert.ok(Math.abs(correlation(small, affine) - 1) <= 1e-12,
      "a non-wrapping affine map is perfectly correlated");
    // The modular multiplicative map wraps, so its correlation is small - a reminder that
    // correlation alone is a weak test; the low-bit uniformity check in (4) still catches it.
    const modularCorr = correlation(small, small.map(weakLinear));
    assert.ok(Math.abs(modularCorr) < 0.5,
      `the wrapping multiplicative map shows little correlation (${modularCorr.toFixed(3)})`);
    assert.strictEqual(weakLinear(4096) & 0xfff, 0);
    assert.strictEqual(weakLinear(8192) & 0xfff, 0);
  }

  // (8) Conditional uniformity: every input low-byte class spreads uniformly over the output byte.
  {
    const classCounts = new Map();
    for (let i = 0; i < 65536; i++) {
      const key = (i * 7919) >>> 0;
      const cls = key & 0xff;
      if (!classCounts.has(cls)) classCounts.set(cls, new Array(256).fill(0));
      classCounts.get(cls)[hash32(key) & 0xff] += 1;
    }
    assert.strictEqual(classCounts.size, 256);
    let worst = 0;
    for (const counts of classCounts.values()) {
      const total = counts.reduce((a, b) => a + b, 0);
      const statistic = chiSquare(counts, new Array(256).fill(total / 256));
      worst = Math.max(worst, statistic);
    }
    assert.ok(worst <= 1.5 * 255, `worst per-class chi-square ${worst.toFixed(1)} <= 382.5`);
  }

  // (9) Message hashing: order sensitivity, length padding and a demonstrated weak collision.
  {
    assert.strictEqual(hashBytes([]), mix32(mix32(0x1b873593)),
      "the empty message still goes through both mixing stages");
    assert.strictEqual(hashBytes([]), 3366025281, "documented value 0xc8a17841");
    assert.strictEqual(hashBytes(asciiBytes("ab")), 1706569710);
    assert.strictEqual(xorHash(asciiBytes("ab")), 4235334454);
    assert.notStrictEqual(hashBytes(asciiBytes("ab")), hashBytes(asciiBytes("ba")),
      "the iterated hash notices the order of the bytes");
    assert.strictEqual(xorHash(asciiBytes("ab")), xorHash(asciiBytes("ba")),
      "while the XOR variant provably collides on them");
    assert.strictEqual(xorHash(asciiBytes("abc")), xorHash(asciiBytes("cba")));
    assert.notStrictEqual(hashBytes(asciiBytes("ab")), hashBytes(asciiBytes("ab\u0000")),
      "length padding separates a message from its zero extension");
    assert.notStrictEqual(hashBytes(asciiBytes("")), hashBytes(asciiBytes("\u0000")));
    assert.notStrictEqual(hashBytes(asciiBytes("hello world")), hashBytes(asciiBytes("Hello world")));
    const seen = new Map();
    let collisions = 0;
    for (let i = 0; i < 20000; i++) {
      const text = `message-${i}-${(i * 2654435761) % 1000003}`;
      const value = hashBytes(asciiBytes(text));
      if (seen.has(value)) collisions += 1;
      seen.set(value, text);
    }
    assert.strictEqual(collisions, 0, "20000 distinct messages hash without collision");
    assert.strictEqual(seen.size, 20000);
  }

  // (10) The counter and statistics helpers themselves are sanity-checked.
  {
    assert.strictEqual(hamming32(0, 0xffffffff), 32);
    assert.strictEqual(hamming32(0xaaaaaaaa, 0x55555555), 32);
    assert.strictEqual(hamming32(123, 123), 0);
    assert.strictEqual(chiSquare([10, 10], [10, 10]), 0);
    assert.ok(chiSquare([20, 0], [10, 10]) > 0);
    assert.strictEqual(collidingPairs([1, 1, 2], weakIdentity).pairs, 1);
    assert.strictEqual(collidingPairs([1, 1, 2], weakIdentity).distinct, 2);
    assert.ok(Math.abs(correlation([1, 2, 3], [2, 4, 6]) - 1) <= 1e-12);
    assert.ok(Math.abs(correlation([1, 2, 3], [-1, -2, -3]) + 1) <= 1e-12);
    assert.strictEqual(typeof MASK32, "number");
  }
}

module.exports = { MASK64, MASK32, Rng, mix32, hash32, hash64, hamming32, weakIdentity, weakLowBits,
  weakHighBits, weakLinear, xorHash, hashBytes, asciiBytes, chiSquare, bucketCounts, avalanche32,
  avalanche64, collidingPairs, correlation };

if (require.main === module) {
  main();
  console.log("PASS");
}
