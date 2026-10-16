/** Task 652 - Implement bitmap index and fast AND/OR operations. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * A bitmap index stores one bitset per distinct column value; row id i is a member of the
 * bitmap of `column[i]`. Predicate evaluation becomes word-parallel set algebra: an
 * equality predicate is one bitmap lookup, a conjunction is a word-wise AND, a disjunction
 * is a word-wise OR, and NOT is a complement inside the row universe.
 *
 * Two representations implement the same algebra and cross-check each other:
 *   - Bitmap  : Uint32Array words, the production representation, with an operation counter
 *               proving the algebra costs O(words) per input instead of O(rows);
 *   - BigInt  : the same bits packed in one arbitrary-precision integer, used as an
 *               independent oracle for AND / OR / XOR / ANDNOT / NOT and popcount.
 *
 * Invariant verified: for every predicate the bitmap answer equals a brute-force row scan,
 * and for every generated bitset pair the word representation equals the BigInt oracle.
 */
"use strict";
const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");
/** Deterministic xorshift32; returns integers in [0, bound). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x2545f491;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
function popcountWord(x) {
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return (Math.imul(x, 0x01010101) >>> 24) & 0x3f;
}
class Bitmap {
  constructor(bitCount) {
    this.bitCount = bitCount | 0;
    this.words = new Uint32Array((this.bitCount + 31) >>> 5);
    this.wordOps = 0; // number of 32-bit word combinations performed
  }
  static fromIndices(bitCount, indices) {
    const bm = new Bitmap(bitCount);
    for (const i of indices) bm.set(i);
    return bm;
  }
  set(i) {
    if (i < 0 || i >= this.bitCount) throw new RangeError(`bit ${i} outside universe ${this.bitCount}`);
    this.words[i >>> 5] |= 1 << (i & 31);
    return this;
  }
  clear(i) {
    if (i >= 0 && i < this.bitCount) this.words[i >>> 5] &= ~(1 << (i & 31));
    return this;
  }
  get(i) {
    if (i < 0 || i >= this.bitCount) return false;
    return (this.words[i >>> 5] & (1 << (i & 31))) !== 0;
  }
  clone() {
    const bm = new Bitmap(this.bitCount);
    bm.words.set(this.words);
    return bm;
  }
  /** Lowest set bit index >= from, or -1. Uses the de Bruijn-free clz32 trick. */
  nextSetBit(from) {
    if (from < 0) from = 0;
    for (let w = from >>> 5; w < this.words.length; w++) {
      let word = this.words[w];
      if (w === from >>> 5) word &= 0xffffffff << (from & 31);
      if (word !== 0) {
        const lowest = (word & -word) >>> 0;
        return (w << 5) + (31 - Math.clz32(lowest));
      }
    }
    return -1;
  }
  popcount() {
    let total = 0;
    for (let w = 0; w < this.words.length; w++) total += popcountWord(this.words[w]);
    return total;
  }
  static combine(a, b, op) {
    const size = Math.max(a.bitCount, b.bitCount);
    const out = new Bitmap(size);
    const n = out.words.length;
    for (let w = 0; w < n; w++) {
      const x = w < a.words.length ? a.words[w] : 0;
      const y = w < b.words.length ? b.words[w] : 0;
      out.words[w] = op(x, y);
    }
    out.wordOps = a.wordOps + b.wordOps + n;
    return out;
  }
  and(other) {
    return Bitmap.combine(this, other, (x, y) => (x & y) >>> 0);
  }
  or(other) {
    return Bitmap.combine(this, other, (x, y) => (x | y) >>> 0);
  }
  xor(other) {
    return Bitmap.combine(this, other, (x, y) => (x ^ y) >>> 0);
  }
  andNot(other) {
    return Bitmap.combine(this, other, (x, y) => (x & ~y) >>> 0);
  }
  /** Complement inside the row universe [0, bitCount). */
  not() {
    const out = new Bitmap(this.bitCount);
    for (let w = 0; w < this.words.length; w++) out.words[w] = ~this.words[w] >>> 0;
    if (this.bitCount % 32 !== 0 && this.words.length > 0) {
      const mask = (1 << (this.bitCount % 32)) - 1;
      out.words[this.words.length - 1] &= mask >>> 0;
    }
    out.wordOps = this.wordOps + this.words.length;
    return out;
  }
  isEmpty() {
    for (let w = 0; w < this.words.length; w++) if (this.words[w] !== 0) return false;
    return true;
  }
  equals(other) {
    return this.bitCount === other.bitCount && this.toBigInt() === other.toBigInt();
  }
  toIndices() {
    const out = [];
    for (let i = this.nextSetBit(0); i !== -1; i = this.nextSetBit(i + 1)) out.push(i);
    return out;
  }
  /** Oracle representation: the whole bitmap as one BigInt (bit 0 = row 0). */
  toBigInt() {
    let acc = 0n;
    for (let w = this.words.length - 1; w >= 0; w--) acc = (acc << 32n) | BigInt(this.words[w] >>> 0);
    return this.bitCount === 0 ? 0n : acc;
  }
  static fromBigInt(bitCount, value) {
    const bm = new Bitmap(bitCount);
    let rest = value < 0n ? value + (1n << BigInt(bitCount)) : value;
    for (let w = 0; w < bm.words.length && rest > 0n; w++) {
      bm.words[w] = Number(rest & 0xffffffffn) >>> 0;
      rest >>= 32n;
    }
    return bm;
  }
  static andAll(bitmaps) {
    if (bitmaps.length === 0) throw new RangeError("andAll needs at least one bitmap");
    let acc = bitmaps[0];
    for (let i = 1; i < bitmaps.length; i++) acc = acc.and(bitmaps[i]);
    return acc;
  }
  static orAll(bitmaps, bitCount) {
    let acc = new Bitmap(bitmaps.length ? bitmaps[0].bitCount : bitCount);
    for (const bm of bitmaps) acc = acc.or(bm);
    return acc;
  }
}
class BitmapIndex {
  constructor(values) {
    this.values = values.slice();
    this.bitCount = values.length;
    this.postings = new Map();
    for (let row = 0; row < values.length; row++) {
      const key = values[row];
      let bm = this.postings.get(key);
      if (bm === undefined) {
        bm = new Bitmap(this.bitCount);
        this.postings.set(key, bm);
      }
      bm.set(row);
    }
    this.distinct = this.postings.size;
  }
  /** Equality predicate. */
  equals(value) {
    const bm = this.postings.get(value);
    return bm === undefined ? new Bitmap(this.bitCount) : bm;
  }
  /** IN-list predicate (OR over the listed values). */
  anyOf(values) {
    const parts = [];
    for (const v of values) {
      const bm = this.postings.get(v);
      if (bm !== undefined) parts.push(bm);
    }
    return Bitmap.orAll(parts, this.bitCount);
  }
  /** Numeric range predicate implemented as an IN-list over the numeric keys. */
  numericRange(low, high) {
    const parts = [];
    for (const [key, bm] of this.sortedKeys()) {
      if (key >= low && key <= high) parts.push(bm);
    }
    return Bitmap.orAll(parts, this.bitCount);
  }
  sortedKeys() {
    return [...this.postings.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  }
  /** Conjunction of several equality predicates. */
  andEquals(values) {
    const parts = values.map((v) => this.equals(v));
    return Bitmap.andAll(parts);
  }
  complement(bm) {
    return bm.not();
  }
}
function bruteForceMatch(rows, predicate) {
  const out = [];
  for (let i = 0; i < rows.length; i++) if (predicate(rows[i])) out.push(i);
  return out;
}
module.exports = { Bitmap, BitmapIndex, popcountWord, makeRng, bruteForceMatch };
if (require.main === module) {
  // -- word-boundary behaviour ------------------------------------------------------
  const bm = new Bitmap(70);
  for (const i of [0, 31, 32, 33, 63, 64, 69]) bm.set(i);
  strictEqual(bm.popcount(), 7);
  deepStrictEqual(bm.toIndices(), [0, 31, 32, 33, 63, 64, 69]);
  ok(bm.get(31) && bm.get(32) && bm.get(69));
  ok(!bm.get(30) && !bm.get(70) && !bm.get(-1), "out-of-universe bits read as 0");
  throws(() => bm.set(70), RangeError, "setting a bit outside the universe must throw");
  // -- empty / singleton universes --------------------------------------------------
  strictEqual(new Bitmap(0).popcount(), 0);
  strictEqual(new Bitmap(0).toBigInt(), 0n);
  deepStrictEqual(new Bitmap(1).toIndices(), []);
  ok(new Bitmap(1).not().isEmpty() === false);
  ok(new Bitmap(1).not().get(0), "complement of the empty universe is full");
  // -- BigInt oracle cross-check ----------------------------------------------------
  const rng = makeRng(0xb17b17);
  let oracleChecked = 0;
  for (let trial = 0; trial < 60; trial++) {
    const bits = 33 + rng(64);
    const mask = (1n << BigInt(bits)) - 1n;
    const ia = [], ib = [];
    for (let i = 0; i < bits; i++) {
      if (rng(100) < 50) ia.push(i);
      if (rng(100) < 50) ib.push(i);
    }
    const a = Bitmap.fromIndices(bits, ia);
    const b = Bitmap.fromIndices(bits, ib);
    const ba = a.toBigInt();
    const bb = b.toBigInt();
    strictEqual(a.and(b).toBigInt(), (ba & bb) & mask, "AND must match BigInt oracle");
    strictEqual(a.or(b).toBigInt(), (ba | bb) & mask, "OR must match BigInt oracle");
    strictEqual(a.xor(b).toBigInt(), (ba ^ bb) & mask, "XOR must match BigInt oracle");
    strictEqual(a.andNot(b).toBigInt(), (ba & ~bb) & mask, "ANDNOT must match BigInt oracle");
    strictEqual(a.not().toBigInt(), (~ba) & mask, "NOT must match BigInt oracle");
    strictEqual(a.popcount(), a.toIndices().length, "popcount must equal enumerated size");
    strictEqual(Bitmap.fromBigInt(bits, ba).toBigInt(), ba, "BigInt round-trip");
    oracleChecked++;
  }
  strictEqual(oracleChecked, 60);
  // -- algebraic identities ---------------------------------------------------------
  const u = Bitmap.fromIndices(50, Array.from({ length: 12 }, (_, i) => i * 4));
  const v = Bitmap.fromIndices(50, Array.from({ length: 9 }, (_, i) => i * 5));
  ok(u.and(u).equals(u) && u.or(u).equals(u), "idempotence");
  ok(u.and(v).equals(v.and(u)) && u.or(v).equals(v.or(u)), "commutativity");
  ok(u.andNot(u).isEmpty(), "a ANDNOT a is empty");
  ok(u.or(new Bitmap(50)).equals(u), "empty is the OR identity");
  ok(u.and(new Bitmap(50)).isEmpty(), "empty is the AND annihilator");
  ok(u.not().not().equals(u), "double complement is the identity");
  // De Morgan on the shared universe.
  ok(u.not().and(v.not()).equals(u.or(v).not()), "NOT(a OR b) == NOT a AND NOT b");
  ok(u.not().or(v.not()).equals(u.and(v).not()), "NOT(a AND b) == NOT a OR NOT b");
  // -- index predicates vs brute force ----------------------------------------------
  const city = ["oslo", "lima", "oslo", "kyiv", "lima", "oslo", "kyiv", "lima", "oslo", "kyiv"];
  const score = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
  const rows = city.map((c, i) => ({ city: c, score: score[i] }));
  const cityIdx = new BitmapIndex(city);
  const scoreIdx = new BitmapIndex(score);
  strictEqual(cityIdx.distinct, 3);
  strictEqual(cityIdx.bitCount, 10);
  deepStrictEqual(cityIdx.equals("oslo").toIndices(), bruteForceMatch(rows, (r) => r.city === "oslo"));
  deepStrictEqual(cityIdx.equals("rome").toIndices(), [], "unknown key yields the empty bitmap");
  deepStrictEqual(cityIdx.anyOf(["kyiv", "lima"]).toIndices(),
    bruteForceMatch(rows, (r) => r.city === "kyiv" || r.city === "lima"));
  deepStrictEqual(cityIdx.anyOf([]).toIndices(), [], "empty IN-list matches nothing");
  deepStrictEqual(scoreIdx.numericRange(35, 85).toIndices(),
    bruteForceMatch(rows, (r) => r.score >= 35 && r.score <= 85));
  deepStrictEqual(scoreIdx.numericRange(200, 300).toIndices(), [], "range above the column max");
  const conjunction = cityIdx.equals("oslo").and(scoreIdx.numericRange(0, 55));
  deepStrictEqual(conjunction.toIndices(),
    bruteForceMatch(rows, (r) => r.city === "oslo" && r.score <= 55));
  deepStrictEqual(cityIdx.complement(cityIdx.equals("oslo")).toIndices(),
    bruteForceMatch(rows, (r) => r.city !== "oslo"));
  // Interval semantics: [30, 30] AND [40, 100] must be empty (disjoint ranges).
  ok(scoreIdx.numericRange(30, 30).and(scoreIdx.numericRange(40, 100)).isEmpty());
  // -- randomised predicate cross-check on a wider table ----------------------------
  const prng = makeRng(90210);
  const bigCity = Array.from({ length: 512 }, () => `c${prng(7)}`);
  const bigScore = Array.from({ length: 512 }, () => prng(40));
  const bCity = new BitmapIndex(bigCity);
  const bScore = new BitmapIndex(bigScore);
  const bigRows = bigCity.map((c, i) => ({ city: c, score: bigScore[i] }));
  for (let t = 0; t < 40; t++) {
    const pick = [`c${prng(7)}`, `c${prng(7)}`];
    const lo = prng(40), hi = lo + prng(20);
    const got = bCity.anyOf(pick).and(bScore.numericRange(lo, hi)).toIndices();
    const want = bruteForceMatch(bigRows,
      (r) => (r.city === pick[0] || r.city === pick[1]) && r.score >= lo && r.score <= hi);
    deepStrictEqual(got, want, `random predicate ${t}`);
  }
  // -- cost model: word-parallel AND is O(words), not O(rows) -----------------------
  const n = 4096;
  const words = (n + 31) >>> 5;
  const predicates = Array.from({ length: 32 }, (_, k) => Bitmap.fromIndices(n,
    Array.from({ length: n / 8 }, (_, i) => (i * 7 + k) % n)));
  const folded = Bitmap.andAll(predicates);
  const rowComparisons = n * (predicates.length - 1);
  ok(folded.wordOps <= predicates.length * words + words,
    `AND must touch ~${predicates.length * words} words, saw ${folded.wordOps}`);
  ok(folded.wordOps * 16 < rowComparisons,
    `word-parallel AND must be at least 16x cheaper than ${rowComparisons} row comparisons`);
  deepStrictEqual(folded.toIndices(),
    bruteForceMatch(Array.from({ length: n }, (_, i) => i),
      (i) => predicates.every((bm) => bm.get(i))),
    "32-way AND must agree with the row-wise filter");
  console.log(`bitmaps=60x2 oracle checks, words=${words}, andWordOps=${folded.wordOps}, rowComparisons=${rowComparisons}`);
  console.log("PASS");
}
