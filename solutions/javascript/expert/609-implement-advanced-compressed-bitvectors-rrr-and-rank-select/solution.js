/** Task 609 - Implement advanced compressed bitvectors (RRR) and rank/select. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Raman-Raman-Rao block coding. The bitvector is cut into fixed blocks of BLOCK_BITS bits.
 * A block holding c ones is stored as its *class* c plus its *offset*: the colex rank of the
 * c positions inside the block. Colex (the combinatorial number system) is a bijection
 * [0, C(b,c)) <-> c-subsets of a b-set, so a block costs ceil(log2 C(b,c)) bits instead of
 * b. That is the "R" of RRR: the offset is coded combinatorially, the information-theoretic
 * optimum for a fixed class, so blocks with few (or nearly all) ones get much cheaper.
 * The class of every block lives in a unary class bitvector (c ones then a zero): the number
 * of ones before block b is recovered from it, so rank1(i) = superblock sample + the classes
 * of the blocks inside the superblock + the ones of the partial block decoded with colex
 * unrank. select1(j) descends the same way and then walks the decoded block.
 *
 * What is simplified: the class bitvector is stored plainly instead of being RRR-compressed
 * recursively (a full RRR recurses on it until it fits in a machine word) and the ones
 * samples are kept every SUPERBLOCK_BLOCKS blocks. What is still verified: the colex
 * bijection exhaustively over all 2^16 blocks, bitwise equality of the rebuilt vector with
 * the input at every position, rank1/rank0/select1/select0 against a naive array oracle at
 * every index, the rank/select inverse identities, input validation, and the compression
 * claim (offset bits <= n always, total < n for skewed vectors; dense vectors are not the
 * regime RRR wins in and the diagnostics print that honestly).
 */
"use strict";

const BLOCK_BITS = 16;
const SUPERBLOCK_BLOCKS = 8;

/** Binomial table C(n, k) for 0 <= n, k <= BLOCK_BITS, i.e. 0 whenever k > n. */
const BINOM = (() => {
  const table = [];
  for (let n = 0; n <= BLOCK_BITS; n++) {
    const row = new Array(BLOCK_BITS + 1).fill(0);
    row[0] = 1;
    for (let k = 1; k <= n; k++) row[k] = table[n - 1][k - 1] + table[n - 1][k];
    table.push(row);
  }
  return table;
})();

/** Bits needed for the offset of a block of class c: ceil(log2 C(b, c)), computed exactly. */
const OFFSET_BITS = (() => {
  const bits = new Array(BLOCK_BITS + 1).fill(0);
  for (let c = 0; c <= BLOCK_BITS; c++) {
    const combinations = BINOM[BLOCK_BITS][c];
    let k = 0;
    while ((1 << k) < combinations) k++;   // integer loop: no float log2 rounding traps
    bits[c] = k;
  }
  return bits;
})();

/** Colex rank of an increasing list of 0-based block positions: sum_i C(p_i, i + 1). */
function colexRank(positions) {
  let rank = 0;
  for (let i = 0; i < positions.length; i++) rank += BINOM[positions[i]][i + 1];
  return rank;
}

/** Inverse of colexRank: greedy unpacking from the highest index down. */
function colexUnrank(rank, count) {
  const positions = new Array(count);
  let rest = rank;
  for (let i = count; i >= 1; i--) {
    let chosen = i - 1;
    for (let candidate = BLOCK_BITS - 1; candidate >= i - 1; candidate--) {
      if (BINOM[candidate][i] <= rest) { chosen = candidate; break; }
    }
    positions[i - 1] = chosen;
    rest -= BINOM[chosen][i];
  }
  return positions;
}

class RRRBitvector {
  constructor(bits) {
    const source = RRRBitvector.#asBits(bits);
    this.n = source.length;
    this.classes = [];
    this.offsets = [];
    this.superSamples = [];
    const blockCount = Math.ceil(this.n / BLOCK_BITS);
    let onesSoFar = 0;
    for (let b = 0; b < blockCount; b++) {
      if (b % SUPERBLOCK_BLOCKS === 0) this.superSamples.push(onesSoFar);
      const limit = Math.min(BLOCK_BITS, this.n - b * BLOCK_BITS);
      const ones = [];
      for (let i = 0; i < limit; i++) if (source[b * BLOCK_BITS + i]) ones.push(i);
      this.classes.push(ones.length);
      this.offsets.push(colexRank(ones));
      onesSoFar += ones.length;
    }
    this.ones = onesSoFar;
    this.#cache = new Map();
  }

  #cache;

  static #asBits(bits) {
    if (typeof bits === "string") {
      const out = new Array(bits.length);
      for (let i = 0; i < bits.length; i++) {
        if (bits[i] !== "0" && bits[i] !== "1") {
          throw new TypeError(`bitvector string must contain only '0'/'1' (got '${bits[i]}')`);
        }
        out[i] = bits[i] === "1" ? 1 : 0;
      }
      return out;
    }
    if (Array.isArray(bits) || ArrayBuffer.isView(bits)) {
      return Array.from(bits, (v) => {
        if (v !== 0 && v !== 1 && v !== false && v !== true) {
          throw new TypeError(`bitvector element must be 0/1 (got ${v})`);
        }
        return v ? 1 : 0;
      });
    }
    throw new TypeError("bitvector must be a string of 0/1 or an array of 0/1");
  }

  get length() { return this.n; }
  get blockCount() { return this.classes.length; }
  get zeroes() { return this.n - this.ones; }

  /** Ones positions (0-based, relative to the block) of block b, decoded through colex unrank. */
  blockOnes(b) {
    let decoded = this.#cache.get(b);
    if (decoded === undefined) {
      decoded = colexUnrank(this.offsets[b], this.classes[b]);
      this.#cache.set(b, decoded);
    }
    return decoded;
  }

  onesBeforeBlock(b) {
    const superIndex = Math.floor(b / SUPERBLOCK_BLOCKS);
    let total = this.superSamples[superIndex];
    for (let i = superIndex * SUPERBLOCK_BLOCKS; i < b; i++) total += this.classes[i];
    return total;
  }

  onesInBlockRange(b, from, to) {
    let count = 0;
    for (const pos of this.blockOnes(b)) if (pos >= from && pos <= to) count++;
    return count;
  }

  bitAt(i) {
    if (i < 0 || i >= this.n) throw new RangeError(`bit index ${i} out of range`);
    const b = Math.floor(i / BLOCK_BITS);
    return this.blockOnes(b).includes(i - b * BLOCK_BITS) ? 1 : 0;
  }

  /** Number of ones in positions 0..i; rank1(-1) === 0 by convention. */
  rank1(i) {
    if (i < -1 || i > this.n - 1) throw new RangeError(`rank index ${i} out of range`);
    if (i < 0) return 0;
    const b = Math.floor(i / BLOCK_BITS);
    return this.onesBeforeBlock(b) + this.onesInBlockRange(b, 0, i - b * BLOCK_BITS);
  }

  rank0(i) {
    if (i < -1 || i > this.n - 1) throw new RangeError(`rank index ${i} out of range`);
    if (i < 0) return 0;
    return i + 1 - this.rank1(i);
  }

  /** Position (0-based) of the (j + 1)-th one. */
  select1(j) {
    if (j < 0 || j >= this.ones) throw new RangeError(`select1(${j}) out of range`);
    let b = 0;
    for (let s = this.superSamples.length - 1; s >= 0; s--) {
      if (this.superSamples[s] <= j) { b = s * SUPERBLOCK_BLOCKS; break; }
    }
    let remaining = j - this.onesBeforeBlock(b);
    while (remaining >= this.classes[b]) { remaining -= this.classes[b]; b++; }
    return b * BLOCK_BITS + this.blockOnes(b)[remaining];
  }

  /** Position (0-based) of the (j + 1)-th zero. */
  select0(j) {
    if (j < 0 || j >= this.zeroes) throw new RangeError(`select0(${j}) out of range`);
    let seen = 0;
    for (let b = 0; b < this.blockCount; b++) {
      const limit = Math.min(BLOCK_BITS, this.n - b * BLOCK_BITS);
      const zeroes = limit - this.classes[b];
      if (j < seen + zeroes) {
        const ones = this.blockOnes(b);
        let missing = j - seen;
        for (let i = 0; i < limit; i++) {
          if (!ones.includes(i) && missing-- === 0) return b * BLOCK_BITS + i;
        }
      }
      seen += zeroes;
    }
    throw new RangeError(`select0(${j}) out of range`);
  }

  /** Rebuild the plain bit vector from the encoded blocks (uses unrank only). */
  toDigits() {
    const out = new Array(this.n).fill("0");
    for (let b = 0; b < this.blockCount; b++) {
      for (const pos of this.blockOnes(b)) out[b * BLOCK_BITS + pos] = "1";
    }
    return out.join("");
  }

  offsetBits() {
    let bits = 0;
    for (const c of this.classes) bits += OFFSET_BITS[c];
    return bits;
  }

  classVectorBits() { return this.ones + this.blockCount; }   // unary: c ones + one zero

  sampleBits() {
    const perSample = Math.max(1, Math.ceil(Math.log2(this.n + 1)));
    return this.superSamples.length * perSample;
  }

  compressedBits() { return this.offsetBits() + this.classVectorBits() + this.sampleBits(); }

  /** Shannon-flavoured lower bound: sum over blocks of log2 C(b, class). */
  entropyBits() {
    let bits = 0;
    for (const c of this.classes) bits += Math.log2(Math.max(1, BINOM[BLOCK_BITS][c]));
    return bits;
  }
}

/** Naive array oracle used to pin down rank/select. */
class NaiveBitvector {
  constructor(bits) {
    this.bits = typeof bits === "string" ? Array.from(bits, (c) => (c === "1" ? 1 : 0)) : Array.from(bits, Number);
    this.n = this.bits.length;
  }

  rank1(i) { let r = 0; for (let k = 0; k <= i; k++) r += this.bits[k]; return r; }
  rank0(i) { let r = 0; for (let k = 0; k <= i; k++) r += this.bits[k] === 0 ? 1 : 0; return r; }
  select1(j) { let seen = 0; for (let i = 0; i < this.n; i++) if (this.bits[i] === 1 && seen++ === j) return i; return -1; }
  select0(j) { let seen = 0; for (let i = 0; i < this.n; i++) if (this.bits[i] === 0 && seen++ === j) return i; return -1; }
}

function xorshift32(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= (state << 13) >>> 0; state >>>= 0;
    state ^= state >>> 17;
    state ^= (state << 5) >>> 0; state >>>= 0;
    return state / 4294967296;
  };
}

function randomBits(n, density, seed) {
  const next = xorshift32(seed);
  let out = "";
  for (let i = 0; i < n; i++) out += next() < density ? "1" : "0";
  return out;
}

function popcount16(mask) {
  let count = 0;
  while (mask) { mask &= mask - 1; count++; }
  return count;
}

module.exports = { RRRBitvector, NaiveBitvector, colexRank, colexUnrank, BINOM, OFFSET_BITS, BLOCK_BITS };

if (require.main === module) {
  const assert = require("node:assert");

  // Exhaustive check of the colex bijection over every possible 16-bit block.
  for (let c = 0; c <= BLOCK_BITS; c++) {
    const seen = new Set();
    for (let mask = 0; mask < (1 << BLOCK_BITS); mask++) {
      if (popcount16(mask) !== c) continue;
      const positions = [];
      for (let i = 0; i < BLOCK_BITS; i++) if (mask & (1 << i)) positions.push(i);
      const rank = colexRank(positions);
      assert.ok(rank >= 0 && rank < BINOM[BLOCK_BITS][c], `colex rank ${rank} out of class range`);
      assert.ok(!seen.has(rank), `colex rank ${rank} duplicated in class ${c}`);
      seen.add(rank);
      assert.deepStrictEqual(colexUnrank(rank, c), positions, `colex unrank mismatch class ${c}`);
    }
    assert.strictEqual(seen.size, BINOM[BLOCK_BITS][c], `class ${c} did not cover all subsets`);
  }

  const cases = [
    ["", 0], ["0", 0], ["1", 1],
    ["1010", 2], ["0110", 2],
    ["1".repeat(BLOCK_BITS), BLOCK_BITS],
    ["0".repeat(BLOCK_BITS), 0],
    ["1".repeat(BLOCK_BITS) + "1", BLOCK_BITS + 1],
    ["1".repeat(2 * BLOCK_BITS + 5), 2 * BLOCK_BITS + 5],
    ["1" + "0".repeat(30) + "1", 2],
    ["01".repeat(20), 20],
    ["0".repeat(13) + "111", 3],
  ];

  for (const [text, ones] of cases) {
    const vector = new RRRBitvector(text);
    assert.strictEqual(vector.length, text.length, "length");
    assert.strictEqual(vector.ones, ones, `ones of ${text}`);
    assert.strictEqual(vector.toDigits(), text, `round-trip of ${text}`);
    const naive = new NaiveBitvector(text);
    for (let i = -1; i < text.length; i++) {
      assert.strictEqual(vector.rank1(i), naive.rank1(i), `rank1(${i}) of ${text}`);
      assert.strictEqual(vector.rank0(i), naive.rank0(i), `rank0(${i}) of ${text}`);
    }
    for (let j = 0; j < ones; j++) assert.strictEqual(vector.select1(j), naive.select1(j), `select1(${j}) of ${text}`);
    for (let j = 0; j < text.length - ones; j++) assert.strictEqual(vector.select0(j), naive.select0(j), `select0(${j}) of ${text}`);
    assert.strictEqual(vector.rank1(text.length - 1), ones, "total ones");
  }

  const sparse = ["2 ones per 16-bit block", randomBits(512, 0.125, 0x1234)];
  const skewed = ["1 one per block", ("1" + "0".repeat(15)).repeat(32)];
  const dense = ["density 1/2", randomBits(512, 0.5, 0xbeef)];
  const periodic = ["period 8 (01010101...)", "01010101".repeat(64)];
  const allOnes = ["all ones", "1".repeat(256)];

  for (const [label, text] of [sparse, skewed, dense, periodic, allOnes]) {
    const vector = new RRRBitvector(text);
    const naive = new NaiveBitvector(text);
    assert.strictEqual(vector.toDigits(), text, `round-trip (${label})`);
    for (let i = -1; i < text.length; i++) {
      assert.strictEqual(vector.rank1(i), naive.rank1(i), `rank1 (${label})`);
      assert.strictEqual(vector.rank0(i), naive.rank0(i), `rank0 (${label})`);
    }
    for (let j = 0; j < vector.ones; j++) assert.strictEqual(vector.select1(j), naive.select1(j), `select1 (${label}) j=${j}`);
    for (let j = 0; j < vector.zeroes; j++) assert.strictEqual(vector.select0(j), naive.select0(j), `select0 (${label}) j=${j}`);
    // rank/select are inverse to each other on the whole dictionary
    for (let i = 0; i < text.length; i++) {
      const rank = vector.rank1(i);
      if (rank > 0) assert.ok(vector.select1(rank - 1) <= i, `select1(rank1(${i})-1) <= ${i} (${label})`);
      if (vector.bitAt(i) === 1) assert.strictEqual(vector.rank1(vector.select1(rank - 1)), rank, `inverse (${label})`);
    }
    assert.ok(vector.offsetBits() <= text.length, `offset bits must never exceed n (${label})`);
    assert.ok(vector.compressedBits() <= 1.7 * text.length, `total bits bounded (${label})`);
    console.log(
      `${label}: n=${text.length} ones=${vector.ones} offset=${vector.offsetBits()} ` +
      `class=${vector.classVectorBits()} total=${vector.compressedBits()} ratio=${(vector.compressedBits() / text.length).toFixed(3)}`
    );
  }

  const sparseVector = new RRRBitvector(sparse[1]);
  assert.ok(sparseVector.compressedBits() < sparse[1].length * 0.85, "sparse vector must compress");
  const skewedVector = new RRRBitvector(skewed[1]);
  assert.ok(skewedVector.compressedBits() < skewed[1].length * 0.6, "one-one-per-block vector must compress well");
  const allOnesVector = new RRRBitvector(allOnes[1]);
  assert.strictEqual(allOnesVector.offsetBits(), 0, "uniform blocks need zero offset bits");

  // Edge cases: empty, single element, partial trailing block.
  const empty = new RRRBitvector("");
  assert.strictEqual(empty.length, 0);
  assert.strictEqual(empty.ones, 0);
  assert.strictEqual(empty.rank1(-1), 0);
  assert.strictEqual(empty.toDigits(), "");
  assert.throws(() => empty.select1(0), RangeError);
  assert.throws(() => empty.select0(0), RangeError);
  const single = new RRRBitvector("1");
  assert.strictEqual(single.rank1(0), 1);
  assert.strictEqual(single.select1(0), 0);
  assert.strictEqual(single.rank0(0), 0);
  const trailing = new RRRBitvector("1".repeat(17));
  assert.strictEqual(trailing.rank1(16), 17);
  assert.strictEqual(trailing.select1(16), 16);
  assert.throws(() => trailing.select1(17), RangeError);
  assert.throws(() => trailing.rank1(17), RangeError);
  assert.throws(() => trailing.rank1(-2), RangeError);

  // Input validation.
  assert.throws(() => new RRRBitvector("012"), TypeError);
  assert.throws(() => new RRRBitvector("1 0"), TypeError);
  assert.throws(() => new RRRBitvector([0, 2]), TypeError);
  assert.throws(() => new RRRBitvector(7), TypeError);
  assert.deepStrictEqual(new RRRBitvector([1, 0, 1]).toDigits(), "101");

  console.log("PASS");
}
