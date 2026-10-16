/** Task 606 - Implement wavelet matrix for large alphabet rank/select. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A wavelet matrix stores an integer array of `bits` bits per element as one bit vector per bit
 * position (most significant first). Level L stably partitions its input into zeros followed by ones,
 * so element i moves to rank0(L, i) or zeroCounts[L] + rank1(L, i); the sequence below the last level
 * is ordered by the bit-reversed code, which the checks confirm. Only the bit vectors, their zero
 * counts and 256-bit rank checkpoints are stored - N * bits bits plus o(N) per level - which is the
 * point of using it for a *large* alphabet (16- and 20-bit values here) instead of per-character
 * indices.
 *
 * Operations: access, rank(value, i), select(value, k) (binary search over the monotone prefix rank),
 * rangeCount via two less-than walks, quantile (k-th smallest in a range) and range minimum/maximum.
 *
 * Checks: every operation cross-verified against brute force on a small alphabet, 16-bit and 20-bit
 * values from a seeded LCG, exhaustive rank/select/quantile loops, quantile over the whole array
 * reproducing the sorted array, the bit-reversal structure of the last level, out-of-range selects
 * returning -1, and boundary shapes (empty, single, constant, alphabet boundary, oversize value).
 */
"use strict";

const WORD_BITS = 32;
const CHECKPOINT_WORDS = 8; // rank checkpoint every 256 bits

function popcount32(x) {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
}

class WaveletMatrix {
  constructor(values, bitWidth = null) {
    this.length = values.length;
    let maxValue = 0;
    for (const value of values) if (value > maxValue) maxValue = value;
    if (bitWidth !== null && maxValue >= 2 ** bitWidth) throw new Error("value exceeds the bit width");
    this.bits = bitWidth === null ? Math.max(1, 32 - Math.clz32(maxValue)) : bitWidth;
    if (this.bits < 1) throw new Error("the bit width must be positive");
    const wordCount = Math.max(1, Math.ceil(this.length / WORD_BITS));
    this.marks = [];
    this.zeroCounts = [];
    this.checkpoints = [];
    let current = Int32Array.from(values);
    for (let level = 0; level < this.bits; level++) {
      const shift = this.bits - 1 - level;
      const marks = new Uint32Array(wordCount);
      const zeros = new Int32Array(this.length);
      const ones = new Int32Array(this.length);
      let zeroTotal = 0;
      for (let i = 0; i < this.length; i++) {
        if ((current[i] >>> shift) & 1) {
          ones[i - zeroTotal] = current[i]; // i - zeroTotal = number of ones seen so far
          marks[i >>> 5] |= 1 << (i & 31);
        } else {
          zeros[zeroTotal++] = current[i];
        }
      }
      const next = new Int32Array(this.length);
      for (let i = 0; i < zeroTotal; i++) next[i] = zeros[i];
      for (let i = 0; i < this.length - zeroTotal; i++) next[zeroTotal + i] = ones[i];
      this.marks.push(marks);
      this.zeroCounts.push(zeroTotal);
      const checkpoints = new Int32Array(Math.ceil(wordCount / CHECKPOINT_WORDS) + 1);
      let total = 0;
      for (let word = 0; word < wordCount; word++) {
        if (word % CHECKPOINT_WORDS === 0) checkpoints[word / CHECKPOINT_WORDS] = total;
        total += popcount32(marks[word]);
      }
      checkpoints[Math.ceil(wordCount / CHECKPOINT_WORDS)] = total;
      this.checkpoints.push(checkpoints);
      current = next;
    }
    this.lastLevel = current;
    const counts = new Map();
    for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
    this.valueIndex = new Map();
    this.counts = [];
    for (const [value, count] of [...counts.entries()].sort((a, b) => a[0] - b[0])) {
      this.valueIndex.set(value, this.counts.length);
      this.counts.push(count);
    }
  }

  /** Number of ones in marks[level][0..i). */
  rank1(level, i) {
    if (i <= 0) return 0;
    const word = i >>> 5;
    const block = (word / CHECKPOINT_WORDS) | 0;
    let total = this.checkpoints[level][block];
    for (let w = block * CHECKPOINT_WORDS; w < word; w++) total += popcount32(this.marks[level][w]);
    const mask = (i & 31) === 0 ? 0 : 0xffffffff >>> (32 - (i & 31));
    return total + popcount32(this.marks[level][word] & mask);
  }

  rank0(level, i) {
    return i - this.rank1(level, i);
  }

  bitAt(value, level) {
    return (value >>> (this.bits - 1 - level)) & 1;
  }

  /** values[i]. */
  access(i) {
    if (i < 0 || i >= this.length) return -1;
    let value = 0;
    let position = i;
    for (let level = 0; level < this.bits; level++) {
      if ((this.marks[level][position >>> 5] >>> (position & 31)) & 1) {
        value |= 1 << (this.bits - 1 - level);
        position = this.zeroCounts[level] + this.rank1(level, position);
      } else {
        position = this.rank0(level, position);
      }
    }
    return value;
  }

  /** Number of occurrences of `value` in values[0..i). */
  rank(value, i) {
    if (i <= 0 || this.valueIndex.get(value) === undefined) return 0;
    const bound = Math.min(i, this.length);
    let left = 0;
    let right = bound;
    for (let level = 0; level < this.bits; level++) {
      if (this.bitAt(value, level)) {
        left = this.zeroCounts[level] + this.rank1(level, left);
        right = this.zeroCounts[level] + this.rank1(level, right);
      } else {
        left = this.rank0(level, left);
        right = this.rank0(level, right);
      }
    }
    return right - left;
  }

  total(value) {
    const index = this.valueIndex.get(value);
    return index === undefined ? 0 : this.counts[index];
  }

  /** Index of the (k+1)-th occurrence of `value`, or -1; rank(value, i) is monotone in i. */
  select(value, k) {
    if (k < 0 || k >= this.total(value)) return -1;
    let low = 0;
    let high = this.length - 1;
    while (low < high) {
      const mid = (low + high) >> 1;
      if (this.rank(value, mid + 1) > k) high = mid;
      else low = mid + 1;
    }
    return low;
  }

  /** Number of values smaller than `upper` inside values[l..r). */
  lessThan(upper, l, r) {
    if (upper <= 0) return 0;
    let count = 0;
    let left = l;
    let right = r;
    for (let level = 0; level < this.bits; level++) {
      const left0 = this.rank0(level, left);
      const right0 = this.rank0(level, right);
      if (this.bitAt(upper, level)) {
        count += right0 - left0;
        left = this.zeroCounts[level] + (left - left0);
        right = this.zeroCounts[level] + (right - right0);
      } else {
        left = left0;
        right = right0;
      }
      if (left >= right) break;
    }
    return count;
  }

  /** Number of values in [low, high) inside values[l..r). */
  rangeCount(low, high, l, r) {
    const from = Math.max(0, l);
    const to = Math.min(this.length, r);
    if (from >= to || low >= high) return 0;
    return this.lessThan(high, from, to) - this.lessThan(low, from, to);
  }

  /** k-th smallest value in values[l..r) (0-based k). */
  quantile(l, r, k) {
    if (l < 0 || r > this.length || l >= r || k < 0 || k >= r - l) return -1;
    let left = l;
    let right = r;
    let value = 0;
    for (let level = 0; level < this.bits; level++) {
      const left0 = this.rank0(level, left);
      const right0 = this.rank0(level, right);
      if (k < right0 - left0) {
        left = left0;
        right = right0;
      } else {
        value |= 1 << (this.bits - 1 - level);
        k -= right0 - left0;
        left = this.zeroCounts[level] + (left - left0);
        right = this.zeroCounts[level] + (right - right0);
      }
    }
    return value;
  }

  rangeMin(l, r) {
    return this.quantile(l, r, 0);
  }

  rangeMax(l, r) {
    return this.quantile(l, r, r - l - 1);
  }

  /** Structural invariant: the last level is ordered by reversed bit code. */
  isStructureValid() {
    const reverse = (value) => {
      let out = 0;
      for (let i = 0; i < this.bits; i++) if ((value >>> i) & 1) out |= 1 << (this.bits - 1 - i);
      return out;
    };
    for (let i = 1; i < this.length; i++) {
      if (reverse(this.lastLevel[i - 1]) > reverse(this.lastLevel[i])) return false;
    }
    return true;
  }
}

/** 32-bit LCG with exact integer arithmetic; the high bits avoid the short period of the low bits. */
function lcgValues(length, bound, seed) {
  let state = seed >>> 0;
  const values = [];
  for (let i = 0; i < length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values.push((state >>> 8) % bound);
  }
  return [values, state];
}

const bruteRank = (values, value, i) => values.slice(0, i).filter((v) => v === value).length;
const bruteSelect = (values, value, k) => (k < 0 ? -1 : values.indexOf(value, 0) === -1 ? -1
  : values.reduce((found, v, i) => (found >= 0 || v !== value ? found : k-- === 0 ? i : -1), -1));

module.exports = { WaveletMatrix, popcount32, lcgValues, bruteRank, bruteSelect };

if (require.main === module) {
  const assert = require("node:assert");

  const empty = new WaveletMatrix([], 8);
  assert.deepStrictEqual([empty.rank(5, 0), empty.select(5, 0), empty.quantile(0, 0, 0), empty.access(0)],
    [0, -1, -1, -1]);

  const one = new WaveletMatrix([7], 4);
  assert.deepStrictEqual([one.bits, one.access(0), one.rank(7, 1), one.rank(7, 0), one.select(7, 0), one.select(7, 1)],
    [4, 7, 1, 0, 0, -1]);
  assert.strictEqual(one.rangeCount(7, 8, 0, 1), 1);
  assert.strictEqual(one.rangeCount(0, 7, 0, 1), 0);

  const constant = new WaveletMatrix(new Array(100).fill(9), 5);
  assert.deepStrictEqual([constant.total(9), constant.rank(9, 100), constant.select(9, 99), constant.select(9, 100)],
    [100, 100, 99, -1]);
  assert.deepStrictEqual([constant.quantile(0, 100, 50), constant.rangeMin(0, 100), constant.rangeMax(0, 100)],
    [9, 9, 9]);

  const small = [5, 1, 5, 3, 0, 7, 5, 2, 1, 6, 3, 3, 0, 9, 4];
  const matrix = new WaveletMatrix(small, 4);
  const sorted = [...small].sort((a, b) => a - b);
  assert.ok(matrix.isStructureValid());
  assert.deepStrictEqual([...matrix.lastLevel].sort((a, b) => a - b), sorted);
  for (let i = 0; i < small.length; i++) {
    assert.deepStrictEqual([matrix.access(i), matrix.quantile(0, small.length, i)], [small[i], sorted[i]]);
  }
  for (let value = 0; value < 16; value++) {
    assert.strictEqual(matrix.total(value), bruteRank(small, value, small.length), `total ${value}`);
    for (let i = 0; i <= small.length; i++) {
      assert.strictEqual(matrix.rank(value, i), bruteRank(small, value, i), `rank ${value} ${i}`);
    }
    for (let k = -1; k <= bruteRank(small, value, small.length); k++) {
      assert.strictEqual(matrix.select(value, k), bruteSelect(small, value, k), `select ${value} ${k}`);
    }
  }
  for (let l = 0; l < small.length; l++) {
    for (let r = l + 1; r <= small.length; r++) {
      const slice = small.slice(l, r).sort((a, b) => a - b);
      for (let k = 0; k < r - l; k++) {
        assert.strictEqual(matrix.quantile(l, r, k), slice[k], `quantile ${l} ${r} ${k}`);
      }
      for (let low = 0; low <= 10; low++) {
        for (let high = low; high <= 10; high++) {
          assert.strictEqual(matrix.rangeCount(low, high, l, r),
            small.slice(l, r).filter((v) => v >= low && v < high).length, `range ${low} ${high} ${l} ${r}`);
        }
      }
      assert.deepStrictEqual([matrix.rangeMin(l, r), matrix.rangeMax(l, r)], [slice[0], slice[slice.length - 1]]);
    }
  }

  const boundary = new WaveletMatrix([65535, 0, 32768, 65535, 1, 65534], 16);
  assert.deepStrictEqual([boundary.bits, boundary.access(2), boundary.total(65535), boundary.select(65535, 0),
    boundary.select(65535, 1), boundary.rank(65535, 4), boundary.rangeCount(1, 65535, 0, 6)], [16, 32768, 2, 0, 3, 2, 3]);
  assert.deepStrictEqual([boundary.rangeMin(0, 6), boundary.rangeMax(0, 6)], [0, 65535]);
  assert.throws(() => new WaveletMatrix([65536], 16), /bit width/);
  assert.deepStrictEqual([new WaveletMatrix([3]).bits, new WaveletMatrix([0]).bits], [2, 1]);

  // Large alphabet: 16- and 20-bit values from a deterministic PRNG, plus a dense 12-bit array.
  let seed = 606;
  for (const [bits, bound, length] of [[16, 65536, 300], [20, 1048576, 250], [12, 4096, 400]]) {
    let values;
    [values, seed] = lcgValues(length, bound, seed);
    const indexed = new WaveletMatrix(values, bits);
    assert.ok(indexed.isStructureValid(), `structure ${bits}`);
    assert.deepStrictEqual([...indexed.lastLevel].sort((a, b) => a - b), [...values].sort((a, b) => a - b));
    for (let i = 0; i < values.length; i++) assert.strictEqual(indexed.access(i), values[i], `access ${bits}`);
    for (let probe = 0; probe < 30; probe++) {
      const value = values[(probe * 97) % values.length];
      assert.strictEqual(indexed.rank(value, length), bruteRank(values, value, length), `rank ${value}`);
      assert.strictEqual(indexed.select(value, 0), bruteSelect(values, value, 0), `select ${value}`);
      const l = (probe * 13) % (length - 20);
      const r = l + 15;
      assert.strictEqual(indexed.rank(value, r) - indexed.rank(value, l),
        values.slice(l, r).filter((v) => v === value).length, `rank range ${value}`);
      const slice = values.slice(l, r).sort((a, b) => a - b);
      assert.strictEqual(indexed.quantile(l, r, probe % 15), slice[probe % 15], `quantile ${bits}`);
      assert.deepStrictEqual([indexed.rangeMin(l, r), indexed.rangeMax(l, r)], [slice[0], slice[14]]);
    }
    const sortedValues = [...values].sort((a, b) => a - b);
    assert.deepStrictEqual([indexed.quantile(0, length, 0), indexed.quantile(0, length, length - 1)],
      [sortedValues[0], sortedValues[length - 1]]);
    assert.strictEqual(indexed.select(Math.max(...values) + 1, 0), -1);
  }

  let bigSeed = seed;
  let big;
  [big, bigSeed] = lcgValues(4000, 1 << 16, bigSeed);
  const bigMatrix = new WaveletMatrix(big, 16);
  const sortedBig = [...big].sort((a, b) => a - b);
  for (let probe = 0; probe < 150; probe++) {
    const value = big[(probe * 31) % big.length];
    assert.strictEqual(bigMatrix.total(value), bruteRank(big, value, big.length), `big total ${value}`);
    assert.strictEqual(bigMatrix.select(value, 0), bruteSelect(big, value, 0), `big select ${value}`);
  }
  for (let i = 0; i < big.length; i += 250) assert.strictEqual(bigMatrix.quantile(0, big.length, i), sortedBig[i]);
  assert.ok(bigMatrix.isStructureValid());

  console.log(`wavelet matrix: 16 bits x ${bigMatrix.length} values, ` +
    `${bigMatrix.marks.reduce((a, m) => a + m.length * 4, 0)} bytes of bit vectors`);
  console.log("PASS");
}
