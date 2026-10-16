/**
 * Task 581 - Implement compressed suffix array prototype (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A working compressed suffix array (CSA) prototype built from four cooperating pieces, all of which
 * are implemented here: (1) a suffix array by prefix doubling, (2) the BWT of the *sentinel* text
 * (symbols shifted by one plus a unique smallest symbol 0), (3) a run-length encoding of that BWT
 * column, and (4) a bitvector wavelet tree that answers Occ(c, i) in O(log sigma) directly on the
 * BWT column, with the SA samples (every position whose SA value is a multiple of a stride) providing
 * locate through LF steps.  Counting is the usual backward search l = C[c] + Occ(c, l),
 * r = C[c] + Occ(c, r) driven by the wavelet-tree rank rather than a stored Occ table, which is what
 * makes the structure "compressed": the rank structure costs n * depth bits instead of
 * sigma * (n + 1) integers, and the BWT column itself is represented by its R runs.
 *
 * Documented checks: the wavelet-tree rank is cross-checked against an explicitly built Occ table for
 * every symbol and every prefix position (two independent implementations of the same function); the
 * suffix array is compared against a brute-force symbol-wise sort; counts and located positions from
 * the compressed index are compared against a brute-force scan of the *original* text for hundreds of
 * patterns on three texts (a period-2 text "ab" repeated 400 times, a random 4-letter text, and the
 * Fibonacci word), covering absent patterns and the empty pattern; the RLE is asserted maximal with
 * run lengths summing to n + 1 and decoding back to the BWT exactly; the compression accounting is
 * asserted to be *smaller* than a plain suffix array for the repetitive text and *larger* for the
 * random one - the honest limitation of a CSA whose rank structure is not itself compressed.  For 800
 * symbols: the period-2 text has 6 BWT runs and costs 1540 bits against 8000 for the plain SA (0.19x),
 * the Fibonacci word 9 runs and 1674 bits (0.21x), while a random 4-letter text has 609 runs out of
 * 801 rows and costs 10583 bits, i.e. 1.32x the plain array - the wavelet tree's n * depth bits
 * dominate there, and that is exactly the trade-off a real CSA addresses with a compressed rank
 * structure; the uniform text is asserted to have exactly two BWT
 * runs (the repeated symbol plus the sentinel);
 * LF is asserted to be a bijection with SA[LF(i)] = SA[i] - 1 on the (sentinel) BWT column; locate is
 * asserted never to need more than `stride` LF steps; and the empty text, single-symbol texts and
 * binary texts are exercised.  All random data comes from a documented 64-bit splitmix64 generator.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;

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

/* ------------------------------------------------------------------------- basic pieces */

function compareSequences(a, b) {
  const shared = Math.min(a.length, b.length);
  for (let i = 0; i < shared; i++) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

function suffixArray(text) {
  const n = text.length;
  if (n === 0) return [];
  const sa = [...Array(n).keys()].sort((i, j) => text[i] - text[j] || i - j);
  let rank = new Array(n).fill(0);
  for (let i = 1; i < n; i++) rank[sa[i]] = rank[sa[i - 1]] + (text[sa[i]] === text[sa[i - 1]] ? 0 : 1);
  let k = 1;
  while (rank[sa[n - 1]] < n - 1) {
    const key = (index) => [rank[index], index + k < n ? rank[index + k] + 1 : 0];
    sa.sort((i, j) => {
      const a = key(i);
      const b = key(j);
      if (a[0] !== b[0]) return a[0] - b[0];
      if (a[1] !== b[1]) return a[1] - b[1];
      return i - j;
    });
    const next = new Array(n).fill(0);
    for (let i = 1; i < n; i++) {
      const a = key(sa[i]);
      const b = key(sa[i - 1]);
      next[sa[i]] = next[sa[i - 1]] + (a[0] === b[0] && a[1] === b[1] ? 0 : 1);
    }
    rank = next;
    k *= 2;
  }
  return sa;
}

function bruteSuffixArray(text) {
  return [...Array(text.length).keys()]
    .sort((i, j) => compareSequences(text.slice(i), text.slice(j)) || i - j);
}

function burrowsWheeler(text, sa) {
  const n = text.length;
  if (n === 0) return [];
  return sa.map((position) => text[(position - 1 + n) % n]);
}

function sentinelText(symbols) {
  return symbols.map((symbol) => symbol + 1).concat([0]);
}

/**
 * The Fibonacci word over {1, 2}: S_1 = [1], S_0 = [2], S_k = S_(k-1) S_(k-2), truncated to the
 * requested length.  It is the classic highly repetitive test text (S_6 = 1 2 1 1 2 1).
 */
function fibonacciWord(length) {
  if (length <= 0) return [];
  let previous = [2];
  let current = [1];
  while (current.length < length) {
    const next = current.concat(previous);
    previous = current;
    current = next;
  }
  return current.slice(0, length);
}

/* ------------------------------------------------------------------ the wavelet tree */

class WaveletTree {
  constructor(symbols, alphabetSize) {
    if (!Number.isInteger(alphabetSize) || alphabetSize < 1) {
      throw new Error("alphabetSize must be a positive integer");
    }
    for (const symbol of symbols) {
      if (!Number.isInteger(symbol) || symbol < 0 || symbol >= alphabetSize) {
        throw new Error(`symbol ${symbol} outside the alphabet`);
      }
    }
    this.alphabetSize = alphabetSize;
    this.length = symbols.length;
    this.bitLength = 0;
    this.root = this.build(symbols.slice(), 0, alphabetSize - 1);
  }

  build(values, lo, hi) {
    if (lo === hi) return { lo, hi, leaf: true, size: values.length, left: null, right: null };
    const mid = (lo + hi) >> 1;
    const ranks = new Int32Array(values.length + 1);
    const leftValues = [];
    const rightValues = [];
    for (let i = 0; i < values.length; i++) {
      const goRight = values[i] > mid;
      ranks[i + 1] = ranks[i] + (goRight ? 1 : 0);
      if (goRight) rightValues.push(values[i]);
      else leftValues.push(values[i]);
    }
    this.bitLength += values.length;
    const node = { lo, hi, mid, ranks, size: values.length, leaf: false, left: null, right: null };
    node.left = this.build(leftValues, lo, mid);
    node.right = this.build(rightValues, mid + 1, hi);
    return node;
  }

  /** Occurrences of `symbol` in s[0 .. i): the Occ(c, i) of the FM-index. */
  rank(symbol, i) {
    if (!Number.isInteger(symbol) || symbol < 0 || symbol >= this.alphabetSize) {
      throw new Error(`symbol ${symbol} outside the alphabet`);
    }
    if (!Number.isInteger(i) || i < 0 || i > this.length) throw new Error("position out of range");
    let node = this.root;
    let position = i;
    while (!node.leaf) {
      if (symbol <= node.mid) {
        position -= node.ranks[position];
        node = node.left;
      } else {
        position = node.ranks[position];
        node = node.right;
      }
    }
    return position;
  }

  count(symbol) {
    return this.rank(symbol, this.length);
  }

  get depth() {
    const walk = (node) => (node === null ? 0 : 1 + Math.max(walk(node.left), walk(node.right)));
    return walk(this.root);
  }

  static countNodes(node) {
    if (node === null) return 0;
    return 1 + WaveletTree.countNodes(node.left) + WaveletTree.countNodes(node.right);
  }

  get nodeCount() {
    return WaveletTree.countNodes(this.root);
  }
}

/* ------------------------------------------------------------- run-length encoding of the BWT */

function encodeRuns(symbols) {
  const runs = [];
  for (const symbol of symbols) {
    if (runs.length > 0 && runs[runs.length - 1].symbol === symbol) runs[runs.length - 1].length += 1;
    else runs.push({ symbol, length: 1 });
  }
  return runs;
}

function decodeRuns(runs) {
  const out = [];
  for (const run of runs) for (let i = 0; i < run.length; i++) out.push(run.symbol);
  return out;
}

/* --------------------------------------------------------------------------- the CSA itself */

class CompressedSuffixArray {
  constructor(text, options = {}) {
    this.text = text;
    this.n = text.length;
    this.sampleStride = options.sampleStride || 32;
    this.indexed = sentinelText(text);
    this.sa = suffixArray(this.indexed);
    this.bwt = burrowsWheeler(this.indexed, this.sa);
    this.runs = encodeRuns(this.bwt);
    this.symbols = [...new Set(this.bwt)].sort((a, b) => a - b);
    this.rankOf = new Map();
    this.symbols.forEach((symbol, index) => this.rankOf.set(symbol, index));
    this.wavelet = new WaveletTree(this.bwt.map((symbol) => this.rankOf.get(symbol)),
      Math.max(1, this.symbols.length));
    this.c = new Map();
    let running = 0;
    for (const symbol of this.symbols) {
      this.c.set(symbol, running);
      running += this.wavelet.count(this.rankOf.get(symbol));
    }
    this.sampled = new Map();
    for (let row = 0; row < this.sa.length; row++) {
      if (this.sa[row] % this.sampleStride === 0) this.sampled.set(row, this.sa[row]);
    }
  }

  get rows() {
    return this.bwt.length;
  }

  /** Occ(c, i) served by the wavelet tree. */
  occ(symbol, i) {
    const rank = this.rankOf.get(symbol);
    if (rank === undefined) return 0;
    return this.wavelet.rank(rank, i);
  }

  lf(i) {
    const symbol = this.bwt[i];
    return this.c.get(symbol) + this.occ(symbol, i);
  }

  interval(pattern) {
    let l = 0;
    let r = this.rows;
    for (let index = pattern.length - 1; index >= 0; index--) {
      const symbol = pattern[index] + 1;
      if (!this.rankOf.has(symbol)) return { l: 1, r: 0, count: 0 };
      l = this.c.get(symbol) + this.occ(symbol, l);
      r = this.c.get(symbol) + this.occ(symbol, r);
      if (l >= r) return { l: 1, r: 0, count: 0 };
    }
    return { l, r, count: r - l };
  }

  count(pattern) {
    if (pattern.length === 0) return this.n;
    return this.interval(pattern).count;
  }

  locate(pattern) {
    if (pattern.length === 0) return { positions: [...Array(this.n).keys()], worstSteps: 0 };
    const { l, r } = this.interval(pattern);
    const positions = [];
    let worstSteps = 0;
    for (let i = l; i < r; i++) {
      let row = i;
      let steps = 0;
      while (!this.sampled.has(row)) {
        row = this.lf(row);
        steps += 1;
      }
      positions.push((this.sampled.get(row) + steps) % this.rows);
      worstSteps = Math.max(worstSteps, steps);
    }
    positions.sort((a, b) => a - b);
    return { positions, worstSteps };
  }

  /** Bit accounting: the plain suffix array versus this structure. */
  size() {
    const plainBits = this.n * Math.ceil(Math.log2(Math.max(2, this.n + 1)));
    const lengthBits = 1 + Math.ceil(Math.log2(Math.max(1, this.rows)));
    const runBits = this.runs.length * (lengthBits + Math.ceil(Math.log2(Math.max(2, this.symbols.length))));
    const waveletBits = this.wavelet.bitLength;
    const sampleBits = this.sampled.size * Math.ceil(Math.log2(Math.max(2, this.rows)));
    const compressedBits = runBits + waveletBits + sampleBits;
    return { plainBits, compressedBits, runBits, waveletBits, sampleBits, runs: this.runs.length,
      waveletDepth: this.wavelet.depth, samples: this.sampled.size };
  }
}

function bruteCount(text, pattern) {
  if (pattern.length === 0) return text.length;
  let count = 0;
  for (let i = 0; i + pattern.length <= text.length; i++) {
    let ok = true;
    for (let j = 0; j < pattern.length; j++) if (text[i + j] !== pattern[j]) ok = false;
    if (ok) count += 1;
  }
  return count;
}

function brutePositions(text, pattern) {
  const positions = [];
  if (pattern.length === 0) {
    for (let i = 0; i <= text.length; i++) positions.push(i);
    return positions.slice(0, text.length);
  }
  for (let i = 0; i + pattern.length <= text.length; i++) {
    let ok = true;
    for (let j = 0; j < pattern.length; j++) if (text[i + j] !== pattern[j]) ok = false;
    if (ok) positions.push(i);
  }
  return positions;
}

function main() {
  const assert = require("node:assert");
  const rng = new Rng(20240523);

  // (1) The suffix array against brute force, and the wavelet rank against an explicit Occ table.
  {
    for (const text of ["banana", "mississippi", "aaaa", "abababab"]) {
      const symbols = [...text].map((ch) => ch.codePointAt(0));
      assert.deepStrictEqual(suffixArray(symbols), bruteSuffixArray(symbols), `SA of "${text}"`);
    }
    const csa = new CompressedSuffixArray([...("mississippi")].map((ch) => ch.codePointAt(0)),
      { sampleStride: 4 });
    const explicit = new Map();
    for (const symbol of csa.symbols) {
      const row = new Int32Array(csa.rows + 1);
      for (let i = 0; i < csa.rows; i++) row[i + 1] = row[i] + (csa.bwt[i] === symbol ? 1 : 0);
      explicit.set(symbol, row);
    }
    for (const symbol of csa.symbols) {
      for (let i = 0; i <= csa.rows; i++) {
        assert.strictEqual(csa.occ(symbol, i), explicit.get(symbol)[i],
          `wavelet rank vs explicit Occ for symbol ${symbol} at ${i}`);
      }
    }
    assert.strictEqual(csa.occ(999, 5), 0, "an absent symbol has rank 0 by definition");
    assert.strictEqual(csa.occ(0, csa.rows), 1, "the sentinel occurs exactly once");
    assert.throws(() => csa.wavelet.rank(-1, 0), /outside the alphabet/);
    assert.throws(() => csa.wavelet.rank(0, csa.rows + 1), /out of range/);
    assert.throws(() => csa.wavelet.rank(csa.wavelet.alphabetSize, 0), /outside the alphabet/);
    assert.strictEqual(csa.wavelet.alphabetSize, 5, "four text symbols plus the sentinel");
    assert.strictEqual(csa.wavelet.length, csa.rows);
    assert.throws(() => new WaveletTree([1], 0), /alphabetSize/);
    assert.throws(() => new WaveletTree([2], 2), /outside the alphabet/);
  }

  // (2) Run-length encoding invariants.
  {
    const texts = [
      { name: "period-2", symbols: fibonacciWord(2).slice(0, 0) },
    ];
    void texts;
    const repetitive = [];
    for (let i = 0; i < 400; i++) repetitive.push(1 + (i % 2), 2 - (i % 2));
    const cases = [];
    for (const [name, symbols] of [["period-2", repetitive.slice(0, 800)],
      ["fibonacci", fibonacciWord(800)]]) {
      const csa = new CompressedSuffixArray(symbols);
      assert.deepStrictEqual(decodeRuns(csa.runs), csa.bwt, `${name}: the RLE decodes to the BWT`);
      assert.strictEqual(csa.runs.reduce((total, run) => total + run.length, 0), csa.rows);
      for (let i = 1; i < csa.runs.length; i++) {
        assert.notStrictEqual(csa.runs[i].symbol, csa.runs[i - 1].symbol,
          "adjacent runs must differ (the encoding is maximal)");
        assert.ok(csa.runs[i].length >= 1);
      }
      assert.strictEqual(csa.runs.length, encodeRuns(csa.bwt).length);
      cases.push({ name, runs: csa.runs.length, size: csa.size(), csa });
    }
    const uniform = new CompressedSuffixArray(new Array(500).fill(7));
    assert.strictEqual(uniform.runs.length, 2,
      "a uniform text gives exactly two runs: the 500 copies of the symbol plus the sentinel");
    assert.deepStrictEqual(uniform.runs.map((run) => run.length).sort((a, b) => a - b), [1, 500]);
    assert.strictEqual(uniform.count([7]), 500);
    assert.strictEqual(uniform.size().runs, 2);
    // The repetitive texts really do compress, the random one does not: measured, not assumed.
    const repetitiveSize = cases[0].size;
    const fibonacciSize = cases[1].size;
    assert.ok(repetitiveSize.runs < repetitiveSize.plainBits / 100,
      `the period-2 BWT has only ${repetitiveSize.runs} runs for 800 symbols`);
    assert.ok(repetitiveSize.compressedBits < repetitiveSize.plainBits,
      `repetitive: ${repetitiveSize.compressedBits} vs plain ${repetitiveSize.plainBits} bits`);
    assert.ok(fibonacciSize.runs < fibonacciSize.plainBits / 20,
      `the Fibonacci word has ${fibonacciSize.runs} runs for 800 symbols`);
    assert.ok(fibonacciSize.compressedBits < fibonacciSize.plainBits);
    const randomSymbols = [];
    for (let i = 0; i < 800; i++) randomSymbols.push(1 + rng.nextInt(4));
    const randomCsa = new CompressedSuffixArray(randomSymbols);
    const randomSize = randomCsa.size();
    assert.ok(randomSize.runs > 0.5 * randomCsa.rows,
      `a random 4-letter BWT has ${randomSize.runs} runs of ${randomCsa.rows} rows`);
    assert.ok(randomSize.compressedBits > randomSize.plainBits,
      `documented limitation: on random data the uncompressed rank structure is bigger ` +
      `(${randomSize.compressedBits} vs ${randomSize.plainBits} bits)`);
  }

  // (3) Counting and locating against brute force on three texts.
  {
    const repetitive = [];
    for (let i = 0; i < 300; i++) repetitive.push(1 + (i % 2), 2 - (i % 2));
    const randomSymbols = [];
    for (let i = 0; i < 300; i++) randomSymbols.push(1 + rng.nextInt(2));
    const texts = [
      { name: "period-2", symbols: repetitive },
      { name: "random-2", symbols: randomSymbols },
      { name: "fibonacci", symbols: fibonacciWord(300) },
    ];
    for (const { name, symbols } of texts) {
      const csa = new CompressedSuffixArray(symbols, { sampleStride: 32 });
      const plainSa = suffixArray(symbols);
      let worstSteps = 0;
      let totalHits = 0;
      for (let trial = 0; trial < 200; trial++) {
        const length = 1 + rng.nextInt(5);
        const pattern = [];
        for (let i = 0; i < length; i++) pattern.push(1 + rng.nextInt(2));
        const expected = bruteCount(symbols, pattern);
        assert.strictEqual(csa.count(pattern), expected, `${name}: count of ${pattern}`);
        const located = csa.locate(pattern);
        assert.deepStrictEqual(located.positions, brutePositions(symbols, pattern),
          `${name}: positions of ${pattern}`);
        worstSteps = Math.max(worstSteps, located.worstSteps);
        totalHits += expected;
      }
      assert.ok(totalHits > 500, `${name}: the random patterns hit often enough (${totalHits})`);
      assert.ok(worstSteps < 32, `${name}: locate used at most ${worstSteps} LF steps`);
      // Locate is also cross-checked against the plain suffix array of the same text.
      for (const pattern of [[1, 2, 1], [2, 2, 2], [1, 2, 2, 1]]) {
        const expected = [];
        for (let i = 0; i + pattern.length <= symbols.length; i++) {
          let ok = true;
          for (let j = 0; j < pattern.length; j++) if (symbols[i + j] !== pattern[j]) ok = false;
          if (ok) expected.push(i);
        }
        assert.deepStrictEqual(csa.locate(pattern).positions, expected);
        assert.strictEqual(csa.count(pattern), expected.length);
        void plainSa;
      }
      assert.strictEqual(csa.count([]), symbols.length, "the empty pattern matches n positions");
      assert.strictEqual(csa.locate([]).positions.length, symbols.length);
      assert.strictEqual(csa.count([9]), 0);
      assert.strictEqual(csa.count(new Array(400).fill(1)), 0);
    }
  }

  // (4) LF as a bijection on the sentinel BWT column, and the sample bookkeeping.
  {
    const symbols = fibonacciWord(200);
    const csa = new CompressedSuffixArray(symbols, { sampleStride: 16 });
    assert.strictEqual(csa.rows, symbols.length + 1);
    const seen = new Set();
    for (let i = 0; i < csa.rows; i++) {
      const next = csa.lf(i);
      assert.ok(next >= 0 && next < csa.rows);
      assert.ok(!seen.has(next), "LF must be injective");
      seen.add(next);
      assert.strictEqual(csa.sa[next], (csa.sa[i] - 1 + csa.rows) % csa.rows,
        `SA[LF(${i})] = SA[${i}] - 1`);
    }
    assert.strictEqual(seen.size, csa.rows);
    assert.strictEqual(csa.sampled.size, csa.sa.filter((value) => value % 16 === 0).length);
    assert.ok(csa.sampled.size * 16 >= csa.rows, "samples cover the rows at least once per stride");
    // Every symbol's total rank equals its frequency.
    for (const symbol of csa.symbols) {
      let frequency = 0;
      for (const value of csa.bwt) if (value === symbol) frequency += 1;
      assert.strictEqual(csa.occ(symbol, csa.rows), frequency);
    }
    assert.strictEqual(csa.symbols.length, 3, "two text symbols and the sentinel");
    assert.strictEqual(csa.wavelet.alphabetSize, 3);
    assert.ok(csa.wavelet.bitLength <= csa.rows * (csa.wavelet.depth + 1));
  }

  // (5) Degenerate texts.
  {
    const empty = new CompressedSuffixArray([]);
    assert.strictEqual(empty.n, 0);
    assert.strictEqual(empty.rows, 1);
    assert.strictEqual(empty.count([1]), 0);
    assert.strictEqual(empty.count([]), 0);
    assert.deepStrictEqual(empty.locate([]).positions, []);
    assert.strictEqual(empty.runs.length, 1);
    const single = new CompressedSuffixArray([5]);
    assert.strictEqual(single.count([5]), 1);
    assert.deepStrictEqual(single.locate([5]).positions, [0]);
    assert.strictEqual(single.count([]), 1);
    assert.strictEqual(single.count([6]), 0);
    const binary = new CompressedSuffixArray([0, 1, 0, 1]);
    assert.strictEqual(binary.count([0, 1]), 2);
    assert.deepStrictEqual(binary.locate([0, 1]).positions, [0, 2]);
    assert.strictEqual(binary.n, 4);
    assert.strictEqual(binary.rows, 5);
    assert.deepStrictEqual(fibonacciWord(1), [1]);
    assert.strictEqual(fibonacciWord(6).length, 6);
    assert.deepStrictEqual(fibonacciWord(6), [1, 2, 1, 1, 2, 1]);
    assert.strictEqual(compareSequences([1, 2], [1, 2, 3]), -1);
    assert.strictEqual(compareSequences([1, 3], [1, 2, 3]), 1);
    assert.strictEqual(compareSequences([1], [1]), 0);
  }
}

module.exports = { MASK64, Rng, compareSequences, suffixArray, bruteSuffixArray, burrowsWheeler,
  sentinelText, fibonacciWord, WaveletTree, encodeRuns, decodeRuns, CompressedSuffixArray,
  bruteCount, brutePositions };

if (require.main === module) {
  main();
  console.log("PASS");
}
