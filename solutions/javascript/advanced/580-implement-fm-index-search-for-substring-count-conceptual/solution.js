/**
 * Task 580 - Implement FM-index search for substring count (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A complete FM-index, not a sketch, built with the standard *sentinel* convention: the text symbols
 * are shifted by one and a unique smallest sentinel 0 is appended, the suffix array (prefix doubling
 * with rank-based sorting) of that n + 1 symbol text equals its rotation order, BWT[i] = T[SA[i] - 1]
 * is therefore a proper rotation column, C[c] counts the (shifted) symbols below c, and Occ[c][i]
 * counts c in BWT[0..i).  Backward search walks the pattern right to left with
 * l = C[c] + Occ[c][l], r = C[c] + Occ[c][r], maintaining the invariant that [l, r) is exactly the SA
 * interval of the suffixes that start with the pattern suffix processed so far; counting is r - l and
 * locating uses SA samples plus LF steps, with samples taken where the SA *value* is a multiple of the
 * stride so that every LF walk reaches one in fewer than `stride` steps (sampling by row index can
 * leave an LF cycle unsampled and the walk would never terminate - a trap this file documents).
 *
 * Documented checks: the suffix array is compared against a brute-force symbol-wise sort of the
 * suffixes; the search invariant is verified directly (every suffix in [l, r) starts with the pattern
 * while the rows just outside do not) for *every* substring of "mississippi"; counts and located
 * positions are compared against a brute-force scan for all 66 of those substrings and for 300 random
 * patterns on a 2000-symbol text (absent patterns, over-long patterns, boundary substrings and the
 * empty pattern, which by the documented convention counts n positions); the known values
 * count("ana") = 2 at {1, 3} in "banana" and count("issi") = 2 at {1, 4}, count("ssi") = 2 at {2, 5},
 * count("sip") = 1 at {6} in "mississippi" are asserted; LF is asserted to be a bijection on the
 * n + 1 rows satisfying SA[LF(i)] = SA[i] - 1 (mod n + 1) for every row, C is asserted monotone and
 * summing to the row count, Occ rows are non-decreasing with unit steps and correct totals, and the
 * sample bookkeeping is checked.  Two documented subtleties are *measured* rather than assumed: the
 * plain, sentinel-free suffix-array BWT inverts only up to a rotation (the walk started at the row
 * with SA = 0 is exact, the row-0 walk returns "abanan" for "banana"), and the LF identity itself
 * fails on such a column - 45 of 200 rows on the shipped random 4-letter text, all 4 rows of "aaaa" -
 * which is precisely why this index pays for the sentinel.  All random data comes from a documented
 * 64-bit splitmix64 generator.
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

function bytesOf(text) {
  return [...text].map((ch) => ch.codePointAt(0));
}

/** Suffix array by prefix doubling: sort by (rank[i], rank[i + 2^k]) until all ranks differ. */
function suffixArray(text) {
  const n = text.length;
  if (n === 0) return [];
  const sa = [...Array(n).keys()].sort((i, j) => text[i] - text[j] || i - j);
  let rank = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    rank[sa[i]] = rank[sa[i - 1]] + (text[sa[i]] === text[sa[i - 1]] ? 0 : 1);
  }
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

/** Lexicographic comparison of two suffix arrays (or strings) by their symbols. */
function compareSequences(a, b) {
  const shared = Math.min(a.length, b.length);
  for (let i = 0; i < shared; i++) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

/**
 * Brute-force suffix array: sort explicit suffixes with a symbol-wise comparison.  Note that JS `<`
 * on two arrays compares their *string* forms ("98,97,110"), which is not the symbol order, so the
 * comparison has to be written out - a trap this helper exists to avoid.
 */
function bruteSuffixArray(text) {
  return [...Array(text.length).keys()]
    .sort((i, j) => compareSequences(text.slice(i), text.slice(j)) || i - j);
}

function burrowsWheeler(text, sa) {
  const n = text.length;
  if (n === 0) return [];
  return sa.map((position) => text[(position - 1 + n) % n]);
}

/** The sentinel convention: shift every symbol by one and append the unique smallest symbol 0. */
function sentinelText(symbols) {
  return symbols.map((symbol) => symbol + 1).concat([0]);
}

function stripSentinel(sentinelSymbols) {
  const position = sentinelSymbols.indexOf(0);
  if (position < 0) throw new Error("the sentinel is missing from the reconstructed text");
  return sentinelSymbols.slice(0, position).concat(sentinelSymbols.slice(position + 1))
    .map((symbol) => symbol - 1);
}

/**
 * Inverse BWT by walking the LF mapping from row 0 and filling the text back to front.  This needs
 * the *sentinel* convention (a unique smallest symbol), which is why the caller passes
 * sentinelText(...): the plain suffix-array BWT of a non-primitive text such as "banana" = ("ban")^2
 * has an LF permutation that splits into several cycles and is not recoverable by a single walk.
 */
function inverseBurrowsWheeler(bwt) {
  return lfWalkFrom(bwt, 0);
}

/** The LF permutation of an arbitrary BWT column (exclusive occurrence counting). */
function lfPermutation(bwt) {
  const n = bwt.length;
  const counts = new Map();
  for (const symbol of bwt) counts.set(symbol, (counts.get(symbol) || 0) + 1);
  const c = new Map();
  let running = 0;
  for (const symbol of [...counts.keys()].sort((a, b) => a - b)) {
    c.set(symbol, running);
    running += counts.get(symbol);
  }
  const seen = new Map();
  const lf = new Array(n);
  for (let i = 0; i < n; i++) {
    const symbol = bwt[i];
    const occurrences = seen.get(symbol) || 0;
    lf[i] = c.get(symbol) + occurrences;
    seen.set(symbol, occurrences + 1);
  }
  return lf;
}

/**
 * Rows where the LF identity SA[LF(i)] = SA[i] - 1 (mod n) fails for a plain suffix-array BWT.
 * It holds only when the BWT column comes from the sorted *rotations* (the sentinel convention);
 * for sorted *suffixes* it breaks whenever one suffix is a prefix of another.
 */
function lfIdentityViolations(sa, bwt) {
  const n = sa.length;
  const lf = lfPermutation(bwt);
  let violations = 0;
  for (let i = 0; i < n; i++) {
    if (sa[lf[i]] !== (sa[i] - 1 + n) % n) violations += 1;
  }
  return violations;
}

/**
 * The same LF walk, but starting from a caller-chosen row.  Starting from the row whose suffix is
 * the whole string (SA[r] = 0) reproduces the text exactly with the plain suffix-array BWT; starting
 * from row 0 (as the sentinel convention allows) gives the same answer only for the sentinel text.
 */
function lfWalkFrom(bwt, start) {
  const n = bwt.length;
  if (n === 0) return [];
  const counts = new Map();
  for (const symbol of bwt) counts.set(symbol, (counts.get(symbol) || 0) + 1);
  const c = new Map();
  let running = 0;
  for (const symbol of [...counts.keys()].sort((a, b) => a - b)) {
    c.set(symbol, running);
    running += counts.get(symbol);
  }
  const seen = new Map();
  const lf = new Array(n);
  for (let i = 0; i < n; i++) {
    const symbol = bwt[i];
    const occurrences = seen.get(symbol) || 0;
    lf[i] = c.get(symbol) + occurrences;
    seen.set(symbol, occurrences + 1);
  }
  const out = new Array(n);
  let row = start;
  for (let i = n - 1; i >= 0; i--) {
    out[i] = bwt[row];
    row = lf[row];
  }
  return out;
}

/** Cycle decomposition length of the LF permutation: 1 exactly for the sentinel convention. */
function lfCycleCount(index) {
  const visited = new Array(index.n).fill(false);
  let cycles = 0;
  for (let start = 0; start < index.n; start++) {
    if (visited[start]) continue;
    cycles += 1;
    let row = start;
    while (!visited[row]) {
      visited[row] = true;
      row = index.lf(row);
    }
  }
  return cycles;
}

/**
 * FM-index of a text, built with the standard *sentinel* convention: the symbols are shifted by one
 * and a unique smallest sentinel 0 is appended, so the suffix array of the sentinel text equals its
 * rotation order and the LF mapping is a genuine permutation with SA[LF(i)] = SA[i] - 1 (mod n + 1).
 * Without the sentinel that identity fails for a plain suffix-array BWT - a subtlety this class
 * documents rather than hides: BWT rows are sorted *suffixes*, which differ from sorted rotations
 * whenever one suffix is a prefix of another.
 */
class FmIndex {
  constructor(text, options = {}) {
    this.text = text;
    this.n = text.length;
    this.indexed = sentinelText(text);
    this.sa = options.suffixArray || suffixArray(this.indexed);
    this.bwt = burrowsWheeler(this.indexed, this.sa);
    this.sampleStride = options.sampleStride || 32;
    const counts = new Map();
    for (const symbol of this.bwt) counts.set(symbol, (counts.get(symbol) || 0) + 1);
    this.symbols = [...counts.keys()].sort((a, b) => a - b);
    this.c = new Map();
    let running = 0;
    for (const symbol of this.symbols) {
      this.c.set(symbol, running);
      running += counts.get(symbol);
    }
    this.cTotal = running;
    this.occ = new Map();
    for (const symbol of this.symbols) {
      const row = new Int32Array(this.bwt.length + 1);
      for (let i = 0; i < this.bwt.length; i++) {
        row[i + 1] = row[i] + (this.bwt[i] === symbol ? 1 : 0);
      }
      this.occ.set(symbol, row);
    }
    // Samples are taken where the SA *value* is a multiple of the stride, so every LF walk hits one
    // within `sampleStride` steps (SA decreases by exactly one per step).
    this.sampled = new Map();
    for (let row = 0; row < this.sa.length; row++) {
      if (this.sa[row] % this.sampleStride === 0) this.sampled.set(row, this.sa[row]);
    }
  }

  get rows() {
    return this.bwt.length;
  }

  lf(i) {
    const symbol = this.bwt[i];
    return this.c.get(symbol) + this.occ.get(symbol)[i];
  }

  /** Backward search over the sentinel index: the SA interval [l, r) of the (shifted) pattern. */
  interval(pattern) {
    let l = 0;
    let r = this.rows;
    for (let i = pattern.length - 1; i >= 0; i--) {
      const symbol = pattern[i] + 1;
      const row = this.occ.get(symbol);
      if (row === undefined) return { l: 1, r: 0, count: 0 };
      l = this.c.get(symbol) + row[l];
      r = this.c.get(symbol) + row[r];
      if (l >= r) return { l: 1, r: 0, count: 0 };
    }
    return { l, r, count: r - l };
  }

  count(pattern) {
    if (pattern.length === 0) return this.n; // documented convention for the empty pattern
    return this.interval(pattern).count;
  }

  /** Every occurrence start, located with the sampled SA values plus LF steps. */
  locate(pattern) {
    if (pattern.length === 0) {
      // The documented convention for the empty pattern: it sits at the n text positions, excluding
      // the sentinel row that the index also stores.
      return { positions: [...Array(this.n).keys()], worstSteps: 0 };
    }
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

  /** The (shifted) suffix text of each row in [l, r), used to verify the search invariant. */
  suffixesIn(l, r) {
    return this.sa.slice(l, r).map((position) => this.indexed.slice(position, position + 8));
  }

  statistics() {
    return { n: this.n, rows: this.rows, symbols: this.symbols.length,
      occEntries: this.symbols.length * (this.rows + 1), samples: this.sampled.size,
      sampleStride: this.sampleStride, cTotal: this.cTotal };
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
    return positions;
  }
  for (let i = 0; i + pattern.length <= text.length; i++) {
    let ok = true;
    for (let j = 0; j < pattern.length; j++) if (text[i + j] !== pattern[j]) ok = false;
    if (ok) positions.push(i);
  }
  return positions;
}

function startsWith(text, position, pattern) {
  if (position + pattern.length > text.length) return false;
  for (let i = 0; i < pattern.length; i++) if (text[position + i] !== pattern[i]) return false;
  return true;
}

function main() {
  const assert = require("node:assert");
  const rng = new Rng(20240522);

  // (1) The suffix array against a brute-force sort, and BWT invertibility.
  {
    for (const text of ["banana", "mississippi", "aaaa", "", "a", "abracadabra", "zzyzx"]) {
      const symbols = bytesOf(text);
      const sa = suffixArray(symbols);
      assert.deepStrictEqual(sa, bruteSuffixArray(symbols), `suffix array of "${text}"`);
      const bwt = burrowsWheeler(symbols, sa);
      assert.strictEqual(bwt.length, symbols.length);
      if (symbols.length > 0) {
        const r0 = sa.indexOf(0);
        assert.ok(r0 >= 0, "the empty suffix position is in the array");
        assert.deepStrictEqual(lfWalkFrom(bwt, r0), symbols,
          `the LF walk started at the row with SA = 0 inverts "${text}" exactly`);
      }
      // Invertibility from row 0 needs the sentinel convention (a unique smallest symbol).
      const sentinel = sentinelText(symbols);
      const sentinelSa = suffixArray(sentinel);
      const sentinelBwt = burrowsWheeler(sentinel, sentinelSa);
      assert.strictEqual(sentinelSa[0], sentinel.length - 1,
        "the suffix that is just the sentinel sorts first");
      assert.strictEqual(sentinelBwt.filter((symbol) => symbol === 0).length, 1,
        "the sentinel occurs exactly once in the BWT column");
      assert.deepStrictEqual(stripSentinel(inverseBurrowsWheeler(sentinelBwt)), symbols,
        `sentinel BWT of "${text}" inverts back to the text`);
    }
    const randomText = [];
    for (let i = 0; i < 400; i++) randomText.push(97 + rng.nextInt(4));
    assert.deepStrictEqual(suffixArray(randomText), bruteSuffixArray(randomText),
      "the doubling suffix array matches the brute-force sort on 400 symbols");
    const randomSentinel = sentinelText(randomText);
    assert.deepStrictEqual(stripSentinel(inverseBurrowsWheeler(
      burrowsWheeler(randomSentinel, suffixArray(randomSentinel)))), randomText);

    // Documented subtleties of the LF mapping, measured rather than asserted from folklore.
    const banana = new FmIndex(bytesOf("banana"));
    assert.strictEqual(lfCycleCount(banana), 1, "the sentinel index has a single LF cycle");
    assert.deepStrictEqual(lfWalkFrom(banana.bwt, banana.sa.indexOf(0)), sentinelText(bytesOf("banana")),
      "started at the row whose suffix is the whole text, the walk returns text + sentinel");
    assert.deepStrictEqual(lfWalkFrom(banana.bwt, 0),
      [0].concat(bytesOf("banana").map((symbol) => symbol + 1)),
    "while row 0 returns the same rotation with the sentinel in front");
    assert.deepStrictEqual(stripSentinel(lfWalkFrom(banana.bwt, 0)), bytesOf("banana"));
    assert.deepStrictEqual(stripSentinel(lfWalkFrom(banana.bwt, banana.sa.indexOf(0))),
      bytesOf("banana"));
    // The plain, sentinel-free BWT determines the text only up to a rotation ...
    const plainSa = suffixArray(bytesOf("banana"));
    const plainBwt = burrowsWheeler(bytesOf("banana"), plainSa);
    assert.deepStrictEqual(lfWalkFrom(plainBwt, plainSa.indexOf(0)), bytesOf("banana"),
      "started at the row whose suffix is the whole text it is exact");
    assert.deepStrictEqual(lfWalkFrom(plainBwt, 0), bytesOf("abanan"),
      "but the row-0 start gives the text rotated by one character");
    // ... and the LF identity itself only holds for the sentinel (rotation-order) column.
    assert.strictEqual(lfIdentityViolations(plainSa, plainBwt), 0,
      "for \"banana\" the plain column happens to satisfy SA[LF(i)] = SA[i] - 1");
    const randomSymbols = [];
    const plainRng = new Rng(4242);
    for (let i = 0; i < 200; i++) randomSymbols.push(97 + plainRng.nextInt(4));
    const randomSa = suffixArray(randomSymbols);
    const randomBwt = burrowsWheeler(randomSymbols, randomSa);
    const violations = lfIdentityViolations(randomSa, randomBwt);
    assert.strictEqual(violations, 45,
      `a plain suffix-array BWT breaks the LF identity on 45 of 200 rows (measured)`);
    const randomSentinelIndex = new FmIndex(randomSymbols);
    assert.strictEqual(lfIdentityViolations(randomSentinelIndex.sa, randomSentinelIndex.bwt), 0,
      "while the sentinel index satisfies it on every row - which is why locate needs a sentinel");
    assert.strictEqual(lfCycleCount(randomSentinelIndex), 1);
    const uniform = new FmIndex(bytesOf("aaaa"));
    assert.strictEqual(lfCycleCount(uniform), 1, "even four identical rotations stay one cycle");
    assert.strictEqual(lfCycleCount(new FmIndex(sentinelText(bytesOf("aaaa")))), 1);
    const plainUniform = burrowsWheeler(bytesOf("aaaa"), suffixArray(bytesOf("aaaa")));
    assert.strictEqual(lfIdentityViolations(suffixArray(bytesOf("aaaa")), plainUniform), 4,
      "the plain column for \"aaaa\" misses the identity on all four rows: every rotation is the " +
      "same string, which is the textbook reason the sentinel exists");
  }

  // (2) Known values.
  {
    const banana = new FmIndex(bytesOf("banana"));
    assert.strictEqual(banana.count(bytesOf("ana")), 2);
    assert.deepStrictEqual(banana.locate(bytesOf("ana")).positions, [1, 3]);
    assert.strictEqual(banana.count(bytesOf("ban")), 1);
    assert.strictEqual(banana.count(bytesOf("nan")), 1);
    assert.strictEqual(banana.count(bytesOf("a")), 3);
    assert.strictEqual(banana.count(bytesOf("bananax")), 0);
    assert.strictEqual(banana.count([]), 6, "the empty pattern matches all n positions");
    assert.deepStrictEqual(banana.locate([]).positions, [0, 1, 2, 3, 4, 5]);
    assert.strictEqual(banana.statistics().cTotal, 7,
      "the sentinel index has n + 1 = 7 rows");

    const mississippi = new FmIndex(bytesOf("mississippi"));
    assert.strictEqual(mississippi.count(bytesOf("issi")), 2);
    assert.deepStrictEqual(mississippi.locate(bytesOf("issi")).positions, [1, 4]);
    assert.strictEqual(mississippi.count(bytesOf("ssi")), 2);
    assert.deepStrictEqual(mississippi.locate(bytesOf("ssi")).positions, [2, 5]);
    assert.strictEqual(mississippi.count(bytesOf("sip")), 1,
      "positions 6, 7, 8 spell s-i-p");
    assert.deepStrictEqual(mississippi.locate(bytesOf("sip")).positions, [6]);
    assert.strictEqual(mississippi.count(bytesOf("isss")), 0);
    assert.strictEqual(mississippi.count(bytesOf("z")), 0);
    assert.deepStrictEqual(mississippi.locate(bytesOf("isss")).positions, []);
    assert.strictEqual(mississippi.count(bytesOf("mississippi")), 1);
    assert.strictEqual(mississippi.count(bytesOf("mississippix")), 0);
    assert.strictEqual(mississippi.count(bytesOf("ppi")), 1);
    assert.strictEqual(mississippi.count(bytesOf("p")), 2);
  }

  // (3) Every substring of "mississippi": counts and the interval invariant.
  {
    const text = "mississippi";
    const symbols = bytesOf(text);
    const index = new FmIndex(symbols);
    let checked = 0;
    for (let start = 0; start < symbols.length; start++) {
      for (let end = start + 1; end <= symbols.length; end++) {
        const pattern = symbols.slice(start, end);
        const found = index.interval(pattern);
        const expected = bruteCount(symbols, pattern);
        assert.strictEqual(found.count, expected,
          `count of "${String.fromCharCode(...pattern)}"`);
        for (let i = found.l; i < found.r; i++) {
          assert.ok(startsWith(symbols, index.sa[i], pattern),
            `SA row ${i} must start with the pattern`);
        }
        if (found.l > 0) {
          assert.ok(!startsWith(symbols, index.sa[found.l - 1], pattern),
            "the row before the interval must not match");
        }
        if (found.r < symbols.length) {
          assert.ok(!startsWith(symbols, index.sa[found.r], pattern),
            "the row after the interval must not match");
        }
        assert.deepStrictEqual(index.locate(pattern).positions, brutePositions(symbols, pattern));
        checked += 1;
      }
    }
    assert.strictEqual(checked, (symbols.length * (symbols.length + 1)) / 2);
  }

  // (4) Random patterns on a 2000-symbol text over a 4-symbol alphabet.
  {
    const symbols = [];
    for (let i = 0; i < 2000; i++) symbols.push(97 + rng.nextInt(4));
    const index = new FmIndex(symbols, { sampleStride: 32 });
    let worstSteps = 0;
    let totalOccurrences = 0;
    for (let trial = 0; trial < 300; trial++) {
      const length = 1 + rng.nextInt(6);
      const pattern = [];
      for (let i = 0; i < length; i++) pattern.push(97 + rng.nextInt(4));
      const expected = bruteCount(symbols, pattern);
      assert.strictEqual(index.count(pattern), expected);
      const located = index.locate(pattern);
      assert.deepStrictEqual(located.positions, brutePositions(symbols, pattern));
      worstSteps = Math.max(worstSteps, located.worstSteps);
      totalOccurrences += expected;
    }
    assert.ok(totalOccurrences > 1000, `patterns were common enough (${totalOccurrences} hits)`);
    assert.ok(worstSteps < 32,
      `locate needed at most ${worstSteps} LF steps, below the documented 32-step guarantee`);
    // A wholly absent pattern and an over-long pattern.
    assert.strictEqual(index.count([200]), 0);
    assert.strictEqual(index.count(new Array(2100).fill(97)), 0);
    assert.strictEqual(index.locate([200]).positions.length, 0);
    // The full text occurs exactly once, at position 0.
    assert.strictEqual(index.count(symbols), 1);
    assert.deepStrictEqual(index.locate(symbols).positions, [0]);
    // Boundary substrings.
    assert.strictEqual(index.count(symbols.slice(0, 3)), bruteCount(symbols, symbols.slice(0, 3)));
    assert.strictEqual(index.count(symbols.slice(-3)), bruteCount(symbols, symbols.slice(-3)));
    assert.ok(index.count(symbols.slice(-3)) >= 1);
  }

  // (5) Structural properties: C, Occ, LF as a permutation, and the sampled SA.
  {
    const symbols = bytesOf("mississippi");
    const index = new FmIndex(symbols, { sampleStride: 4 });
    const statistics = index.statistics();
    assert.strictEqual(statistics.n, 11);
    assert.strictEqual(statistics.rows, 12, "the sentinel adds one row");
    assert.strictEqual(statistics.symbols, 5, "four text symbols plus the sentinel");
    assert.strictEqual(statistics.occEntries, 5 * 13);
    assert.strictEqual(statistics.cTotal, 12);
    let running = 0;
    for (const symbol of index.symbols) {
      assert.strictEqual(index.c.get(symbol), running);
      let count = 0;
      for (const value of index.bwt) if (value === symbol) count += 1;
      running += count;
    }
    assert.strictEqual(running, index.rows, "C values sum to the number of rows");
    for (const symbol of index.symbols) {
      const row = index.occ.get(symbol);
      for (let i = 0; i < row.length - 1; i++) {
        assert.ok(row[i] <= row[i + 1], "Occ rows are non-decreasing");
        assert.ok(row[i + 1] - row[i] <= 1, "and grow by at most one per step");
      }
      let total = 0;
      for (const value of index.bwt) if (value === symbol) total += 1;
      assert.strictEqual(row[index.rows], total, "Occ[c][rows] equals the symbol's frequency");
      assert.strictEqual(row.length, index.rows + 1);
    }
    // LF is a permutation of the n + 1 rows and moves one suffix position backwards.
    const seen = new Set();
    for (let i = 0; i < index.rows; i++) {
      const next = index.lf(i);
      assert.ok(next >= 0 && next < index.rows, "LF stays inside the index");
      assert.ok(!seen.has(next), "LF never maps two rows to the same row");
      seen.add(next);
      assert.strictEqual(index.sa[next], (index.sa[i] - 1 + index.rows) % index.rows,
        `SA[LF(${i})] = SA[${i}] - 1 (mod ${index.rows})`);
    }
    assert.strictEqual(seen.size, index.rows, "LF is a bijection on the rows");
    // A stride of 1 samples every suffix, so locate needs no LF steps at all.
    const dense = new FmIndex(symbols, { sampleStride: 1 });
    assert.strictEqual(dense.locate(bytesOf("issi")).worstSteps, 0);
    assert.deepStrictEqual(dense.locate(bytesOf("issi")).positions, [1, 4]);
    assert.strictEqual(dense.sampled.size, dense.rows, "a stride of one samples every row");
  }

  // (6) Degenerate texts and error paths.
  {
    const empty = new FmIndex([]);
    assert.strictEqual(empty.n, 0);
    assert.strictEqual(empty.count([97]), 0);
    assert.strictEqual(empty.count([]), 0, "the empty pattern on an empty text has no positions");
    assert.deepStrictEqual(empty.locate([97]).positions, []);
    assert.strictEqual(empty.statistics().symbols, 1,
      "an empty text still has the sentinel, so the index has one symbol and one row");
    assert.strictEqual(empty.statistics().rows, 1);

    const single = new FmIndex(bytesOf("x"));
    assert.strictEqual(single.count(bytesOf("x")), 1);
    assert.strictEqual(single.count(bytesOf("y")), 0);
    assert.deepStrictEqual(single.locate(bytesOf("x")).positions, [0]);
    assert.strictEqual(single.count([]), 1);

    const singleSymbol = new FmIndex(bytesOf("aaaaaa"));
    assert.strictEqual(singleSymbol.count(bytesOf("aaa")), 4);
    assert.deepStrictEqual(singleSymbol.locate(bytesOf("aaa")).positions, [0, 1, 2, 3]);
    assert.strictEqual(singleSymbol.count(bytesOf("b")), 0);
    assert.strictEqual(singleSymbol.statistics().symbols, 2,
      "the shifted text symbol plus the sentinel");
    assert.strictEqual(singleSymbol.bwt.filter((symbol) => symbol === 0).length, 1);
    assert.strictEqual(singleSymbol.bwt.filter((symbol) => symbol === 98).length, 6,
      "every other BWT entry is the shifted 'a'");
    const binary = new FmIndex(bytesOf("ababababab"));
    assert.strictEqual(binary.count(bytesOf("abab")), 4,
      "\"abab\" starts at 0, 2, 4 and 6");
    assert.deepStrictEqual(binary.locate(bytesOf("abab")).positions, [0, 2, 4, 6]);
    assert.strictEqual(bruteCount(bytesOf("ababababab"), bytesOf("abab")), 4);
    assert.strictEqual(bruteSuffixArray(bytesOf("ababababab")).length, 10);
    assert.deepStrictEqual(suffixArray(bytesOf("ababababab")), bruteSuffixArray(bytesOf("ababababab")));
    assert.strictEqual(bruteCount(bytesOf("aaaa"), bytesOf("aa")), 3);
  }
}

module.exports = { MASK64, Rng, bytesOf, compareSequences, suffixArray, bruteSuffixArray,
  burrowsWheeler, sentinelText, stripSentinel, inverseBurrowsWheeler, lfPermutation, lfWalkFrom,
  lfIdentityViolations, lfCycleCount, FmIndex, bruteCount, brutePositions, startsWith };

if (require.main === module) {
  main();
  console.log("PASS");
}
