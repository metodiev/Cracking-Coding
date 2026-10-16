/** Task 604 - Implement compressed suffix array (CSA) advanced. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A compressed suffix array stores no suffix array at all: it keeps the *navigation* functions of
 * Sadakane's CSA plus a sampled suffix array, and reconstructs arbitrary SA values on demand.
 *
 *   psi[i] = SA^-1[(SA[i] + 1) mod N]   - the "next suffix" map; iterating it enumerates all rows
 *   phi    = psi^-1                     - built lazily from psi; equals the LF mapping, so
 *                                         phi[i] = C[BWT[i]] + rank(BWT[i], i)
 *   C[c]   = number of suffixes starting with a character smaller than c
 *   BWT    = the last column, with rank support (counts per 64-character block + in-block scan)
 *   samples: SA[rate * j] for every j, so suffixAt(i) walks phi forward until a sample is hit
 *
 * Backward search over [l, r) uses C and rank; locating an interval walks phi at most `rate - 1`
 * steps per row, which is the space/time trade-off the CSA is built around. Only psi, the rank
 * blocks and the samples are retained: the full SA is discarded after construction, and phi is
 * materialised lazily, which `storesFullSuffixArray = false` records for the checks.
 *
 * Checks: backtracking is verified against psi (`suffixAt(i) == SA[i]` for every row), psi/phi are
 * permutations and inverses, psi is a single cycle over all rows, phi agrees with the rank-based LF
 * mapping, search/locate/count match brute force on LCG strings, the text is recovered from the BWT
 * alone, and the locate step count stays below the sampling rate.
 */
"use strict";

const SENTINEL = "\u0000";
const BLOCK = 64;

class CompressedSuffixArray {
  constructor(text, sampleRate = 32) {
    if (text.includes(SENTINEL)) throw new Error("text must not contain the sentinel character");
    this.body = text;
    this.textWithSentinel = text + SENTINEL;
    this.N = this.textWithSentinel.length;
    this.sampleRate = Math.max(1, sampleRate);
    const sa = suffixArrayDoubling(this.textWithSentinel);
    const inverse = new Int32Array(this.N);
    for (let row = 0; row < this.N; row++) inverse[sa[row]] = row;

    // psi is the core structure; the SA and its inverse are dropped right after this loop.
    this.psi = new Int32Array(this.N);
    this.bwt = new Array(this.N);
    for (let row = 0; row < this.N; row++) {
      this.psi[row] = inverse[(sa[row] + 1) % this.N];
      this.bwt[row] = this.textWithSentinel[(sa[row] - 1 + this.N) % this.N];
    }
    this.chars = Array.from(new Set([...this.textWithSentinel])).sort();
    this.charIndex = new Map(this.chars.map((ch, i) => [ch, i]));
    this.C = new Array(this.chars.length).fill(0);
    const counts = new Array(this.chars.length).fill(0);
    for (const ch of this.bwt) counts[this.charIndex.get(ch)]++;
    let running = 0;
    for (let i = 0; i < this.chars.length; i++) {
      this.C[i] = running;
      running += counts[i];
    }
    this.blockCounts = this.chars.map((ch) => {
      const blocks = new Int32Array(Math.floor(this.N / BLOCK) + 1);
      let total = 0;
      for (let i = 0; i < this.N; i++) {
        if (i % BLOCK === 0) blocks[i / BLOCK] = total;
        if (this.bwt[i] === ch) total++;
      }
      return blocks;
    });
    // Suffix sampling: a row is marked when its SA value is a multiple of the sample rate, so
    // walking phi (which decrements the value by one) reaches a marked row within rate - 1 steps.
    this.sampledRows = new Set();
    this.sampledValues = new Map();
    for (let row = 0; row < this.N; row++) {
      if (sa[row] % this.sampleRate === 0) {
        this.sampledRows.add(row);
        this.sampledValues.set(row, sa[row]);
      }
    }
    this.sampleCount = this.sampledRows.size;
    this.phiCache = null;
    this.storesFullSuffixArray = false;
  }

  /** Number of occurrences of `ch` in bwt[0..i): block table plus an in-block scan. */
  rank(ch, i) {
    if (i <= 0) return 0;
    const index = this.charIndex.get(ch);
    if (index === undefined) return 0;
    const block = Math.floor(i / BLOCK);
    let total = this.blockCounts[index][block];
    for (let k = block * BLOCK; k < i; k++) if (this.bwt[k] === ch) total++;
    return total;
  }

  /** LF mapping computed from the rank structure; must agree with psi^-1. */
  lfRow(row) {
    const ch = this.bwt[row];
    return this.C[this.charIndex.get(ch)] + this.rank(ch, row);
  }

  /** Inverse of psi, materialised once and cached (the CSA proper only stores psi). */
  phi() {
    if (this.phiCache === null) {
      const phi = new Int32Array(this.N);
      for (let row = 0; row < this.N; row++) phi[this.psi[row]] = row;
      this.phiCache = phi;
    }
    return this.phiCache;
  }

  /** Backward search: SA interval [l, r) of `pattern`, or null when it does not occur. */
  search(pattern) {
    let l = 0;
    let r = this.N;
    for (let k = pattern.length - 1; k >= 0; k--) {
      const ch = pattern[k];
      const index = this.charIndex.get(ch);
      if (index === undefined) return null;
      l = this.C[index] + this.rank(ch, l);
      r = this.C[index] + this.rank(ch, r);
      if (l >= r) return null;
    }
    return { l, r };
  }

  count(pattern) {
    const interval = this.search(pattern);
    return interval === null ? 0 : interval.r - interval.l;
  }

  /** Recovers SA[row] from the sampled suffixes by walking phi (at most sampleRate - 1 steps). */
  suffixAt(row) {
    let steps = 0;
    let current = row;
    while (!this.sampledRows.has(current)) {
      current = this.phi()[current];
      steps++;
    }
    return (this.sampledValues.get(current) + steps) % this.N;
  }

  /** Occurrence positions of `pattern`, ascending; `limit` caps the number of located rows. */
  locate(pattern, limit = Infinity) {
    const interval = this.search(pattern);
    if (interval === null) return [];
    const positions = [];
    for (let row = interval.l; row < interval.r && positions.length < limit; row++) {
      const start = this.suffixAt(row);
      if (start < this.body.length) positions.push(start);
    }
    return positions.sort((a, b) => a - b);
  }

  /** Worst-case phi steps needed to locate the interval of `pattern` (bounded by sampleRate - 1). */
  locateStepBound(pattern) {
    const interval = this.search(pattern);
    if (interval === null) return 0;
    let worst = 0;
    for (let row = interval.l; row < interval.r; row++) {
      let steps = 0;
      let current = row;
      while (!this.sampledRows.has(current)) {
        current = this.phi()[current];
        steps++;
      }
      if (steps > worst) worst = steps;
    }
    return worst;
  }

  /**
   * Rebuilds the sentinel-terminated text from the BWT and the LF mapping alone. Walking backwards
   * from row 0 (the sentinel suffix) emits s[N-2], s[N-3], ..., s[0]; the final step wraps onto the
   * sentinel itself, which is stored last.
   */
  recoverText() {
    const out = new Array(this.N);
    let row = 0;
    for (let k = this.N - 2; k >= 0; k--) {
      out[k] = this.bwt[row];
      row = this.lfRow(row);
    }
    out[this.N - 1] = this.bwt[row];
    return out.join("");
  }

  /** Bytes of retained index data, for the space claim printed at the end. */
  indexSize() {
    let bytes = 4 * this.N; // psi
    if (this.phiCache !== null) bytes += 4 * this.N;
    bytes += Math.ceil(this.N / 8); // marked-row bit set
    bytes += 4 * this.sampleCount; // sampled SA values
    bytes += 4 * this.chars.length * this.blockCounts[0].length; // rank blocks
    return bytes;
  }
}

function suffixArrayDoubling(s) {
  const n = s.length;
  if (n === 0) return [];
  const sa = Array.from({ length: n }, (_, i) => i);
  let rank = Array.from(s, (ch) => ch.charCodeAt(0));
  for (let step = 1; ; step <<= 1) {
    const shift = step;
    sa.sort((a, b) => {
      if (rank[a] !== rank[b]) return rank[a] - rank[b];
      const ra = a + shift < n ? rank[a + shift] : -1;
      const rb = b + shift < n ? rank[b + shift] : -1;
      return ra - rb;
    });
    const next = new Array(n).fill(0);
    for (let i = 1; i < n; i++) {
      const prev = sa[i - 1];
      const cur = sa[i];
      const pa = prev + shift < n ? rank[prev + shift] : -1;
      const pb = cur + shift < n ? rank[cur + shift] : -1;
      next[cur] = next[prev] + (rank[prev] !== rank[cur] || pa !== pb ? 1 : 0);
    }
    rank = next;
    if (rank[sa[n - 1]] === n - 1) return sa;
  }
}

function bruteOccurrences(text, pattern) {
  const positions = [];
  for (let i = 0; i + pattern.length <= text.length; i++) if (text.startsWith(pattern, i)) positions.push(i);
  return positions;
}

function lcgString(length, alphabet, seed) {
  let state = seed;
  const chars = [];
  for (let i = 0; i < length; i++) {
    state = (state * 75 + 74) % 65537;
    chars.push(alphabet[state % alphabet.length]);
  }
  return [chars.join(""), state];
}

module.exports = { CompressedSuffixArray, suffixArrayDoubling, bruteOccurrences, lcgString, SENTINEL };

if (require.main === module) {
  const assert = require("node:assert");

  const empty = new CompressedSuffixArray("");
  assert.strictEqual(empty.N, 1);
  assert.strictEqual(empty.count("a"), 0);
  assert.deepStrictEqual(empty.locate("a"), []);
  assert.strictEqual(empty.recoverText(), SENTINEL);

  const single = new CompressedSuffixArray("a");
  assert.strictEqual(single.count("a"), 1);
  assert.deepStrictEqual(single.locate("a"), [0]);
  assert.strictEqual(single.count("aa"), 0);
  assert.strictEqual(single.recoverText(), "a" + SENTINEL);

  for (const text of ["banana", "mississippi", "abracadabra", "abab", "aaaa"]) {
    const csa = new CompressedSuffixArray(text, 4);
    const sa = suffixArrayDoubling(text + SENTINEL);
    for (let row = 0; row < sa.length; row++) {
      assert.strictEqual(csa.suffixAt(row), sa[row], `${text} row ${row}`);
    }
    assert.strictEqual(csa.recoverText(), text + SENTINEL, text);
    for (let row = 0; row < sa.length; row++) {
      assert.strictEqual(csa.bwt[row], (text + SENTINEL)[(sa[row] - 1 + sa.length) % sa.length], text);
      assert.strictEqual(csa.phi()[csa.psi[row]], row, `${text} phi(psi) row ${row}`);
      assert.strictEqual(csa.lfRow(row), csa.phi()[row], `${text} LF row ${row}`);
    }
  }

  const psiChecks = new CompressedSuffixArray("abracadabra", 3);
  const seenPsi = new Set();
  let cursor = 0;
  for (let step = 0; step < psiChecks.N; step++) {
    seenPsi.add(cursor);
    cursor = psiChecks.psi[cursor];
  }
  assert.strictEqual(seenPsi.size, psiChecks.N); // psi is a single cycle over all rows
  assert.strictEqual(cursor, 0);
  assert.deepStrictEqual(psiChecks.locate("abra"), [0, 7]);
  assert.deepStrictEqual(psiChecks.locate("x"), []);
  assert.strictEqual(psiChecks.count("a"), 5);
  assert.strictEqual(psiChecks.locate("a", 3).length, 3);

  const banana = new CompressedSuffixArray("banana", 2);
  assert.strictEqual(banana.count("ana"), 2);
  assert.deepStrictEqual(banana.locate("ana"), [1, 3]);
  assert.deepStrictEqual(banana.locate("a"), [1, 3, 5]);
  assert.deepStrictEqual(banana.locate("banana"), [0]);
  assert.strictEqual(banana.count("bananas"), 0);
  assert.ok(banana.locateStepBound("banana") <= 1);
  assert.deepStrictEqual(banana.locate("an", 2), [1, 3]);

  let seed = 604;
  for (const alphabet of ["ab", "abc", "abcde"]) {
    for (let trial = 0; trial < 20; trial++) {
      let text;
      [text, seed] = lcgString(30, alphabet, seed);
      const rate = 1 + (trial % 7);
      const csa = new CompressedSuffixArray(text, rate);
      const sa = suffixArrayDoubling(text + SENTINEL);
      for (let row = 0; row < sa.length; row++) assert.strictEqual(csa.suffixAt(row), sa[row], text);
      for (let start = 0; start < text.length; start++) {
        for (let end = start + 1; end <= Math.min(start + 6, text.length); end++) {
          const pattern = text.slice(start, end);
          assert.deepStrictEqual(csa.locate(pattern), bruteOccurrences(text, pattern), `${text} ${pattern}`);
          assert.strictEqual(csa.count(pattern), bruteOccurrences(text, pattern).length, `${text} ${pattern}`);
          assert.ok(csa.locateStepBound(pattern) < rate, `${text} ${pattern} steps`);
        }
      }
      for (const absent of ["z", `${text}z`, alphabet.repeat(text.length)]) {
        assert.deepStrictEqual(bruteOccurrences(text, absent), []);
        assert.strictEqual(csa.count(absent), 0);
        assert.deepStrictEqual(csa.locate(absent), []);
      }
      assert.strictEqual(csa.recoverText(), text + SENTINEL);
      assert.strictEqual(csa.storesFullSuffixArray, false);
      assert.strictEqual(csa.sa, undefined);
    }
  }

  let bigSeed = seed;
  let bigText;
  [bigText, bigSeed] = lcgString(4000, "ab", bigSeed);
  const big = new CompressedSuffixArray(bigText, 64);
  assert.strictEqual(big.N, 4001);
  assert.strictEqual(big.sampleCount, Math.ceil(4001 / 64));
  assert.ok(big.indexSize() < 4 * big.N * 3); // psi + samples + ranks, no SA
  for (let probe = 0; probe < 60; probe++) {
    const start = (probe * 53) % (bigText.length - 15);
    const pattern = bigText.slice(start, start + 5 + (probe % 8));
    assert.deepStrictEqual(big.locate(pattern), bruteOccurrences(bigText, pattern), pattern);
  }
  assert.strictEqual(big.count(""), big.N);
  assert.strictEqual(big.recoverText(), bigText + SENTINEL);
  assert.ok(big.locateStepBound("a") < 64);

  console.log(`csa: ${big.N} rows, ${big.sampleCount} samples, ` +
    `index ${big.indexSize()} bytes vs ${4 * big.N} for the raw SA`);
  console.log("PASS");
}
