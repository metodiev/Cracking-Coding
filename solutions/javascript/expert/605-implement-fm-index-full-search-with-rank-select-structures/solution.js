/** Task 605 - Implement FM-index full search with rank/select structures. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The FM-index is the BWT of text$ (sentinel appended) plus the cumulative table C and a rank/select
 * directory, which supports pattern search *without* the suffix array: backward search over the
 * interval [l, r) applies l = C[c] + rank(c, l), r = C[c] + rank(c, r) from right to left. Occurrences
 * are recovered by walking lf(row) = C[BWT[row]] + rank(BWT[row], row) until a sampled row is reached,
 * which takes at most sampleRate - 1 steps because LF decrements the suffix position by one.
 *
 * Rank/select structures: one bit vector per character, packed into 32-bit words with popcount, plus a
 * two-level directory - a checkpoint of the running count every 128 bits (Rank9-style) and, for
 * select, the position of every 64th one.
 *
 * Checks: rank verified exhaustively against brute force, select round trips, frequency totals,
 * backward-search counts and located positions against brute force, boundary cases (empty pattern,
 * pattern longer than the text, unary text) and a 4000-character text with 60 random patterns. */
"use strict";

const SENTINEL = "\u0000";
const CHECKPOINT_WORDS = 4; // checkpoint every 128 bits
const SELECT_SAMPLE_ONES = 64;

function popcount32(x) {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
}

/** Bit mask keeping the `count` lowest bits of a word (0..32). */
function lowMask(count) {
  return count === 0 ? 0 : 0xffffffff >>> (32 - count);
}

class FMIndex {
  constructor(text, sampleRate = 32) {
    if (text.includes(SENTINEL)) throw new Error("text must not contain the sentinel character");
    this.body = text;
    this.sampleRate = Math.max(1, sampleRate);
    const terminated = text + SENTINEL;
    this.N = terminated.length;
    const sa = suffixArrayDoubling(terminated);
    this.bwt = new Array(this.N);
    for (let row = 0; row < this.N; row++) this.bwt[row] = terminated[(sa[row] - 1 + this.N) % this.N];

    this.chars = Array.from(new Set([...terminated])).sort();
    this.charIndex = new Map(this.chars.map((ch, i) => [ch, i]));

    const wordCount = Math.ceil(this.N / 32);
    this.words = this.chars.map(() => new Uint32Array(wordCount));
    this.selectSamples = this.chars.map(() => []);
    this.ones = new Array(this.chars.length).fill(0);
    for (let row = 0; row < this.N; row++) {
      const c = this.charIndex.get(this.bwt[row]);
      this.words[c][row >>> 5] |= 1 << (row & 31);
      // Every SELECT_SAMPLE_ONES-th one of a character is recorded for select.
      if (this.ones[c] % SELECT_SAMPLE_ONES === 0) this.selectSamples[c].push(row);
      this.ones[c]++;
    }
    this.checkpoints = this.chars.map(() => new Int32Array(Math.ceil(wordCount / CHECKPOINT_WORDS) + 1));
    for (let c = 0; c < this.chars.length; c++) {
      let total = 0;
      for (let word = 0; word < wordCount; word++) {
        if (word % CHECKPOINT_WORDS === 0) this.checkpoints[c][word / CHECKPOINT_WORDS] = total;
        total += popcount32(this.words[c][word]);
      }
      this.checkpoints[c][Math.ceil(wordCount / CHECKPOINT_WORDS)] = total;
    }
    // C[c] counts the suffixes starting with a character smaller than c, i.e. the ones before it.
    this.C = this.chars.map((_, i) => this.ones.slice(0, i).reduce((acc, count) => acc + count, 0));

    this.sampledRows = new Set();
    this.sampledValues = new Map();
    for (let row = 0; row < this.N; row++) {
      if (sa[row] % this.sampleRate === 0) {
        this.sampledRows.add(row);
        this.sampledValues.set(row, sa[row]);
      }
    }
    this.sampleCount = this.sampledRows.size;
  }

  /** Occurrences of character index `c` in bwt[0..i): checkpoint + popcounts + masked word. */
  rankIndex(c, i) {
    if (i <= 0) return 0;
    const word = i >>> 5;
    const block = word / CHECKPOINT_WORDS | 0;
    let total = this.checkpoints[c][block];
    for (let w = block * CHECKPOINT_WORDS; w < word; w++) total += popcount32(this.words[c][w]);
    total += popcount32(this.words[c][word] & lowMask(i & 31));
    return total;
  }

  rank(ch, i) {
    const c = this.charIndex.get(ch);
    return c === undefined ? 0 : this.rankIndex(c, i);
  }

  /** Position of the (k+1)-th occurrence of character index `c`, scanning from the nearest sample. */
  selectIndex(c, k) {
    if (k < 0 || k >= this.ones[c]) return -1;
    const words = this.words[c];
    const wordCount = words.length;
    let word = this.selectSamples[c][k / SELECT_SAMPLE_ONES | 0] >>> 5;
    let seen = this.rankIndex(c, word << 5);
    while (word < wordCount) {
      const bits = words[word];
      if (seen + popcount32(bits) > k) {
        let copy = bits;
        while (copy !== 0) {
          const bit = copy & -copy;
          if (seen === k) return (word << 5) + 31 - Math.clz32(bit >>> 0);
          seen++;
          copy ^= bit;
        }
      }
      seen += popcount32(bits);
      word++;
    }
    return -1;
  }

  select(ch, k) {
    const c = this.charIndex.get(ch);
    return c === undefined ? -1 : this.selectIndex(c, k);
  }

  lfRow(row) {
    const c = this.charIndex.get(this.bwt[row]);
    return this.C[c] + this.rankIndex(c, row);
  }

  /** Backward search over the pattern; returns the SA interval [l, r) or null. */
  search(pattern) {
    let l = 0;
    let r = this.N;
    for (let i = pattern.length - 1; i >= 0; i--) {
      const c = this.charIndex.get(pattern[i]);
      if (c === undefined) return null;
      l = this.C[c] + this.rankIndex(c, l);
      r = this.C[c] + this.rankIndex(c, r);
      if (l >= r) return null;
    }
    return { l, r };
  }

  count(pattern) {
    const interval = this.search(pattern);
    return interval === null ? 0 : interval.r - interval.l;
  }

  /** Walks LF until a sampled row, reporting that row and the number of steps taken. */
  walkToSample(row) {
    let steps = 0;
    let current = row;
    while (!this.sampledRows.has(current)) {
      current = this.lfRow(current);
      steps++;
    }
    return { row: current, steps };
  }

  suffixAt(row) {
    const hit = this.walkToSample(row);
    return (this.sampledValues.get(hit.row) + hit.steps) % this.N;
  }

  /** Occurrence positions of `pattern` in text order; `limit` caps how many are returned. */
  locate(pattern, limit = Infinity) {
    const interval = this.search(pattern);
    if (interval === null) return [];
    const positions = [];
    for (let row = interval.l; row < interval.r; row++) {
      const start = this.suffixAt(row);
      if (start < this.body.length) positions.push(start);
    }
    positions.sort((a, b) => a - b);
    return positions.slice(0, limit);
  }

  /**
   * Longest prefix of `query` that occurs in the text. "Prefix of length L occurs" is monotone in L,
   * so the length is found by a binary search whose probes are ordinary backward searches.
   */
  longestMatch(query) {
    const occurs = (length) => this.search(query.slice(0, length)) !== null;
    let low = 0;
    let high = query.length;
    while (low < high) {
      const mid = (low + high + 1) >> 1;
      if (occurs(mid)) low = mid;
      else high = mid - 1;
    }
    const interval = low === 0 ? { l: 0, r: this.N } : this.search(query.slice(0, low));
    return { length: low, pattern: query.slice(0, low), count: interval.r - interval.l };
  }

  /** Worst-case LF steps spent locating an interval; must stay below the sampling rate. */
  locateStepBound(pattern) {
    const interval = this.search(pattern);
    if (interval === null) return 0;
    let worst = 0;
    for (let row = interval.l; row < interval.r; row++) {
      worst = Math.max(worst, this.walkToSample(row).steps);
    }
    return worst;
  }

  indexSize() {
    const words = this.chars.length * Math.ceil(this.N / 32) * 4;
    const ranks = this.chars.length * this.checkpoints[0].length * 4;
    const selects = this.selectSamples.reduce((acc, samples) => acc + samples.length * 4, 0);
    return words + ranks + selects + 4 * this.sampleCount;
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

/** Brute-force rank of a character in a packed array prefix, mirroring rank(ch, i). */
function bruteRank(sequence, ch, i) {
  let count = 0;
  for (let k = 0; k < i; k++) if (sequence[k] === ch) count++;
  return count;
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

module.exports = {
  FMIndex, popcount32, lowMask, suffixArrayDoubling, bruteOccurrences, bruteRank, lcgString, SENTINEL,
};

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(popcount32(0) + popcount32(0b101101), 4);
  assert.ok(popcount32(0xffffffff >>> 0) === 32 && lowMask(0) === 0 && (lowMask(32) >>> 0) === 0xffffffff);

  const empty = new FMIndex("");
  assert.ok(empty.N === 1 && empty.count("a") === 0 && empty.locate("a").length === 0);

  const unary = new FMIndex("a".repeat(300), 16);
  assert.ok(unary.count("a") === 300 && unary.count("") === unary.N && unary.count("a".repeat(301)) === 0);
  assert.deepStrictEqual([unary.select("a", 0), unary.select("a", 299), unary.select("a", 300)], [0, 299, -1]);
  assert.ok(unary.rank("a", 150) === 150 && unary.locate("a", 3).join(",") === "0,1,2");

  for (const text of ["banana", "mississippi", "abracadabra", "abab", "aaa", "abcabc"]) {
    const index = new FMIndex(text, 5);
    for (let c = 0; c < index.chars.length; c++) {
      const ch = index.chars[c];
      assert.strictEqual(index.ones[c], bruteRank(index.bwt, ch, index.N), `${text} ${ch}`);
      for (let i = 0; i <= index.N; i++) {
        assert.strictEqual(index.rank(ch, i), bruteRank(index.bwt, ch, i), `${text} rank ${ch} ${i}`);
      }
    }
    assert.strictEqual(index.ones.reduce((a, b) => a + b, 0), index.N);
    assert.ok(index.rank("z", 4) === 0 && index.select("z", 0) === -1);
    for (let start = 0; start < text.length; start++) {
      for (let end = start + 1; end <= text.length; end++) {
        const pattern = text.slice(start, end);
        assert.deepStrictEqual(index.locate(pattern), bruteOccurrences(text, pattern), `${text} ${pattern}`);
        assert.strictEqual(index.count(pattern), bruteOccurrences(text, pattern).length, `${text} ${pattern}`);
        assert.ok(index.locateStepBound(pattern) < 5, `${text} ${pattern}`);
      }
    }
  }

  const banana = new FMIndex("banana", 2);
  assert.deepStrictEqual([banana.locate("ana"), banana.locate("a")], [[1, 3], [1, 3, 5]]);
  assert.ok(banana.count("bananas") === 0 && banana.longestMatch("bananas").length === 6);
  assert.deepStrictEqual([banana.longestMatch("ananas"), banana.longestMatch("xbanana")],
    [{ length: 5, pattern: "anana", count: 1 }, { length: 0, pattern: "", count: banana.N }]);

  let seed = 605;
  for (const alphabet of ["ab", "abc", "abcdefghij"]) {
    for (let trial = 0; trial < 8; trial++) {
      let text;
      [text, seed] = lcgString(40, alphabet, seed);
      const rate = 1 + (trial % 9);
      const index = new FMIndex(text, rate);
      for (const ch of index.chars) {
        const total = index.ones[index.charIndex.get(ch)];
        for (let k = 0; k < total; k += 7) assert.strictEqual(index.bwt[index.select(ch, k)], ch, text);
      }
      for (let start = 0; start < text.length; start += 3) {
        for (let end = start + 1; end <= Math.min(start + 8, text.length); end++) {
          const pattern = text.slice(start, end);
          assert.deepStrictEqual(index.locate(pattern), bruteOccurrences(text, pattern), `${text} ${pattern}`);
          assert.ok(index.locateStepBound(pattern) < rate, `${text} ${pattern} steps`);
        }
      }
      for (const absent of [`z${text}`, text + "z", alphabet.repeat(text.length)]) {
        assert.ok(index.count(absent) === 0 && index.locate(absent).length === 0, `${text} ${absent}`);
      }
      assert.strictEqual(index.longestMatch(text).length, text.length);
    }
  }

  let bigText;
  [bigText, seed] = lcgString(4000, "ab", seed);
  const big = new FMIndex(bigText, 64);
  assert.ok(big.N === 4001 && big.sampleCount === Math.ceil(4001 / 64));  for (let probe = 0; probe < 60; probe++) {
    const start = (probe * 61) % (bigText.length - 20);
    const pattern = bigText.slice(start, start + 6 + (probe % 9));
    assert.deepStrictEqual(big.locate(pattern), bruteOccurrences(bigText, pattern), pattern);
    assert.strictEqual(big.count(pattern), bruteOccurrences(bigText, pattern).length, pattern);
  }
  assert.ok(big.count("") === big.N && big.count("a".repeat(4000) + "b") === 0);
  assert.strictEqual(big.rank("a", big.N) + big.rank("b", big.N) + big.rank(SENTINEL, big.N), big.N);

  console.log(`fm-index: ${big.N} rows, ${big.chars.length} characters, ` +
    `${big.sampleCount} sampled suffixes, ${big.indexSize()} bytes`);
  console.log("PASS");
}
