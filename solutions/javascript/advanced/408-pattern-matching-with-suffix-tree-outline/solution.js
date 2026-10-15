/** Task 408 - Pattern matching with a suffix tree (outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Backing store: the suffix array plus the LCP array, which represent the same object as a
 * suffix tree - a suffix-tree node corresponds to an LCP interval, and the locus of a pattern
 * is exactly the maximal interval [lo, hi) of suffixes carrying the pattern as prefix. Descent
 * from the root consumes the pattern one character at a time: two binary searches over the
 * suffix array locate the interval, and every suffix inside it is a match, because in a suffix
 * array all pattern-prefixed suffixes are contiguous. The LCP array documents and validates
 * the locus: all suffixes in [lo, hi) share at least len(pattern) prefix characters, so
 * min(lcp[lo+1 .. hi-1]) >= len(pattern).
 *
 * Checks: "banana" anchors, then LCG texts where locate (and its interval width) agree with a
 * naive scan for substrings, mutated and absent patterns. The empty pattern is documented to
 * match every position.
 */
"use strict";

class SuffixTreeIndex {
  constructor(text) {
    this.text = text;
    this.sa = suffixArrayDoubling(text);
    this.lcp = kasaiLcp(text, this.sa);
  }

  cmp(start, pattern) {
    for (let i = 0; i < pattern.length; i++) {
      if (start + i >= this.text.length) return -1;
      const ch = this.text[start + i];
      if (ch !== pattern[i]) return ch < pattern[i] ? -1 : 1;
    }
    return 0;
  }

  interval(pattern) {
    if (pattern.length === 0) return [0, this.sa.length];
    let lo = 0;
    let hi = this.sa.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.cmp(this.sa[mid], pattern) < 0) lo = mid + 1;
      else hi = mid;
    }
    const lower = lo;
    hi = this.sa.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.cmp(this.sa[mid], pattern) <= 0) lo = mid + 1;
      else hi = mid;
    }
    return [lower, lo];
  }

  locate(pattern) {
    const [lower, upper] = this.interval(pattern);
    return this.sa.slice(lower, upper).sort((a, b) => a - b);
  }

  count(pattern) {
    const [lower, upper] = this.interval(pattern);
    return upper - lower;
  }

  locusSharedPrefix(pattern) {
    const [lower, upper] = this.interval(pattern);
    if (upper - lower <= 1) return 0;
    let best = Infinity;
    for (let i = lower + 1; i < upper; i++) best = Math.min(best, this.lcp[i]);
    return best;
  }
}

function suffixArrayDoubling(s) {
  const n = s.length;
  if (n === 0) return [];
  const sa = Array.from({ length: n }, (_, i) => i);
  let rank = Array.from(s, (ch) => ch.charCodeAt(0));
  for (let k = 1; ; k <<= 1) {
    const step = k;
    sa.sort((a, b) => {
      if (rank[a] !== rank[b]) return rank[a] - rank[b];
      const ra = a + step < n ? rank[a + step] : -1;
      const rb = b + step < n ? rank[b + step] : -1;
      return ra - rb;
    });
    const next = new Array(n).fill(0);
    for (let i = 1; i < n; i++) {
      const prev = sa[i - 1];
      const cur = sa[i];
      const pa = prev + step < n ? rank[prev + step] : -1;
      const pb = cur + step < n ? rank[cur + step] : -1;
      next[cur] = next[prev] + (rank[prev] !== rank[cur] || pa !== pb ? 1 : 0);
    }
    rank = next;
    if (rank[sa[n - 1]] === n - 1) return sa;
  }
}

function kasaiLcp(s, sa) {
  const n = s.length;
  const lcp = new Array(n).fill(0);
  const rank = new Array(n).fill(0);
  for (let i = 0; i < n; i++) rank[sa[i]] = i;
  let h = 0;
  for (let i = 0; i < n; i++) {
    const r = rank[i];
    if (r === 0) {
      h = 0;
      continue;
    }
    const j = sa[r - 1];
    while (i + h < n && j + h < n && s[i + h] === s[j + h]) h++;
    lcp[r] = h;
    if (h > 0) h--;
  }
  return lcp;
}

function compareSuffixes(s, i, j) {
  const n = s.length;
  while (i < n && j < n) {
    if (s[i] !== s[j]) return s[i] < s[j] ? -1 : 1;
    i++;
    j++;
  }
  if (i === j) return 0;
  return i === n ? -1 : 1;
}

function suffixArrayNaive(s) {
  const idx = Array.from({ length: s.length }, (_, i) => i);
  idx.sort((a, b) => compareSuffixes(s, a, b));
  return idx;
}

function naiveLocate(text, pattern) {
  if (pattern.length === 0) return Array.from({ length: text.length }, (_, i) => i);
  const positions = [];
  for (let i = 0; i + pattern.length <= text.length; i++) {
    if (text.startsWith(pattern, i)) positions.push(i);
  }
  return positions;
}

function lcgString(length, alphabet, seed) {
  const chars = [];
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    chars.push(alphabet[seed % alphabet.length]);
  }
  return [chars.join(""), seed];
}

function checkIndex(text, extraPatterns) {
  const assert = require("node:assert");
  const index = new SuffixTreeIndex(text);
  assert.deepStrictEqual(index.sa, suffixArrayNaive(text), text);
  const patterns = [...extraPatterns];
  for (let i = 0; i < text.length; i++) {
    for (let j = i + 1; j <= Math.min(text.length, i + 5); j++) patterns.push(text.slice(i, j));
  }
  patterns.push("zz", "qq", "!");
  for (const pattern of patterns) {
    assert.deepStrictEqual(index.locate(pattern), naiveLocate(text, pattern), `${text}/${pattern}`);
    assert.strictEqual(index.count(pattern), index.locate(pattern).length, pattern);
    const [lower, upper] = index.interval(pattern);
    for (let k = lower; k < upper; k++) {
      if (pattern) assert.strictEqual(text.slice(index.sa[k], index.sa[k] + pattern.length), pattern, pattern);
    }
    if (pattern && lower > 0) {
      const prev = index.sa[lower - 1];
      assert.notStrictEqual(text.slice(prev, prev + pattern.length), pattern, pattern);
    }
    if (pattern && upper < index.sa.length) {
      const nxt = index.sa[upper];
      assert.notStrictEqual(text.slice(nxt, nxt + pattern.length), pattern, pattern);
    }
    if (pattern && upper - lower >= 2) assert.ok(index.locusSharedPrefix(pattern) >= pattern.length, pattern);
  }
}

module.exports = { SuffixTreeIndex, suffixArrayDoubling, kasaiLcp, suffixArrayNaive, naiveLocate, lcgString, checkIndex };

if (require.main === module) {
  const assert = require("node:assert");
  checkIndex("banana", ["ana", "na", "a", "banana", "anana", "bananas", "nan"]);
  const banana = new SuffixTreeIndex("banana");
  assert.deepStrictEqual(banana.locate("ana"), [1, 3]);
  assert.deepStrictEqual(banana.locate("na"), [2, 4]);
  assert.deepStrictEqual(banana.locate("a"), [1, 3, 5]);
  assert.deepStrictEqual(banana.locate("banana"), [0]);
  assert.deepStrictEqual(banana.locate("anana"), [1]);
  assert.deepStrictEqual(banana.locate(""), [0, 1, 2, 3, 4, 5]);
  assert.deepStrictEqual(banana.locate("nana"), [2]);
  assert.deepStrictEqual(banana.locate("bananas"), []);
  assert.deepStrictEqual(banana.interval("ana"), [1, 3]);
  assert.ok(banana.locusSharedPrefix("ana") >= 3);

  checkIndex("mississippi", ["issi", "iss", "ssi", "mississippi", "ppi", "sipp"]);
  checkIndex("", ["a"]);
  checkIndex("a", ["a", "aa"]);
  checkIndex("aaaa", ["aaa", "aa", "a", "aaaaa"]);

  let seed = 2718;
  for (let trial = 0; trial < 30; trial++) {
    let text;
    [text, seed] = lcgString(30, "abc", seed);
    const mutations = [];
    for (let k = 0; k < 6; k++) mutations.push(text.slice(0, k) + "d" + text.slice(k + 1));
    checkIndex(text, mutations);
  }
  for (let trial = 0; trial < 10; trial++) {
    let text;
    [text, seed] = lcgString(60, "ab", seed);
    checkIndex(text, [text, text.slice(0, -1), "z" + text]);
  }
  console.log("PASS");
}
