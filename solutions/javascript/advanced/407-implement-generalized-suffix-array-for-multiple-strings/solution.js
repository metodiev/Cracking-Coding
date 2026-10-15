/** Task 407 - Implement a generalized suffix array for multiple strings. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The m strings are concatenated with m *distinct* sentinels: string k gets the character
 * String.fromCharCode(1 + k). All sentinels are below every real (lowercase) character, and
 * they are distinct and ordered by string index, so a suffix from string i always beats the
 * same text extended in string j when i < j. This is the cheapest correct sentinel scheme for
 * the generalized suffix array.
 *
 * Every suffix also carries an owner (which string) and an offset (position inside it), so
 * occurrences of a pattern are found with two binary searches over the suffix array: the
 * pattern-prefixed suffixes form one contiguous range, and each entry of that range maps back
 * to a (string, position) occurrence.
 *
 * Anchors: ["banana", "ana"] -> occurrences of "ana" = (0,1), (0,3), (1,0). Checks: the
 * combined SA validates against a manual comparator, the smallest suffixes are the sentinels
 * in index order, suffixes of each string keep their own plain relative order, and LCG-built
 * pattern searches agree with a naive scan.
 */
"use strict";

function buildGeneralized(strings) {
  const chars = [];
  const owner = [];
  const offset = [];
  const sentinels = [];
  for (let k = 0; k < strings.length; k++) {
    const s = strings[k];
    for (let pos = 0; pos < s.length; pos++) {
      chars.push(s[pos]);
      owner.push(k);
      offset.push(pos);
    }
    sentinels.push(chars.length);
    chars.push(String.fromCharCode(1 + k));
    owner.push(k);
    offset.push(-1);
  }
  const text = chars.join("");
  const sa = suffixArrayDoubling(text);
  return { text, sa, owner, offset, sentinels };
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

function isValidSuffixArray(s, order) {
  if (order.length !== s.length) return false;
  const seen = new Array(s.length).fill(false);
  for (const start of order) {
    if (start < 0 || start >= s.length || seen[start]) return false;
    seen[start] = true;
  }
  for (let i = 0; i + 1 < order.length; i++) {
    if (compareSuffixes(s, order[i], order[i + 1]) >= 0) return false;
  }
  return true;
}

function cmpPrefix(text, start, pattern) {
  for (let i = 0; i < pattern.length; i++) {
    if (start + i >= text.length) return -1;
    const ch = text[start + i];
    if (ch !== pattern[i]) return ch < pattern[i] ? -1 : 1;
  }
  return 0;
}

function patternRange(text, sa, pattern) {
  let lo = 0;
  let hi = sa.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cmpPrefix(text, sa[mid], pattern) < 0) lo = mid + 1;
    else hi = mid;
  }
  const lower = lo;
  hi = sa.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cmpPrefix(text, sa[mid], pattern) <= 0) lo = mid + 1;
    else hi = mid;
  }
  return [lower, lo];
}

function findOccurrences(strings, pattern) {
  const built = buildGeneralized(strings);
  const [lower, upper] = patternRange(built.text, built.sa, pattern);
  const found = [];
  for (let i = lower; i < upper; i++) {
    const idx = built.sa[i];
    if (built.offset[idx] >= 0) found.push([built.owner[idx], built.offset[idx]]);
  }
  found.sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]));
  return found.map((pair) => `${pair[0]}:${pair[1]}`).join(",");
}

function naiveOccurrences(strings, pattern) {
  const found = [];
  for (let k = 0; k < strings.length; k++) {
    const s = strings[k];
    for (let i = 0; i + pattern.length <= s.length; i++) {
      if (s.startsWith(pattern, i)) found.push([k, i]);
    }
  }
  found.sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]));
  return found.map((pair) => `${pair[0]}:${pair[1]}`).join(",");
}

function suffixArrayNaive(s) {
  const idx = Array.from({ length: s.length }, (_, i) => i);
  idx.sort((a, b) => compareSuffixes(s, a, b));
  return idx;
}

function allShortSubstrings(strings, maxLen) {
  const seen = new Set();
  for (const s of strings) {
    for (let i = 0; i < s.length; i++) {
      for (let j = i + 1; j <= Math.min(s.length, i + maxLen); j++) seen.add(s.slice(i, j));
    }
  }
  return [...seen].sort();
}

function lcgString(length, alphabet, seed) {
  const chars = [];
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    chars.push(alphabet[seed % alphabet.length]);
  }
  return [chars.join(""), seed];
}

module.exports = { buildGeneralized, suffixArrayDoubling, isValidSuffixArray, findOccurrences, naiveOccurrences, suffixArrayNaive, allShortSubstrings, lcgString };

if (require.main === module) {
  const assert = require("node:assert");
  const strings = ["banana", "ana"];
  const built = buildGeneralized(strings);
  assert.strictEqual(built.text.length, 6 + 1 + 3 + 1);
  assert.ok(isValidSuffixArray(built.text, built.sa));
  assert.deepStrictEqual([built.sa[0], built.sa[1]], built.sentinels);
  for (let k = 0; k < strings.length; k++) {
    const kept = built.sa.filter((idx) => built.owner[idx] === k && built.offset[idx] >= 0)
      .map((idx) => built.offset[idx]);
    assert.deepStrictEqual(kept, suffixArrayNaive(strings[k]), `relative ${k}`);
  }

  assert.strictEqual(findOccurrences(strings, "ana"), "0:1,0:3,1:0");
  assert.strictEqual(findOccurrences(strings, "a"), "0:1,0:3,0:5,1:0,1:2");
  assert.strictEqual(findOccurrences(strings, "na"), "0:2,0:4,1:1");
  assert.strictEqual(findOccurrences(strings, "ban"), "0:0");
  assert.strictEqual(findOccurrences(strings, "banana"), "0:0");
  assert.strictEqual(findOccurrences(strings, "b"), "0:0");
  assert.strictEqual(findOccurrences(strings, "xyz"), "");
  assert.strictEqual(findOccurrences(strings, "anan"), "0:1");
  assert.strictEqual(findOccurrences(strings, "nanan"), "");
  assert.strictEqual(findOccurrences(strings, "bananas"), "");

  assert.strictEqual(findOccurrences(["abab"], "ab"), "0:0,0:2");
  assert.strictEqual(findOccurrences(["", "ab"], "ab"), "1:0");
  assert.strictEqual(findOccurrences(["", ""], "a"), "");
  assert.strictEqual(findOccurrences([], "a"), "");

  const quad = ["mississippi", "issi", "i", "ppi"];
  for (const pattern of [...allShortSubstrings(quad, 3), "issi", "ississi", "ssippi", "zz", "q", "ippiz"]) {
    assert.strictEqual(findOccurrences(quad, pattern), naiveOccurrences(quad, pattern), pattern);
  }
  assert.strictEqual(findOccurrences(quad, "i"), naiveOccurrences(quad, "i"));
  assert.strictEqual(findOccurrences(quad, "i").split(",").length, 4 + 2 + 1 + 1);

  let seed = 2718;
  for (let trial = 0; trial < 30; trial++) {
    const random = [];
    for (let k = 0; k < 3; k++) {
      let s;
      [s, seed] = lcgString(12 + (seed % 7), "abc", seed);
      random.push(s);
    }
    const randomBuilt = buildGeneralized(random);
    assert.ok(isValidSuffixArray(randomBuilt.text, randomBuilt.sa), random.join("/"));
    for (let k = 0; k < random.length; k++) {
      assert.strictEqual(randomBuilt.sa[k], randomBuilt.sentinels[k], `lcg sentinel ${k}`);
      const kept = randomBuilt.sa
        .filter((idx) => randomBuilt.owner[idx] === k && randomBuilt.offset[idx] >= 0)
        .map((idx) => randomBuilt.offset[idx]);
      assert.deepStrictEqual(kept, suffixArrayNaive(random[k]), `lcg relative ${k}`);
    }
    for (const pattern of [...allShortSubstrings(random, 3), "zzz", "q", "abcabca"]) {
      assert.strictEqual(findOccurrences(random, pattern), naiveOccurrences(random, pattern), pattern);
    }
  }
  console.log("PASS");
}
