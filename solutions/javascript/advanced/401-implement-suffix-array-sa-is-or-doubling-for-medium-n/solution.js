/** Task 401 - Implement suffix array (SA-IS or doubling) for medium N. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * `suffixArrayDoubling` sorts the suffixes by their first 2^k characters, doubling k until
 * every rank is unique: O(n log^2 n) with comparison sorting. No sentinel is appended - a
 * position past the end of the string ranks below every character, which resolves "one suffix
 * is a prefix of the other" correctly because the shorter suffix must win.
 *
 * Correctness is pinned by two independent oracles: a naive suffix sort and the hand-written
 * character-by-character comparator. Property tests include medium N (a few hundred
 * characters) so the doubling loop really runs several rounds.
 */
"use strict";

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

function suffixArrayNaive(s) {
  const idx = Array.from({ length: s.length }, (_, i) => i);
  idx.sort((a, b) => compareSuffixes(s, a, b));
  return idx;
}

function isValidSuffixArray(s, order) {
  const n = s.length;
  if (order.length !== n) return false;
  const seen = new Array(n).fill(false);
  for (const start of order) {
    if (start < 0 || start >= n || seen[start]) return false;
    seen[start] = true;
  }
  for (let i = 0; i + 1 < n; i++) {
    if (compareSuffixes(s, order[i], order[i + 1]) >= 0) return false;
  }
  return true;
}

function lcgString(length, alphabet, seed) {
  const chars = [];
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    chars.push(alphabet[seed % alphabet.length]);
  }
  return [chars.join(""), seed];
}

function repeat(unit, times) {
  return unit.repeat(times);
}

module.exports = {
  suffixArrayDoubling, compareSuffixes, suffixArrayNaive, isValidSuffixArray, lcgString,
};

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(suffixArrayDoubling("banana").join(","), "5,3,1,0,4,2");
  assert.strictEqual(suffixArrayDoubling("mississippi").join(","), "10,7,4,1,0,9,8,6,3,5,2");

  assert.deepStrictEqual(suffixArrayDoubling(""), []);
  assert.deepStrictEqual(suffixArrayDoubling("a"), [0]);
  assert.deepStrictEqual(suffixArrayDoubling("aaaa"), [3, 2, 1, 0]);
  assert.deepStrictEqual(suffixArrayDoubling("abab"), [2, 0, 3, 1]);
  assert.deepStrictEqual(suffixArrayDoubling("abracadabra"), suffixArrayNaive("abracadabra"));

  const small = ["", "a", "b", "aa", "ab", "ba", "aaaa", "abab", "banana", "mississippi", "abracadabra"];
  for (const s of small) {
    const order = suffixArrayDoubling(s);
    assert.deepStrictEqual(order, suffixArrayNaive(s), s);
    assert.ok(isValidSuffixArray(s, order), s);
  }

  let seed = 2718;
  for (const alphabet of ["ab", "abc", "abcdefghijklmnopqrstuvwxyz"]) {
    for (let trial = 0; trial < 40; trial++) {
      let s;
      [s, seed] = lcgString(28, alphabet, seed);
      const order = suffixArrayDoubling(s);
      assert.deepStrictEqual(order, suffixArrayNaive(s), s);
      assert.ok(isValidSuffixArray(s, order), s);
    }
  }

  const medium = [];
  const [s1, seed2] = lcgString(500, "ab", seed);
  const [s2, seed3] = lcgString(500, "abcdefghijklmnopqrstuvwxyz", seed2);
  seed = seed3;
  medium.push(s1, s2);
  medium.push(repeat("a", 400), repeat("abcab", 100));
  for (const s of medium) {
    const order = suffixArrayDoubling(s);
    assert.deepStrictEqual(order, suffixArrayNaive(s), "medium naive");
    assert.ok(isValidSuffixArray(s, order), "medium comparator");
  }
  console.log("PASS");
}
