/** Task 402 - Build the LCP array from the suffix array (Kasai). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Kasai's algorithm walks the suffixes in text order, reusing the previous longest common
 * prefix: if suffix i shares h characters with its suffix-array predecessor, then suffix i+1
 * shares at least h-1 with its own predecessor, so each position is advanced amortized once
 * and the whole scan is O(n). lcp[i] is the LCP of suffixes sa[i-1] and sa[i], lcp[0] = 0.
 *
 * Hand-checked anchors: banana -> [0, 1, 3, 0, 0, 2] and mississippi -> [0, 1, 1, 4, 0, 0,
 * 1, 0, 2, 1, 3]. A pairwise char-by-char oracle and the identity "distinct substrings =
 * n(n+1)/2 - sum(lcp)" cross-check it on LCG strings.
 */
"use strict";

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

function pairLcp(s, i, j) {
  const n = s.length;
  let h = 0;
  while (i + h < n && j + h < n && s[i + h] === s[j + h]) h++;
  return h;
}

function naiveLcpArray(s, sa) {
  if (sa.length === 0) return [];
  const lcp = [0];
  for (let i = 1; i < sa.length; i++) lcp.push(pairLcp(s, sa[i - 1], sa[i]));
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

function distinctSubstringsBrute(s) {
  const seen = new Set();
  for (let i = 0; i < s.length; i++) {
    for (let j = i + 1; j <= s.length; j++) seen.add(s.slice(i, j));
  }
  return seen.size;
}

function lcgString(length, alphabet, seed) {
  const chars = [];
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    chars.push(alphabet[seed % alphabet.length]);
  }
  return [chars.join(""), seed];
}

module.exports = { kasaiLcp, pairLcp, naiveLcpArray, suffixArrayNaive, distinctSubstringsBrute, lcgString };

if (require.main === module) {
  const assert = require("node:assert");
  const banana = kasaiLcp("banana", [5, 3, 1, 0, 4, 2]);
  assert.deepStrictEqual(banana, [0, 1, 3, 0, 0, 2]);
  assert.deepStrictEqual(banana, naiveLcpArray("banana", [5, 3, 1, 0, 4, 2]));

  const mississippi = kasaiLcp("mississippi", [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]);
  assert.deepStrictEqual(mississippi, [0, 1, 1, 4, 0, 0, 1, 0, 2, 1, 3]);
  assert.deepStrictEqual(mississippi,
    naiveLcpArray("mississippi", [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]));

  assert.deepStrictEqual(kasaiLcp("", []), []);
  assert.deepStrictEqual(kasaiLcp("a", [0]), [0]);
  assert.deepStrictEqual(kasaiLcp("aaaa", [3, 2, 1, 0]), [0, 1, 2, 3]);
  assert.deepStrictEqual(kasaiLcp("abab", [2, 0, 3, 1]), [0, 2, 0, 1]);

  const small = ["", "a", "aaa", "abab", "banana", "mississippi", "abracadabra", "aaaaabaaaab"];
  for (const s of small) {
    const sa = suffixArrayNaive(s);
    const lcp = kasaiLcp(s, sa);
    assert.deepStrictEqual(lcp, naiveLcpArray(s, sa), s);
    assert.strictEqual(lcp.length, s.length, s);
    assert.ok(lcp.length === 0 || lcp[0] === 0, s);
  }

  assert.strictEqual(distinctSubstringsBrute("banana"), 15);
  assert.strictEqual((6 * 7) / 2 - banana.reduce((a, b) => a + b, 0), 15);

  let seed = 2718;
  for (const alphabet of ["ab", "abc"]) {
    for (let trial = 0; trial < 40; trial++) {
      let s;
      [s, seed] = lcgString(24, alphabet, seed);
      const sa = suffixArrayNaive(s);
      const lcp = kasaiLcp(s, sa);
      assert.deepStrictEqual(lcp, naiveLcpArray(s, sa), s);
      assert.strictEqual((s.length * (s.length + 1)) / 2 - lcp.reduce((a, b) => a + b, 0),
        distinctSubstringsBrute(s), s);
    }
  }

  const [medium] = lcgString(400, "ab", seed);
  const mediumSa = suffixArrayNaive(medium);
  const mediumLcp = kasaiLcp(medium, mediumSa);
  assert.deepStrictEqual(mediumLcp, naiveLcpArray(medium, mediumSa), "medium");
  assert.strictEqual(mediumLcp[0], 0, "medium first");
  console.log("PASS");
}
