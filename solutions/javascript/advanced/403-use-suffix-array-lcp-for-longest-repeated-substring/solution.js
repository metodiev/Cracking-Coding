/** Task 403 - Use the suffix array + LCP array for the longest repeated substring. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A substring occurring at least twice is a common prefix of two different suffixes, and those
 * two suffixes are adjacent somewhere in the suffix array, so the answer length is max(lcp)
 * and one witness is the prefix of that length of sa[i] at the first maximizing i.
 *
 * Sentinel convention: when nothing repeats (length < 2, or all substrings unique like "abc")
 * the function returns "" - documented so "no repeat" is distinguishable from a one-character
 * repeat. Anchors: banana -> "ana", mississippi -> "issi", aaaa -> "aaa", abab -> "ab".
 * Cross-checked against a brute-force substring counter on LCG strings.
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

function longestRepeatedSubstring(s) {
  if (s.length < 2) return "";
  const sa = suffixArrayDoubling(s);
  const lcp = kasaiLcp(s, sa);
  let bestLength = 0;
  let bestStart = -1;
  for (let i = 1; i < s.length; i++) {
    if (lcp[i] > bestLength) {
      bestLength = lcp[i];
      bestStart = sa[i];
    }
  }
  return bestLength === 0 ? "" : s.slice(bestStart, bestStart + bestLength);
}

function longestRepeatedLengthBrute(s) {
  const counts = new Map();
  for (let i = 0; i < s.length; i++) {
    for (let j = i + 1; j <= s.length; j++) {
      const t = s.slice(i, j);
      counts.set(t, (counts.get(t) || 0) + 1);
    }
  }
  let best = 0;
  for (const [t, c] of counts) if (c >= 2 && t.length > best) best = t.length;
  return best;
}

function countOccurrences(s, pattern) {
  if (!pattern) return 0;
  let count = 0;
  for (let i = 0; i + pattern.length <= s.length; i++) {
    if (s.startsWith(pattern, i)) count++;
  }
  return count;
}

function lcgString(length, alphabet, seed) {
  const chars = [];
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    chars.push(alphabet[seed % alphabet.length]);
  }
  return [chars.join(""), seed];
}

module.exports = { suffixArrayDoubling, kasaiLcp, longestRepeatedSubstring, longestRepeatedLengthBrute, countOccurrences, lcgString };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestRepeatedSubstring("banana"), "ana");
  assert.strictEqual(countOccurrences("banana", "ana"), 2);
  assert.strictEqual(longestRepeatedSubstring("mississippi"), "issi");
  assert.strictEqual(countOccurrences("mississippi", "issi"), 2);
  assert.strictEqual(longestRepeatedSubstring("aaaa"), "aaa");
  assert.strictEqual(longestRepeatedSubstring("abab"), "ab");
  assert.strictEqual(longestRepeatedSubstring("abcabc"), "abc");
  assert.strictEqual(longestRepeatedSubstring("aabcaa"), "aa");
  assert.strictEqual(longestRepeatedSubstring("abracadabra"), "abra");
  assert.strictEqual(countOccurrences("abracadabra", "abra"), 2);

  assert.strictEqual(longestRepeatedSubstring(""), "");
  assert.strictEqual(longestRepeatedSubstring("a"), "");
  assert.strictEqual(longestRepeatedSubstring("abc"), "");
  assert.strictEqual(longestRepeatedSubstring("abcd"), "");
  assert.strictEqual(longestRepeatedLengthBrute("abc"), 0);

  let seed = 2718;
  for (const alphabet of ["ab", "abc"]) {
    for (let trial = 0; trial < 40; trial++) {
      let s;
      [s, seed] = lcgString(30, alphabet, seed);
      const result = longestRepeatedSubstring(s);
      const best = longestRepeatedLengthBrute(s);
      assert.strictEqual(result.length, best, s);
      if (best === 0) assert.strictEqual(result, "", s);
      else assert.ok(countOccurrences(s, result) >= 2, s);
    }
  }

  let medium;
  [medium, seed] = lcgString(400, "ab", seed);
  const mediumResult = longestRepeatedSubstring(medium);
  assert.ok(mediumResult && countOccurrences(medium, mediumResult) >= 2, "medium witness");
  assert.strictEqual(mediumResult.length, longestRepeatedLengthBrute(medium), "medium length");
  console.log("PASS");
}
