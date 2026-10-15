/** Task 312 - Substring occurrences with a suffix array + LCP (Kasai). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function suffixArray(s) {
  const indices = Array.from({ length: s.length }, (_, i) => i);
  indices.sort((a, b) => {
    const sa = s.slice(a);
    const sb = s.slice(b);
    return sa < sb ? -1 : sa > sb ? 1 : 0;
  });
  return indices;
}

/** Manual lexicographic comparator; the independent oracle. */
function compareSuffixes(s, i, j) {
  const n = s.length;
  while (i < n && j < n) {
    if (s[i] !== s[j]) return s[i] < s[j] ? -1 : 1;
    i++;
    j++;
  }
  if (i === j) return 0;
  return i === n ? -1 : 1;                 // the shorter suffix is smaller
}

function isValidSuffixArray(s, order) {
  const sorted = [...order].sort((a, b) => a - b);
  for (let i = 0; i < sorted.length; i++) if (sorted[i] !== i) return false;
  for (let k = 0; k + 1 < order.length; k++) if (compareSuffixes(s, order[k], order[k + 1]) >= 0) return false;
  return true;
}

/** lcp[i] = LCP(sa[i-1], sa[i]) with lcp[0] = 0; O(n) via the Kasai trick. */
function kasaiLcp(s, sa) {
  const n = s.length;
  const rank = new Array(n).fill(0);
  for (let i = 0; i < n; i++) rank[sa[i]] = i;
  const lcp = new Array(n).fill(0);
  let h = 0;
  for (let i = 0; i < n; i++) {
    if (rank[i] > 0) {
      const j = sa[rank[i] - 1];
      while (i + h < n && j + h < n && s[i + h] === s[j + h]) h++;
      lcp[rank[i]] = h;
      if (h > 0) h--;
    } else {
      h = 0;
    }
  }
  return lcp;
}

/** Direct character-by-character LCP of two suffixes; the independent oracle. */
function lcpOfSuffixes(s, i, j) {
  let h = 0;
  while (i + h < s.length && j + h < s.length && s[i + h] === s[j + h]) h++;
  return h;
}

/**
 * All start indices of the pattern in s, sorted ascending. The empty pattern matches at every
 * valid position (n + 1 of them).
 */
function occurrences(s, pattern, sa = null, lcp = null) {
  const n = s.length;
  const m = pattern.length;
  if (m === 0) return Array.from({ length: n + 1 }, (_, i) => i);
  if (sa === null) sa = suffixArray(s);
  if (lcp === null) lcp = kasaiLcp(s, sa);
  // first index k with suffix sa[k] >= pattern; matching suffixes are exactly the run at k
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (s.slice(sa[mid]) < pattern) lo = mid + 1;
    else hi = mid;
  }
  const k = lo;
  if (k === n || s.slice(sa[k], sa[k] + m) !== pattern) return [];
  const result = [sa[k]];
  let t = k + 1;
  while (t < n && lcp[t] >= m) {
    result.push(sa[t]);
    t++;
  }
  result.sort((a, b) => a - b);
  return result;
}

/** Scan every position; the independent oracle (overlapping matches included). */
function naiveOccurrences(s, pattern) {
  if (pattern === "") return Array.from({ length: s.length + 1 }, (_, i) => i);
  const result = [];
  for (let i = 0; i + pattern.length <= s.length; i++) {
    if (s.slice(i, i + pattern.length) === pattern) result.push(i);
  }
  return result;
}

function lcgStrings(length, count, alphabet = "ab", seed = 3120) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += alphabet[seed % alphabet.length];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { suffixArray, compareSuffixes, isValidSuffixArray, kasaiLcp, lcpOfSuffixes, occurrences, naiveOccurrences, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");

  const s = "banana";
  const sa = suffixArray(s);
  assert.deepStrictEqual(sa, [5, 3, 1, 0, 4, 2]);
  assert.ok(isValidSuffixArray(s, sa));
  const lcp = kasaiLcp(s, sa);
  assert.deepStrictEqual(lcp, [0, 1, 3, 0, 0, 2]);             // hand-checked: a|ana, ana|anana, na|nana
  for (let i = 1; i < sa.length; i++) assert.strictEqual(lcp[i], lcpOfSuffixes(s, sa[i - 1], sa[i]), String(i));

  assert.deepStrictEqual(occurrences(s, "ana", sa, lcp), [1, 3]);
  assert.deepStrictEqual(occurrences(s, "ba", sa, lcp), [0]);
  assert.deepStrictEqual(occurrences(s, "an", sa, lcp), [1, 3]);
  assert.deepStrictEqual(occurrences(s, "a", sa, lcp), [1, 3, 5]);
  assert.deepStrictEqual(occurrences(s, "n", sa, lcp), [2, 4]);
  assert.deepStrictEqual(occurrences(s, "banana", sa, lcp), [0]);
  assert.deepStrictEqual(occurrences(s, "ab", sa, lcp), []);
  assert.deepStrictEqual(occurrences(s, "xy", sa, lcp), []);
  assert.deepStrictEqual(occurrences(s, "bananas", sa, lcp), []);
  assert.deepStrictEqual(occurrences(s, "", sa, lcp), [0, 1, 2, 3, 4, 5, 6]);
  assert.deepStrictEqual(occurrences("", "a"), []);

  const texts = ["a", "aa", "aaaa", "abab", "mississippi", "banana", ""];
  texts.push(...lcgStrings(20, 30, "ab"));
  texts.push(...lcgStrings(16, 10, "abc"));

  for (const text of texts) {
    const textSa = suffixArray(text);
    const textLcp = kasaiLcp(text, textSa);
    assert.ok(isValidSuffixArray(text, textSa), text);
    for (let i = 1; i < textSa.length; i++) {
      assert.strictEqual(textLcp[i], lcpOfSuffixes(text, textSa[i - 1], textSa[i]), `${text} ${i}`);
    }
    const patterns = [];
    for (let i = 0; i < text.length; i++) {
      for (let extra = 1; extra <= 3; extra++) {
        if (i + extra <= text.length) patterns.push(text.slice(i, i + extra));
      }
    }
    patterns.push("zz", "z", "ababab", "", "bbbb", "ccc");
    for (const pattern of patterns) {
      assert.deepStrictEqual(occurrences(text, pattern, textSa, textLcp), naiveOccurrences(text, pattern),
        `${text} / ${pattern}`);
    }
  }

  console.log("PASS");
}
