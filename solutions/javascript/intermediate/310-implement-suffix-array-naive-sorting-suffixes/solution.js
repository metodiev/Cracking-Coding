/** Task 310 - Suffix array by naively sorting suffixes. (Tier: Intermediate, CTCI Ch 1) */
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
  const sorted = [...order].sort((a, b) => a - b);
  for (let i = 0; i < sorted.length; i++) if (sorted[i] !== i) return false;
  for (let k = 0; k + 1 < order.length; k++) if (compareSuffixes(s, order[k], order[k + 1]) >= 0) return false;
  return true;
}

function lcgStrings(length, count, seed = 2718) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += "ab"[seed % 2];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { suffixArray, compareSuffixes, isValidSuffixArray, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(suffixArray("banana"), [5, 3, 1, 0, 4, 2]);
  const mississippi = suffixArray("mississippi");
  assert.deepStrictEqual(mississippi, [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]);
  assert.ok(isValidSuffixArray("mississippi", mississippi));
  assert.deepStrictEqual(suffixArray(""), []);
  assert.deepStrictEqual(suffixArray("a"), [0]);
  assert.deepStrictEqual(suffixArray("aaaa"), [3, 2, 1, 0]);
  assert.deepStrictEqual(suffixArray("abab"), [2, 0, 3, 1]);
  for (const s of lcgStrings(12, 60)) {
    assert.ok(isValidSuffixArray(s, suffixArray(s)), s);
  }
  console.log("PASS");
}
