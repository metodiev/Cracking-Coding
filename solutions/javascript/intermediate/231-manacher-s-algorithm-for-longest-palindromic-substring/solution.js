/** Task 231 - Manacher's algorithm for longest palindromic substring. (Tier: Intermediate, CTCI Ch 1) */
"use strict";

/** Manacher: leftmost longest palindromic substring of s; "" for empty input. */
function longestPalindrome(s) {
  if (s.length === 0) return "";
  let t = "^";
  for (const ch of s) t += "#" + ch;
  t += "#$";
  const n = t.length;
  const p = new Array(n).fill(0);
  let centre = 0;
  let right = 0;
  let bestLen = 0;
  let bestStart = 0;
  for (let i = 1; i < n - 1; i++) {
    const mirror = 2 * centre - i;
    if (i < right) p[i] = Math.min(right - i, p[mirror]);
    while (t[i + 1 + p[i]] === t[i - 1 - p[i]]) p[i]++;
    if (i + p[i] > right) {
      centre = i;
      right = i + p[i];
    }
    if (p[i] > bestLen) {                     // strict: ties keep the leftmost centre
      bestLen = p[i];
      bestStart = (i - p[i]) >> 1;
    }
  }
  return s.slice(bestStart, bestStart + bestLen);
}

/** O(n^2) reference helper: length of the longest palindromic substring of s. */
function expandAroundCentreLength(s) {
  const expand = (lo, hi) => {
    while (lo >= 0 && hi < s.length && s[lo] === s[hi]) { lo--; hi++; }
    return hi - lo - 1;
  };
  let best = 0;
  for (let i = 0; i < s.length; i++) {
    best = Math.max(best, expand(i, i), expand(i, i + 1));
  }
  return best;
}

module.exports = { longestPalindrome, expandAroundCentreLength };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestPalindrome("babad"), "bab");      // leftmost of "bab"/"aba"
  assert.strictEqual(longestPalindrome("cbbd"), "bb");
  assert.strictEqual(longestPalindrome("forgeeksskeegfor"), "geeksskeeg");
  assert.strictEqual(longestPalindrome("abacaba"), "abacaba");
  assert.strictEqual(longestPalindrome("bananas"), "anana");
  assert.strictEqual(longestPalindrome("abcde"), "a");
  assert.strictEqual(longestPalindrome("ccc"), "ccc");
  assert.strictEqual(longestPalindrome(""), "");
  assert.strictEqual(longestPalindrome("a"), "a");

  // deterministic LCG so the cross-check is reproducible
  let seed = 231;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const reverse = (str) => [...str].reverse().join("");
  for (let t = 0; t < 300; t++) {
    const len = Math.floor(rand() * 41);
    let s = "";
    for (let i = 0; i < len; i++) s += rand() < 0.5 ? "a" : "b";
    const got = longestPalindrome(s);
    assert.strictEqual(got, reverse(got), "palindromic result");
    assert.strictEqual(got.length, expandAroundCentreLength(s), "length matches expand-around-centre");
    if (s.length > 0) {
      let first = -1;
      for (let i = 0; i + got.length <= s.length && first < 0; i++) {
        const sub = s.slice(i, i + got.length);
        if (sub === reverse(sub)) first = i;
      }
      assert.strictEqual(s.indexOf(got), first, "leftmost among maximal palindromes");
    }
  }
  console.log("PASS");
}
