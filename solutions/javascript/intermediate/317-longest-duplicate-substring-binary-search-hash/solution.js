/** Task 317 - Longest duplicate substring via binary search + rolling hash. (Tier: Intermediate, CTCI Ch 1) */
"use strict";

const MOD1 = 1_000_000_007;
const MOD2 = 998_244_353;
const BASE = 131;

/** (a * b) % mod without losing precision: JS numbers are exact only below 2^53. */
function mulmod(a, b, mod) {
  const bHi = Math.floor(b / 65536);
  const bLo = b - bHi * 65536;
  return ((a * bHi % mod) * 65536 + a * bLo) % mod;
}

class DoubleRollingHash {
  constructor(s) {
    const n = s.length;
    this.h1 = new Array(n + 1).fill(0);
    this.h2 = new Array(n + 1).fill(0);
    this.p1 = new Array(n + 1).fill(1);
    this.p2 = new Array(n + 1).fill(1);
    for (let i = 0; i < n; i++) {
      const code = s.charCodeAt(i);
      this.h1[i + 1] = (this.h1[i] * BASE + code) % MOD1;
      this.h2[i + 1] = (this.h2[i] * BASE + code) % MOD2;
      this.p1[i + 1] = (this.p1[i] * BASE) % MOD1;
      this.p2[i + 1] = (this.p2[i] * BASE) % MOD2;
    }
  }

  /** Hash of s[lo..hi) as the pair (mod MOD1, mod MOD2). */
  hash(lo, hi) {
    const length = hi - lo;
    const a = (this.h1[hi] - mulmod(this.h1[lo], this.p1[length], MOD1) + MOD1) % MOD1;
    const b = (this.h2[hi] - mulmod(this.h2[lo], this.p2[length], MOD2) + MOD2) % MOD2;
    return [a, b];
  }

  equal(lo1, hi1, lo2, hi2) {
    const left = this.hash(lo1, hi1);
    const right = this.hash(lo2, hi2);
    return left[0] === right[0] && left[1] === right[1];
  }
}

/**
 * Simplest duplicate substring of maximal length, found at the smallest starting index; ""
 * is the documented sentinel meaning "no duplicate substring exists".
 */
function longestDuplicateSubstring(s) {
  const n = s.length;
  if (n <= 1) return "";
  const rolling = new DoubleRollingHash(s);
  const hasDuplicate = (length) => {
    const seen = new Set();
    for (let i = 0; i + length <= n; i++) {
      const key = rolling.hash(i, i + length);
      const packed = `${key[0]},${key[1]}`;      // string key: the product would exceed 2^53
      if (seen.has(packed)) return true;
      seen.add(packed);
    }
    return false;
  };
  let lo = 1;
  let hi = n - 1;
  let best = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (hasDuplicate(mid)) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (best === 0) return "";
  for (let i = 0; i + best <= n; i++) {
    for (let j = i + 1; j + best <= n; j++) {
      if (rolling.equal(i, i + best, j, j + best)) return s.slice(i, i + best);
    }
  }
  throw new Error("binary search and extraction disagree");
}

/** All-pairs LCP scan; the independent oracle for the best length. */
function longestDuplicateNaiveLength(s) {
  let best = 0;
  for (let i = 0; i < s.length; i++) {
    for (let j = i + 1; j < s.length; j++) {
      let h = 0;
      while (j + h < s.length && s[i + h] === s[j + h]) h++;
      if (h > best) best = h;
    }
  }
  return best;
}

/** Occurrences counted with overlaps (non-overlapping counting would undercount repeats). */
function overlappingCount(s, word) {
  let count = 0;
  for (let i = 0; i + word.length <= s.length; i++) {
    if (s.slice(i, i + word.length) === word) count++;
  }
  return count;
}

function lcgStrings(length, count, alphabet = "ab", seed = 3170) {
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

module.exports = { DoubleRollingHash, longestDuplicateSubstring, longestDuplicateNaiveLength, overlappingCount, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(longestDuplicateSubstring("banana"), "ana");           // "ana" at 1 and 3
  assert.strictEqual(longestDuplicateSubstring("aaaa"), "aaa");             // "aaa" at 0 and 1
  assert.strictEqual(longestDuplicateSubstring("abcd"), "");                // nothing repeats
  assert.strictEqual(longestDuplicateSubstring(""), "");
  assert.strictEqual(longestDuplicateSubstring("a"), "");
  assert.strictEqual(longestDuplicateSubstring("aa"), "a");
  assert.strictEqual(longestDuplicateSubstring("abab"), "ab");
  assert.strictEqual(longestDuplicateSubstring("abcabcabc"), "abcabc");     // at 0 and 3
  assert.strictEqual(longestDuplicateSubstring("mississippi"), "issi");     // at 1 and 4

  const samples = ["banana", "aaaa", "abcd", "abab", "abcabcabc", "mississippi", "aabcaabxaaaz",
    ...lcgStrings(24, 30, "ab"), ...lcgStrings(18, 15, "abc")];
  for (const s of samples) {
    const result = longestDuplicateSubstring(s);
    const naive = longestDuplicateNaiveLength(s);
    assert.strictEqual(result.length, naive, s);
    if (naive === 0) {
      assert.strictEqual(result, "", s);
    } else {
      assert.ok(overlappingCount(s, result) > 1, s);                        // a real duplicate ...
      assert.strictEqual(result, longestDuplicateSubstring(s), s);          // ... and deterministic
    }
  }

  console.log("PASS");
}
