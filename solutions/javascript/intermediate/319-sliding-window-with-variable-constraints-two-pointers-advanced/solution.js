/** Task 319 - Sliding window with variable constraints (two pointers, advanced). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

/** Length of the longest substring with at most k distinct characters (0 if k <= 0). */
function longestAtMostKDistinct(s, k) {
  if (k <= 0) return 0;
  const counts = new Map();
  let left = 0;
  let best = 0;
  for (let right = 0; right < s.length; right++) {
    const ch = s[right];
    counts.set(ch, (counts.get(ch) || 0) + 1);
    while (counts.size > k) {
      const out = s[left++];
      const remaining = counts.get(out) - 1;
      if (remaining === 0) counts.delete(out);
      else counts.set(out, remaining);
    }
    best = Math.max(best, right - left + 1);
  }
  return best;
}

/** Length of the longest substring obtainable by replacing at most k characters. */
function characterReplacement(s, k) {
  const counts = new Map();
  let left = 0;
  let best = 0;
  let maxCount = 0;
  for (let right = 0; right < s.length; right++) {
    const ch = s[right];
    counts.set(ch, (counts.get(ch) || 0) + 1);
    maxCount = Math.max(maxCount, counts.get(ch));
    while (right - left + 1 - maxCount > k) {
      const out = s[left++];
      counts.set(out, counts.get(out) - 1);
      maxCount = Math.max(...counts.values());       // recomputed after every shrink
    }
    best = Math.max(best, right - left + 1);
  }
  return best;
}

/** Try every window; the independent oracle for the distinct-characters variant. */
function bruteAtMostKDistinct(s, k) {
  let best = 0;
  for (let i = 0; i < s.length; i++) {
    const seen = new Set();
    for (let j = i; j < s.length; j++) {
      seen.add(s[j]);
      if (seen.size > k) break;
      best = Math.max(best, j - i + 1);
    }
  }
  return best;
}

/** Try every window and count its maximal frequency; the independent oracle. */
function bruteCharacterReplacement(s, k) {
  let best = 0;
  for (let i = 0; i < s.length; i++) {
    const counts = new Map();
    for (let j = i; j < s.length; j++) {
      const ch = s[j];
      counts.set(ch, (counts.get(ch) || 0) + 1);
      if (j - i + 1 - Math.max(...counts.values()) <= k) best = Math.max(best, j - i + 1);
    }
  }
  return best;
}

function lcgStrings(length, count, alphabet = "abc", seed = 3190) {
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

module.exports = { longestAtMostKDistinct, characterReplacement, bruteAtMostKDistinct, bruteCharacterReplacement, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(longestAtMostKDistinct("eceba", 2), 3);              // "ece"
  assert.strictEqual(longestAtMostKDistinct("aa", 1), 2);
  assert.strictEqual(longestAtMostKDistinct("abcadcacacaca", 3), 11);     // "cadcacacaca"
  assert.strictEqual(longestAtMostKDistinct("abcadcacacaca", 2), 8);      // "cacacaca"
  assert.strictEqual(longestAtMostKDistinct("", 2), 0);
  assert.strictEqual(longestAtMostKDistinct("abc", 0), 0);
  assert.strictEqual(longestAtMostKDistinct("abc", 5), 3);
  assert.strictEqual(longestAtMostKDistinct("abaccc", 1), 3);             // "ccc"

  assert.strictEqual(characterReplacement("ABAB", 2), 4);
  assert.strictEqual(characterReplacement("AABABBA", 1), 4);
  assert.strictEqual(characterReplacement("AAAA", 2), 4);
  assert.strictEqual(characterReplacement("ABCDE", 0), 1);
  assert.strictEqual(characterReplacement("", 0), 0);
  assert.strictEqual(characterReplacement("A", 0), 1);
  assert.strictEqual(characterReplacement("ABAB", 0), 1);

  const samples = ["eceba", "aa", "abcadcacacaca", ...lcgStrings(20, 30, "abc"), ...lcgStrings(14, 15, "ab")];
  for (const s of samples) {
    for (let k = 0; k < 4; k++) {
      assert.strictEqual(longestAtMostKDistinct(s, k), bruteAtMostKDistinct(s, k), `${s} ${k}`);
      assert.strictEqual(characterReplacement(s, k), bruteCharacterReplacement(s, k), `${s} ${k}`);
    }
    assert.strictEqual(longestAtMostKDistinct(s, 4), s.length, s);
    for (let k = 0; k < 3; k++) {                                          // monotone in k
      assert.ok(longestAtMostKDistinct(s, k) <= longestAtMostKDistinct(s, k + 1), s);
      assert.ok(characterReplacement(s, k) <= characterReplacement(s, k + 1), s);
    }
  }

  console.log("PASS");
}
