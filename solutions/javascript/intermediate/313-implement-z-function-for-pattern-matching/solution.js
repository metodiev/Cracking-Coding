/** Task 313 - The Z-function for pattern matching. (Tier: Intermediate, CTCI Ch 1) */
"use strict";

const SENTINEL = "#";

/** z[i] = longest common prefix of s and s.slice(i); z[0] = 0 by convention. */
function zFunction(s) {
  const n = s.length;
  const z = new Array(n).fill(0);
  let left = 0;
  let right = 0;
  for (let i = 1; i < n; i++) {
    if (i < right) z[i] = Math.min(right - i, z[i - left]);
    while (i + z[i] < n && s[z[i]] === s[i + z[i]]) z[i]++;
    if (i + z[i] > right) {
      left = i;
      right = i + z[i];
    }
  }
  return z;
}

/** Direct O(n^2) definition; the independent oracle. */
function zFunctionNaive(s) {
  const n = s.length;
  const z = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    let h = 0;
    while (i + h < n && s[h] === s[i + h]) h++;
    z[i] = h;
  }
  return z;
}

/** All start indices of pattern in text via the Z-function on pattern + "#" + text. */
function zSearch(text, pattern) {
  const m = pattern.length;
  if (m === 0) return Array.from({ length: text.length + 1 }, (_, i) => i);
  const combined = pattern + SENTINEL + text;
  const z = zFunction(combined);
  const result = [];
  for (let i = m + 1; i < combined.length; i++) {
    if (z[i] >= m) result.push(i - m - 1);
  }
  return result;
}

/** Independent oracle: try every position. */
function naiveSearch(text, pattern) {
  if (pattern === "") return Array.from({ length: text.length + 1 }, (_, i) => i);
  const result = [];
  for (let i = 0; i + pattern.length <= text.length; i++) {
    if (text.slice(i, i + pattern.length) === pattern) result.push(i);
  }
  return result;
}

function lcgStrings(length, count, alphabet = "ab", seed = 3130) {
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

module.exports = { SENTINEL, zFunction, zFunctionNaive, zSearch, naiveSearch, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");

  // Worked example, values checked by hand
  assert.deepStrictEqual(zFunction("aabcaabxaaaz"), [0, 1, 0, 0, 3, 1, 0, 0, 2, 2, 1, 0]);
  assert.deepStrictEqual(zFunction("aaaa"), [0, 3, 2, 1]);
  assert.deepStrictEqual(zFunction("abab"), [0, 0, 2, 0]);
  assert.deepStrictEqual(zFunction("a"), [0]);
  assert.deepStrictEqual(zFunction(""), []);

  const text = "aabcaabxaaaz";
  assert.deepStrictEqual(zSearch(text, "aab"), [0, 4]);
  assert.deepStrictEqual(zSearch(text, "aa"), [0, 4, 8, 9]);
  assert.deepStrictEqual(zSearch(text, "az"), [10]);
  assert.deepStrictEqual(zSearch(text, "x"), [7]);
  assert.deepStrictEqual(zSearch(text, "zz"), []);
  assert.deepStrictEqual(zSearch(text, "bcaab"), [2]);
  assert.deepStrictEqual(zSearch(text, text), [0]);
  assert.deepStrictEqual(zSearch("abc", ""), [0, 1, 2, 3]);
  assert.deepStrictEqual(zSearch("", "a"), []);

  const texts = ["a", "aa", "aaaa", "abab", "aabcaabxaaaz", "mississippi"];
  texts.push(...lcgStrings(24, 30, "ab"));
  texts.push(...lcgStrings(18, 12, "abc"));

  for (const s of texts) {
    const z = zFunction(s);
    assert.deepStrictEqual(z, zFunctionNaive(s), s);
    for (let i = 0; i < s.length; i++) assert.ok(z[i] >= 0 && z[i] <= s.length - i, `${s} ${i}`);
    const patterns = [];
    for (let i = 0; i < s.length; i++) {
      for (let extra = 1; extra <= 3; extra++) {
        if (i + extra <= s.length) patterns.push(s.slice(i, i + extra));
      }
    }
    patterns.push("zz", "ababab", "");
    for (const pattern of patterns) {
      assert.deepStrictEqual(zSearch(s, pattern), naiveSearch(s, pattern), `${s} / ${pattern}`);
    }
  }

  console.log("PASS");
}
