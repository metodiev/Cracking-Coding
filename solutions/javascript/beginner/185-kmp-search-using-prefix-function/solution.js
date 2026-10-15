/** Task 185 - KMP search using prefix function. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

/** Prefix function (LPS array) of `s`, as in task 184. */
function prefixFunction(s) {
  const pi = new Array(s.length).fill(0);
  for (let i = 1; i < s.length; i++) {
    let j = pi[i - 1];
    while (j > 0 && s[i] !== s[j]) j = pi[j - 1];
    if (s[i] === s[j]) j++;
    pi[i] = j;
  }
  return pi;
}

/**
 * All start indices of `pattern` in `text`, in increasing order (overlapping matches
 * included), in O(n + m). An empty pattern returns [] by convention: "no match" avoids
 * an unbounded list of positions.
 */
function kmpSearch(text, pattern) {
  const matches = [];
  if (pattern === "") return matches;
  const pi = prefixFunction(pattern);
  let j = 0;
  for (let i = 0; i < text.length; i++) {
    while (j > 0 && text[i] !== pattern[j]) j = pi[j - 1];
    if (text[i] === pattern[j]) j++;
    if (j === pattern.length) {
      matches.push(i - pattern.length + 1);
      j = pi[j - 1]; // keep going, matches may overlap
    }
  }
  return matches;
}

/** Reference scan used by the self-check. */
function kmpSearchNaive(text, pattern) {
  const matches = [];
  if (pattern === "") return matches;
  for (let i = 0; i + pattern.length <= text.length; i++) {
    if (text.startsWith(pattern, i)) matches.push(i);
  }
  return matches;
}

module.exports = { prefixFunction, kmpSearch, kmpSearchNaive };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(kmpSearch("aaaaa", "aaaa"), [0, 1]);
  assert.deepStrictEqual(kmpSearch("abababa", "aba"), [0, 2, 4]);
  assert.deepStrictEqual(kmpSearch("abcabcabc", "abc"), [0, 3, 6]);
  assert.deepStrictEqual(kmpSearch("abracadabra", "cad"), [4]);
  assert.deepStrictEqual(kmpSearch("mississippi", "issi"), [1, 4]);
  assert.deepStrictEqual(kmpSearch("aaaa", "b"), []);
  assert.deepStrictEqual(kmpSearch("ab", "abc"), []);
  assert.deepStrictEqual(kmpSearch("", "a"), []);
  assert.deepStrictEqual(kmpSearch("", ""), []);
  assert.deepStrictEqual(kmpSearch("anything", ""), []);
  assert.deepStrictEqual(kmpSearch("a", "a"), [0]);
  assert.deepStrictEqual(kmpSearch("aaa", "aa"), [0, 1]);
  assert.deepStrictEqual(kmpSearch("aaa", "aaa"), [0]);

  // agreement with the naive scan on pseudo-random strings
  let state = 13;
  const nextInt = (bound) => {
    state = (state * 48271) % 2147483647;
    return state % bound;
  };
  const letter = () => (nextInt(2) === 0 ? "a" : "b");
  for (let t = 0; t < 300; t++) {
    let text = "";
    const textLength = nextInt(31);
    for (let i = 0; i < textLength; i++) text += letter();
    let pattern = "";
    const patternLength = 1 + nextInt(5);
    for (let i = 0; i < patternLength; i++) pattern += letter();
    assert.deepStrictEqual(kmpSearch(text, pattern), kmpSearchNaive(text, pattern), `${text} / ${pattern}`);
  }
  console.log("PASS");
}
