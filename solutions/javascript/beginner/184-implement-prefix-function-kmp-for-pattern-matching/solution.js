/** Task 184 - Implement prefix function (KMP) for pattern matching. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

/**
 * pi[i] is the length of the longest proper prefix of s.slice(0, i + 1) that is also a
 * suffix of it. The fallback loop runs in O(n) overall because j only ever moves back
 * via earlier prefix-function values.
 */
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

/** O(n^2) reference: longest proper border of every prefix, brute-forced. */
function prefixFunctionNaive(s) {
  const pi = [];
  for (let end = 0; end < s.length; end++) {
    let best = 0;
    for (let length = 1; length <= end; length++) {
      if (s.slice(0, length) === s.slice(end - length + 1, end + 1)) best = length;
    }
    pi.push(best);
  }
  return pi;
}

module.exports = { prefixFunction, prefixFunctionNaive };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(prefixFunction(""), []);
  assert.deepStrictEqual(prefixFunction("a"), [0]);
  assert.deepStrictEqual(prefixFunction("ab"), [0, 0]);
  assert.deepStrictEqual(prefixFunction("ababaca"), [0, 0, 1, 2, 3, 0, 1]);
  assert.deepStrictEqual(prefixFunction("aabaaab"), [0, 1, 0, 1, 2, 2, 3]);
  assert.deepStrictEqual(prefixFunction("aaaa"), [0, 1, 2, 3]);
  assert.deepStrictEqual(prefixFunction("abcabcd"), [0, 0, 0, 1, 2, 3, 0]);
  assert.deepStrictEqual(prefixFunction("abababab"), [0, 0, 1, 2, 3, 4, 5, 6]);
  assert.deepStrictEqual(prefixFunction("aabaaabaaab"), [0, 1, 0, 1, 2, 2, 3, 4, 5, 6, 7]);

  // agreement with the plain definition on pseudo-random strings
  let state = 5;
  const nextInt = (bound) => {
    state = (state * 48271) % 2147483647;
    return state % bound;
  };
  for (let t = 0; t < 200; t++) {
    let s = "";
    const length = nextInt(41);
    for (let i = 0; i < length; i++) s += nextInt(2) === 0 ? "a" : "b";
    assert.deepStrictEqual(prefixFunction(s), prefixFunctionNaive(s), `cross-check ${s}`);
  }
  console.log("PASS");
}
