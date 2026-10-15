/** Task 186 - Rabin-Karp rolling hash string search. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

const DEFAULT_BASE = 256;
const DEFAULT_MOD = 1000000007;

function powMod(base, exponent, mod) {
  let result = 1;
  for (let i = 0; i < exponent; i++) result = (result * base) % mod;
  return result;
}

/**
 * Polynomial window hash h = c0 * base^(m-1) + ... + c(m-1) (mod mod). Sliding one step
 * drops the leaving character's weighted term and appends the new one in O(1). Equal
 * hashes can also mean a collision, so every hash hit is confirmed with a direct
 * substring comparison: a false positive never becomes a reported match.
 */
function rabinKarp(text, pattern, base = DEFAULT_BASE, mod = DEFAULT_MOD) {
  const matches = [];
  if (pattern === "") return matches;
  const m = pattern.length;
  const n = text.length;
  if (m > n) return matches;

  const highest = powMod(base, m - 1, mod); // weight of a window's first character
  let patternHash = 0;
  let windowHash = 0;
  for (let i = 0; i < m; i++) {
    patternHash = (patternHash * base + pattern.charCodeAt(i)) % mod;
    windowHash = (windowHash * base + text.charCodeAt(i)) % mod;
  }

  if (windowHash === patternHash && text.slice(0, m) === pattern) matches.push(0);
  for (let start = 1; start + m <= n; start++) {
    windowHash = (windowHash - text.charCodeAt(start - 1) * highest) % mod;
    if (windowHash < 0) windowHash += mod;
    windowHash = (windowHash * base + text.charCodeAt(start + m - 1)) % mod;
    if (windowHash === patternHash && text.slice(start, start + m) === pattern) matches.push(start);
  }
  return matches;
}

/** Reference scan used by the self-check. */
function rabinKarpNaive(text, pattern) {
  const matches = [];
  if (pattern === "") return matches;
  for (let i = 0; i + pattern.length <= text.length; i++) {
    if (text.startsWith(pattern, i)) matches.push(i);
  }
  return matches;
}

module.exports = { rabinKarp, rabinKarpNaive };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(rabinKarp("abracadabra", "abra"), [0, 7]);
  assert.deepStrictEqual(rabinKarp("aaaaa", "aa"), [0, 1, 2, 3]);
  assert.deepStrictEqual(rabinKarp("abcabcabc", "abc"), [0, 3, 6]);
  assert.deepStrictEqual(rabinKarp("abcd", "bcd"), [1]);
  assert.deepStrictEqual(rabinKarp("aaaa", "b"), []);
  assert.deepStrictEqual(rabinKarp("ab", "abc"), []);
  assert.deepStrictEqual(rabinKarp("", "a"), []);
  assert.deepStrictEqual(rabinKarp("", ""), []);
  assert.deepStrictEqual(rabinKarp("anything", ""), []);
  assert.deepStrictEqual(rabinKarp("a", "a"), [0]);

  // repetition-heavy text
  const heavy = "a".repeat(1000) + "b";
  assert.deepStrictEqual(rabinKarp(heavy, "a".repeat(20)), Array.from({ length: 981 }, (_, i) => i));
  assert.deepStrictEqual(rabinKarp(heavy, "ab"), [999]);
  assert.deepStrictEqual(rabinKarp("ab".repeat(300), "abab"), Array.from({ length: 299 }, (_, i) => i * 2));

  // agreement with the naive scan on pseudo-random strings
  let state = 17;
  const nextInt = (bound) => {
    state = (state * 48271) % 2147483647;
    return state % bound;
  };
  const randomText = (length, alphabet) => {
    let s = "";
    for (let i = 0; i < length; i++) s += String.fromCharCode(97 + nextInt(alphabet));
    return s;
  };
  for (let t = 0; t < 200; t++) {
    const text = randomText(nextInt(41), 2);
    const pattern = randomText(1 + nextInt(5), 2);
    assert.deepStrictEqual(rabinKarp(text, pattern), rabinKarpNaive(text, pattern), `${text} / ${pattern}`);
  }

  // a tiny modulus forces collisions; the substring verification must filter them out
  for (let t = 0; t < 200; t++) {
    const text = randomText(nextInt(41), 3);
    const pattern = randomText(1 + nextInt(4), 3);
    assert.deepStrictEqual(rabinKarp(text, pattern, 256, 101), rabinKarpNaive(text, pattern), `${text} / ${pattern}`);
  }
  const collidingText = randomText(60, 6);
  const seen = new Map();
  let collided = false;
  for (let start = 0; start + 3 <= collidingText.length; start++) {
    const window = collidingText.slice(start, start + 3);
    let h = 0;
    for (let i = 0; i < 3; i++) h = (h * 256 + window.charCodeAt(i)) % 101;
    if (seen.has(h) && seen.get(h) !== window) collided = true;
    if (!seen.has(h)) seen.set(h, window);
  }
  assert.ok(collided, "expected the tiny modulus to produce a real hash collision");
  console.log("PASS");
}
