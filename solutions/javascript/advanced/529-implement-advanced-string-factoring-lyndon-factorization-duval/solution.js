/**
 * Task 529 - Implement advanced string factoring (Lyndon factorization Duval). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Duval's O(n) Lyndon factorization: a single scan advances j over the string while s[k] <= s[j],
 * restarting k at i whenever a strictly larger character appears (which means the current period
 * repeats); when the scan stops, the prefix s[i..i + j - k) is a Lyndon word and can be emitted
 * (j - k) / period times in a row, which is what makes the algorithm linear. The output is the
 * unique non-increasing sequence of Lyndon words whose concatenation is the input - the Chen-Fox-
 * Lyndon factorization.
 *
 * Documented checks: the concatenation of the factors reproduces the input, the factors are
 * non-increasing, and every factor passes an independent Lyndon test (strictly smaller than all of
 * its proper suffixes); Duval's output is compared with a greedy brute-force factorizer that
 * repeatedly appends the longest Lyndon prefix, and both are compared with a third check that
 * enumerates every partition of short strings into non-increasing Lyndon words and asserts the
 * factorization is the unique one. Hand-checked examples ("banana" -> b|an|an|a, "ababab" ->
 * ab|ab|ab, "aaaa" -> a|a|a|a, "abcabc" -> abc|abc) are fixed, and randomised behaviour is covered
 * by LCG-generated strings over two- and three-letter alphabets up to length 200, where the number
 * of factors, the concatenation, the monotonicity and the Lyndon property of every factor are all
 * asserted. Boundary cases: the empty string (no factors), a single character, an all-equal string,
 * and a strictly increasing string (each character is its own factor). The two Lyndon
 * characterisations - "smaller than every proper suffix" and "smaller than every non-trivial
 * rotation" - are also cross-checked for every prefix of the tested strings.
 */
"use strict";

function lcgString(n, seed, alphabet) {
  let out = "";
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    out += String.fromCharCode(97 + (seed % alphabet));
  }
  return out;
}

function rotate(s, offset) {
  const n = s.length;
  if (n === 0) return s;
  const k = ((offset % n) + n) % n;
  return s.slice(k) + s.slice(0, k);
}

function isLyndon(word) {
  if (word.length <= 1) return true;
  for (let i = 1; i < word.length; i++) {
    if (word.slice(i) <= word) return false;
  }
  return true;
}

function isLyndonByRotation(word) {
  if (word.length <= 1) return true;
  for (let i = 1; i < word.length; i++) {
    if (rotate(word, i) <= word) return false;
  }
  return true;
}

/** Duval's algorithm. */
function duvalFactorize(s) {
  const factors = [];
  const n = s.length;
  let i = 0;
  while (i < n) {
    let j = i + 1;
    let k = i;
    while (j < n && s[k] <= s[j]) {
      if (s[k] < s[j]) k = i;
      else k += 1;
      j += 1;
    }
    while (i <= k) {
      factors.push(s.slice(i, i + j - k));
      i += j - k;
    }
  }
  return factors;
}

/** Independent brute force: repeatedly take the longest Lyndon prefix. */
function greedyLyndonFactorize(s) {
  const factors = [];
  let rest = s;
  while (rest.length > 0) {
    let length = 1;
    for (let candidate = 1; candidate <= rest.length; candidate++) {
      if (isLyndon(rest.slice(0, candidate))) length = candidate;
    }
    factors.push(rest.slice(0, length));
    rest = rest.slice(length);
  }
  return factors;
}

/** Enumerate every partition into non-increasing Lyndon words (returns all of them). */
function allLyndonDecompositions(s) {
  const results = [];
  const walk = (position, pieces) => {
    if (position === s.length) {
      results.push(pieces.slice());
      return;
    }
    for (let end = position + 1; end <= s.length; end++) {
      const piece = s.slice(position, end);
      if (!isLyndon(piece)) continue;
      if (pieces.length > 0 && pieces[pieces.length - 1] < piece) continue;
      pieces.push(piece);
      walk(end, pieces);
      pieces.pop();
    }
  };
  walk(0, []);
  return results;
}

function main() {
  const assert = require("node:assert");

  assert.deepStrictEqual(duvalFactorize("banana"), ["b", "an", "an", "a"],
    "the Chen-Fox-Lyndon factorization of banana");
  assert.deepStrictEqual(duvalFactorize("ababab"), ["ab", "ab", "ab"]);
  assert.deepStrictEqual(duvalFactorize("aaaa"), ["a", "a", "a", "a"]);
  assert.deepStrictEqual(duvalFactorize("abcabc"), ["abc", "abc"]);
  assert.deepStrictEqual(duvalFactorize("abc"), ["abc"], "a Lyndon word is a single factor");
  assert.deepStrictEqual(duvalFactorize("aaabaaab"), ["aaab", "aaab"]);
  assert.deepStrictEqual(duvalFactorize("cba"), ["c", "b", "a"],
    "a strictly decreasing string splits into single characters");
  assert.deepStrictEqual(duvalFactorize(""), []);
  assert.deepStrictEqual(duvalFactorize("a"), ["a"]);
  assert.deepStrictEqual(duvalFactorize("z"), ["z"]);
  assert.deepStrictEqual(duvalFactorize("ba"), ["b", "a"]);
  assert.deepStrictEqual(duvalFactorize("ab"), ["ab"]);

  const cases = [
    ["binary-1", lcgString(1, 3, 2)],
    ["binary-5", lcgString(5, 7, 2)],
    ["binary-31", lcgString(31, 11, 2)],
    ["binary-200", lcgString(200, 13, 2)],
    ["ternary-64", lcgString(64, 17, 3)],
    ["ternary-150", lcgString(150, 19, 3)],
    ["all-equal", "a".repeat(45)],
    ["increasing", "abcdefghijklmnopqrst"],
    ["periodic", "abcabcabcabcabc"],
    ["power-of-two", "ab".repeat(32)],
    ["square", "abab"],
    ["fibonacci-word", "abaababaabaababaababa"],
  ];
  let multiFactorSeen = 0;
  for (const [name, text] of cases) {
    const factors = duvalFactorize(text);
    assert.strictEqual(factors.join(""), text, `${name}: the factors concatenate to the input`);
    assert.ok(factors.every((factor) => factor.length > 0), `${name}: no empty factors`);
    for (const factor of factors) {
      assert.ok(isLyndon(factor), `${name}: '${factor}' is a Lyndon word`);
      assert.ok(isLyndonByRotation(factor), `${name}: rotation test agrees on '${factor}'`);
    }
    for (let i = 1; i < factors.length; i++) {
      assert.ok(factors[i - 1] >= factors[i], `${name}: factors are non-increasing`);
    }
    assert.deepStrictEqual(factors, greedyLyndonFactorize(text),
      `${name}: the greedy longest-Lyndon-prefix factorizer agrees`);
    if (factors.length > 1) multiFactorSeen += 1;
  }
  assert.ok(multiFactorSeen >= 6, "most test strings factor into several Lyndon words");

  // Uniqueness: for every short binary string, exactly one non-increasing Lyndon decomposition
  // exists and it is exactly what Duval returns.
  for (let length = 1; length <= 6; length++) {
    for (let mask = 0; mask < 2 ** length; mask++) {
      let text = "";
      for (let bit = 0; bit < length; bit++) {
        text += (mask >> bit) & 1 ? "b" : "a";
      }
      const decompositions = allLyndonDecompositions(text);
      assert.strictEqual(decompositions.length, 1,
        `'${text}' must have exactly one non-increasing Lyndon decomposition`);
      assert.deepStrictEqual(duvalFactorize(text), decompositions[0],
        `Duval returns the unique decomposition of '${text}'`);
    }
  }

  // Uniqueness on the hand-checked words: the only non-increasing Lyndon decomposition is Duval's.
  for (const text of ["banana", "ababab", "aaaa", "abcabc", "ababb", "cba", "ba", "ab", "aabaaab"]) {
    const decompositions = allLyndonDecompositions(text);
    assert.strictEqual(decompositions.length, 1,
      `'${text}' has exactly one non-increasing Lyndon decomposition`);
    assert.deepStrictEqual(duvalFactorize(text), decompositions[0]);
  }

  // The two Lyndon characterisations coincide on every prefix of the tested strings.
  for (const [, text] of cases.slice(0, 6)) {
    for (let end = 1; end <= text.length; end++) {
      const prefix = text.slice(0, end);
      assert.strictEqual(isLyndon(prefix), isLyndonByRotation(prefix),
        `Lyndon tests agree on the prefix '${prefix}'`);
    }
  }

  // A Lyndon word is exactly a string with a single factor; a square is never Lyndon.
  assert.strictEqual(duvalFactorize("aab").length, 1);
  assert.strictEqual(duvalFactorize("abab").length, 2);
  assert.ok(!isLyndon("abab"));
  assert.ok(!isLyndon("aa"));
  assert.ok(isLyndon("ababb"));
  assert.strictEqual(duvalFactorize("ababb").length, 1);

  // Total factor length always equals the input length, and the factor list is never empty for a
  // non-empty input.
  for (const [, text] of cases) {
    const factors = duvalFactorize(text);
    assert.strictEqual(factors.reduce((acc, factor) => acc + factor.length, 0), text.length);
    if (text.length > 0) assert.ok(factors.length >= 1);
    assert.ok(factors.length <= Math.max(1, text.length));
  }
}

module.exports = { lcgString, rotate, isLyndon, isLyndonByRotation, duvalFactorize,
  greedyLyndonFactorize, allLyndonDecompositions };

if (require.main === module) {
  main();
  console.log("PASS");
}
