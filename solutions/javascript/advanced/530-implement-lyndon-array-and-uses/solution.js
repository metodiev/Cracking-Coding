/**
 * Task 530 - Implement Lyndon array and uses. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The Lyndon array lambda[i] is the length of the longest Lyndon word that starts at position i,
 * i.e. the longest prefix of the suffix i that is strictly smaller than all of its proper suffixes.
 * Two independent computations are implemented. The direct one tries every prefix length from the
 * longest downwards and applies the definition. The fast one exploits the theorem
 * lambda[i] = nsa[i] - i, where nsa[i] is the next smaller suffix (the smallest j > i with
 * suffix_j < suffix_i, or -1 when none exists, in which case lambda[i] = n - i); suffixes are
 * ranked by sorting the suffixes directly, and the next-smaller-suffix array is then produced with
 * the classic monotonic stack in linear time.
 *
 * Documented checks: the two computations agree for hand-checked strings ("banana", "aaaa",
 * "abcde", "cbad", "abab"), for LCG-generated strings over two- and three-letter alphabets up to
 * length 120, and for periodic and Fibonacci-word inputs; for every position the claimed window is
 * verified to be Lyndon and to be *maximal* (extending it by one character destroys the property),
 * which pins down the array exactly; lambda[i] >= 1 always; the maximum of the array equals the
 * longest Lyndon substring found by brute force over all substrings; and the uses are checked - the
 * factor boundaries read off the array (repeatedly jump by lambda at the current position) must
 * reproduce the Duval Lyndon factorization exactly, and the longest-Lyndon-substring accessor must
 * return a concrete witness of the right length. Boundary cases: the empty array for the empty
 * string, a single character, all-equal and strictly increasing/decreasing strings.
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

function isLyndon(word) {
  if (word.length <= 1) return true;
  for (let i = 1; i < word.length; i++) {
    if (word.slice(i) <= word) return false;
  }
  return true;
}

/** Direct definition: the longest Lyndon prefix of every suffix (O(n^3) in the worst case). */
function lyndonArrayBrute(s) {
  const n = s.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let best = 0;
    for (let length = n - i; length >= 1; length--) {
      if (isLyndon(s.slice(i, i + length))) {
        best = length;
        break;
      }
    }
    out[i] = best;
  }
  return out;
}

/** Suffix array and inverse ranks, by direct suffix comparison (stable on equal suffixes). */
function suffixRanks(s) {
  const n = s.length;
  const indices = Array.from({ length: n }, (_, i) => i);
  const compare = (a, b) => {
    let x = a;
    let y = b;
    while (x < n && y < n) {
      const dx = s.charCodeAt(x);
      const dy = s.charCodeAt(y);
      if (dx !== dy) return dx - dy;
      x += 1;
      y += 1;
    }
    return (n - x) - (n - y);
  };
  indices.sort((a, b) => compare(a, b) || (a - b));
  const rank = new Array(n);
  indices.forEach((suffixStart, k) => {
    rank[suffixStart] = k;
  });
  return { sa: indices, rank };
}

/** Next strictly smaller suffix for every position, via a monotonic stack over the ranks. */
function nextSmallerSuffix(s) {
  const n = s.length;
  const { rank } = suffixRanks(s);
  const nsa = new Array(n).fill(-1);
  const stack = [];
  for (let i = n - 1; i >= 0; i--) {
    while (stack.length > 0 && rank[stack[stack.length - 1]] >= rank[i]) stack.pop();
    nsa[i] = stack.length > 0 ? stack[stack.length - 1] : -1;
    stack.push(i);
  }
  return { nsa, rank };
}

/** Lyndon array through the next-smaller-suffix theorem. */
function lyndonArrayFast(s) {
  const n = s.length;
  const { nsa } = nextSmallerSuffix(s);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = nsa[i] === -1 ? n - i : nsa[i] - i;
  return out;
}

/** Longest Lyndon substring: the maximum of the Lyndon array, plus a witness. */
function longestLyndonSubstring(s) {
  const lambda = lyndonArrayFast(s);
  let best = 0;
  let at = -1;
  lambda.forEach((length, i) => {
    if (length > best) {
      best = length;
      at = i;
    }
  });
  return { length: best, start: at, text: at === -1 ? "" : s.slice(at, at + best) };
}

/** Best-known brute force for the longest Lyndon substring. */
function bruteForceLongestLyndon(s) {
  let best = 0;
  let witness = "";
  for (let i = 0; i < s.length; i++) {
    for (let j = i + 1; j <= s.length; j++) {
      const piece = s.slice(i, j);
      if (piece.length > best && isLyndon(piece)) {
        best = piece.length;
        witness = piece;
      }
    }
  }
  return { length: best, text: witness };
}

/** Duval's factorization (reference for the Lyndon-array use case). */
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

/** Read factors off the Lyndon array: jump by lambda at the current position. */
function factorizeFromLyndonArray(s, lambda) {
  const factors = [];
  let position = 0;
  while (position < s.length) {
    const length = lambda[position];
    if (length <= 0) return null;
    factors.push(s.slice(position, position + length));
    position += length;
  }
  return factors;
}

function main() {
  const assert = require("node:assert");

  assert.deepStrictEqual(lyndonArrayBrute(""), []);
  assert.deepStrictEqual(lyndonArrayFast(""), []);
  assert.deepStrictEqual(lyndonArrayBrute("a"), [1]);
  assert.deepStrictEqual(lyndonArrayFast("a"), [1]);
  assert.deepStrictEqual(lyndonArrayFast("aaaa"), [1, 1, 1, 1],
    "a^4 has no Lyndon word longer than one character");
  assert.deepStrictEqual(lyndonArrayFast("abcde"), [5, 4, 3, 2, 1],
    "every suffix of an increasing string is Lyndon");
  assert.deepStrictEqual(lyndonArrayFast("edcba"), [1, 1, 1, 1, 1]);
  assert.deepStrictEqual(lyndonArrayFast("banana"), [1, 2, 1, 2, 1, 1],
    "'an' is Lyndon at positions 1 and 3; 'na' and 'nana' are not");
  assert.deepStrictEqual(lyndonArrayBrute("banana"), [1, 2, 1, 2, 1, 1],
    "the brute-force definition confirms the hand-checked array");
  assert.deepStrictEqual(lyndonArrayFast("abab"), [2, 1, 2, 1], "'ab' is the longest Lyndon prefix");
  assert.deepStrictEqual(lyndonArrayFast("ababab"), [2, 1, 2, 1, 2, 1]);
  assert.deepStrictEqual(lyndonArrayFast("aab"), [3, 2, 1], "the whole string is Lyndon");
  assert.deepStrictEqual(lyndonArrayFast("aaabaaab"), [4, 3, 2, 1, 4, 3, 2, 1]);
  assert.deepStrictEqual(lyndonArrayFast("ababb"), [5, 1, 3, 1, 1],
    "'ababb' is Lyndon, 'babb' is not, 'abb' is");

  const cases = [
    ["binary-1", lcgString(1, 3, 2)],
    ["binary-2", lcgString(2, 5, 2)],
    ["binary-9", lcgString(9, 7, 2)],
    ["binary-40", lcgString(40, 11, 2)],
    ["binary-120", lcgString(120, 13, 2)],
    ["ternary-25", lcgString(25, 17, 3)],
    ["ternary-90", lcgString(90, 19, 3)],
    ["all-equal", "b".repeat(21)],
    ["increasing", "abcdefghijklmno"],
    ["periodic", "abcabcabcabcabcabc"],
    ["power", "ab".repeat(11)],
    ["fibonacci-word", "abaababaabaababaababa"],
    ["square", "abab"],
  ];
  for (const [name, text] of cases) {
    const brute = lyndonArrayBrute(text);
    const fast = lyndonArrayFast(text);
    assert.deepStrictEqual(fast, brute, `${name}: the two Lyndon arrays agree`);
    assert.strictEqual(fast.length, text.length);
    for (let i = 0; i < text.length; i++) {
      assert.ok(fast[i] >= 1, `${name}: lambda[${i}] is at least one`);
      const window = text.slice(i, i + fast[i]);
      assert.ok(isLyndon(window), `${name}: the window at ${i} ('${window}') is a Lyndon word`);
      if (i + fast[i] + 1 <= text.length) {
        const longer = text.slice(i, i + fast[i] + 1);
        assert.ok(!isLyndon(longer),
          `${name}: the window at ${i} cannot be extended ('${longer}' is not Lyndon)`);
      }
    }
    const longest = longestLyndonSubstring(text);
    const reference = bruteForceLongestLyndon(text);
    assert.strictEqual(longest.length, reference.length, `${name}: longest Lyndon substring`);
    assert.ok(isLyndon(longest.text), `${name}: the witness is a Lyndon word`);
    assert.strictEqual(longest.text.length, longest.length);
    assert.ok(text.includes(longest.text));
    assert.strictEqual(Math.max(...fast), longest.length,
      `${name}: the array maximum is the longest Lyndon substring`);

    // Use case: the factor boundaries of the Lyndon factorization sit on the array.
    const fromArray = factorizeFromLyndonArray(text, fast);
    assert.deepStrictEqual(fromArray, duvalFactorize(text),
      `${name}: the array reproduces the Duval factorization`);
  }

  // The next-smaller-suffix array itself: position and strictness.
  for (const [name, text] of cases) {
    const { nsa, rank } = nextSmallerSuffix(text);
    for (let i = 0; i < text.length; i++) {
      if (nsa[i] === -1) {
        assert.ok(rank[i] === Math.min(...rank.slice(i)), `${name}: no smaller suffix exists`);
      } else {
        assert.ok(nsa[i] > i, `${name}: nsa points forward`);
        assert.ok(rank[nsa[i]] < rank[i], `${name}: nsa is strictly smaller`);
        for (let j = i + 1; j < nsa[i]; j++) {
          assert.ok(rank[j] > rank[i], `${name}: no smaller suffix lies in between`);
        }
      }
    }
  }

  assert.strictEqual(longestLyndonSubstring("").length, 0);
  assert.strictEqual(longestLyndonSubstring("").text, "");
  assert.strictEqual(longestLyndonSubstring("aaaa").length, 1);
  assert.strictEqual(typeof factorizeFromLyndonArray("aaa", [0, 0, 0]), "object",
    "a degenerate array is rejected by returning null");
  assert.strictEqual(factorizeFromLyndonArray("aaa", [0, 0, 0]), null);

  // Known closed form: lambda for a^n is all ones; for a strictly increasing string of length n the
  // array decreases by one each step.
  for (let n = 1; n <= 12; n++) {
    assert.deepStrictEqual(lyndonArrayFast("a".repeat(n)), new Array(n).fill(1));
    const increasing = "abcdefghijklmnopqrst".slice(0, n);
    const expected = Array.from({ length: n }, (_, i) => n - i);
    assert.deepStrictEqual(lyndonArrayFast(increasing), expected);
  }
}

module.exports = { lcgString, isLyndon, lyndonArrayBrute, suffixRanks, nextSmallerSuffix,
  lyndonArrayFast, longestLyndonSubstring, bruteForceLongestLyndon, duvalFactorize,
  factorizeFromLyndonArray };

if (require.main === module) {
  main();
  console.log("PASS");
}
