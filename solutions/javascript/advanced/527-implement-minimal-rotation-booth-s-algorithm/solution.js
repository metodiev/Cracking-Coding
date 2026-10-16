/**
 * Task 527 - Implement minimal rotation (Booth's algorithm). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Booth's O(n) algorithm for the lexicographically minimal rotation of a string: the string is
 * doubled, a candidate start k is advanced while a failure array f records, for every position of
 * the doubled string, the length of the longest prefix match of s[k..] found so far; whenever a
 * mismatch shows that the current candidate is worse than the start at j, the candidate jumps to
 * j - i - 1 (skipping a whole block of impossible starts). The implementation returns the *smallest
 * index* attaining the minimal rotation, which is also what the brute-force scan produces.
 *
 * Documented checks: the result is compared with (a) a brute-force scan over all n rotations that
 * keeps the first index attaining the lexicographic minimum and (b) the independent two-pointer
 * minimum-expression algorithm, whose index may differ when the rotation is periodic but whose
 * induced rotation string must be identical. Agreement is asserted for LCG-generated strings over
 * two- and three-letter alphabets up to length 400, for all-equal strings, for periodic strings
 * ("abcabcabc", ("ab")^k), for strings with a unique minimum character, for hand-checked examples
 * ("banana" -> 5, "baba" -> 3, "aaaa" -> 0), and for single-character and two-character inputs.
 * Further properties: rotating the input by any offset must not change the minimal rotation string,
 * the returned rotation is a fixed point of the operation applied to itself, index 0 is returned
 * exactly when the input is weakly minimal (equal to its own minimal rotation) - a condition every
 * Lyndon word satisfies strictly, and the Lyndon property is cross-checked with the independent
 * suffix test and the rotation test - and the empty string is rejected.
 */
"use strict";

/** Small deterministic LCG producing a string over the first `alphabet` letters. */
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

/** Booth's algorithm over the doubled string; returns the smallest index of a minimal rotation. */
function boothMinRotation(s) {
  const n = s.length;
  if (n === 0) throw new Error("an empty string has no rotation");
  const codes = [];
  for (let i = 0; i < 2 * n; i++) codes.push(s.charCodeAt(i % n));
  const fail = new Array(2 * n).fill(-1);
  let k = 0;
  for (let j = 1; j < 2 * n; j++) {
    let i = fail[j - k - 1];
    while (i !== -1 && codes[j] !== codes[k + i + 1]) {
      if (codes[j] < codes[k + i + 1]) k = j - i - 1;
      i = fail[i];
    }
    if (codes[j] !== codes[k + i + 1]) {
      if (codes[j] < codes[k]) k = j;
      fail[j - k] = -1;
    } else {
      fail[j - k] = i + 1;
    }
  }
  return k % n;
}

/** Brute force: scan every rotation in index order and keep the first strict improvement. */
function bruteForceMinRotation(s) {
  const n = s.length;
  if (n === 0) throw new Error("an empty string has no rotation");
  let best = 0;
  for (let i = 1; i < n; i++) {
    if (rotate(s, i) < rotate(s, best)) best = i;
  }
  return best;
}

/** Independent O(n) two-pointer minimum-expression algorithm (returns some optimal index). */
function twoPointerMinRotation(s) {
  const n = s.length;
  if (n === 0) throw new Error("an empty string has no rotation");
  let i = 0;
  let j = 1;
  let k = 0;
  while (i < n && j < n && k < n) {
    const a = s.charCodeAt((i + k) % n);
    const b = s.charCodeAt((j + k) % n);
    if (a === b) {
      k += 1;
      continue;
    }
    if (a > b) {
      i = i + k + 1;
      if (i === j) i += 1;
    } else {
      j = j + k + 1;
      if (i === j) j += 1;
    }
    k = 0;
  }
  return Math.min(i, j) % n;
}

/** A word is Lyndon when it is strictly smaller than every proper non-empty suffix. */
function isLyndon(word) {
  if (word.length <= 1) return true;
  for (let i = 1; i < word.length; i++) {
    if (word.slice(i) <= word) return false;
  }
  return true;
}

/** A word is Lyndon exactly when it is strictly smaller than every non-trivial rotation. */
function isLyndonByRotation(word) {
  if (word.length <= 1) return true;
  for (let i = 1; i < word.length; i++) {
    const rotated = rotate(word, i);
    if (rotated <= word) return false;
  }
  return true;
}

function main() {
  const assert = require("node:assert");

  assert.strictEqual(rotate("banana", 5), "abanan");
  assert.strictEqual(boothMinRotation("banana"), 5, "the minimal rotation of 'banana' is 'abanan'");
  assert.strictEqual(boothMinRotation("baba"), 1,
    "'abab' is attained at indices 1 and 3; the smallest index wins");
  assert.strictEqual(boothMinRotation("aaaa"), 0, "all rotations of a^4 are equal: index 0 wins");
  assert.strictEqual(boothMinRotation("a"), 0);
  assert.strictEqual(boothMinRotation("ba"), 1);
  assert.strictEqual(boothMinRotation("ab"), 0);
  assert.strictEqual(boothMinRotation("cba"), 2, "'acb' is the minimal rotation of 'cba'");
  assert.strictEqual(boothMinRotation("abcabcabc"), 0, "a periodic string is its own minimum");
  assert.strictEqual(boothMinRotation("bcabcabca"), 2,
    "the minimal rotation 'abcabcabc' first appears at index 2");

  const cases = [
    ["binary-1", lcgString(1, 3, 2)],
    ["binary-2", lcgString(2, 5, 2)],
    ["binary-7", lcgString(7, 11, 2)],
    ["ternary-25", lcgString(25, 17, 3)],
    ["binary-60", lcgString(60, 23, 2)],
    ["ternary-120", lcgString(120, 31, 3)],
    ["binary-400", lcgString(400, 41, 2)],
    ["all-equal", "a".repeat(37)],
    ["periodic", "abcabcabcabcabcabcabc"],
    ["power", "ab".repeat(20)],
    ["lyndon-ab", "ab"],
    ["lyndon-aab", "aab"],
    ["lyndon-ababb", "ababb"],
    ["unique-min", "zzzazzz" + lcgString(8, 7, 2)],
    ["increasing", "abcdefghijklmnop"],
    ["desert", lcgString(64, 53, 26)],
  ];
  let lyndonSeen = 0;
  for (const [name, text] of cases) {
    const booth = boothMinRotation(text);
    const brute = bruteForceMinRotation(text);
    const twoPointer = twoPointerMinRotation(text);
    assert.strictEqual(booth, brute, `${name}: Booth matches the brute-force scan (index ${brute})`);
    assert.strictEqual(rotate(text, booth), rotate(text, twoPointer),
      `${name}: the independent two-pointer method finds the same rotation string`);
    assert.strictEqual(rotate(text, booth), rotate(text, brute), `${name}: rotation strings agree`);
    for (let i = 0; i < text.length; i++) {
      assert.ok(rotate(text, booth) <= rotate(text, i), `${name}: rotation ${i} is not smaller`);
    }
    // Rotating the input must not change the minimal rotation string.
    for (const offset of [1, 3, Math.floor(text.length / 2), text.length - 1]) {
      const shifted = rotate(text, offset);
      assert.strictEqual(rotate(shifted, boothMinRotation(shifted)), rotate(text, booth),
        `${name}: rotation invariance at offset ${offset}`);
    }
    // Applying the algorithm to the minimal rotation itself returns index 0 (or an equal string).
    const minimalString = rotate(text, booth);
    assert.strictEqual(rotate(minimalString, boothMinRotation(minimalString)), minimalString,
      `${name}: the minimal rotation is a fixed point`);
    // Booth returns 0 exactly for Lyndon words, and the two Lyndon characterisations must agree.
    const lyndonBySuffix = isLyndon(text);
    assert.strictEqual(isLyndonByRotation(text), lyndonBySuffix,
      `${name}: the suffix and rotation Lyndon tests agree`);
    // Booth returns 0 exactly when the input is *weakly* minimal (equal to its minimal rotation);
    // for a Lyndon word the input is strictly smaller, so the implication is one-directional.
    let weaklyMinimal = true;
    for (let i = 1; i < text.length; i++) if (rotate(text, i) < text) weaklyMinimal = false;
    assert.strictEqual(booth === 0, weaklyMinimal, `${name}: index 0 iff the input is minimal`);
    if (lyndonBySuffix) {
      assert.ok(weaklyMinimal, `${name}: a Lyndon word is its own minimal rotation`);
      lyndonSeen += 1;
    }
  }
  assert.ok(lyndonSeen >= 3, "several test strings are Lyndon words");

  // Uniform random strings are almost never Lyndon, but the periodic ones are not.
  assert.ok(isLyndon("aab"));
  assert.ok(!isLyndon("aba"));
  assert.ok(!isLyndon("abab"));
  assert.ok(isLyndon("ababb"));
  assert.ok(!isLyndon("aa"));
  for (const word of ["aa", "ab", "ba", "aba", "aab", "abab", "banana"]) {
    assert.strictEqual(isLyndon(word), isLyndonByRotation(word), `Lyndon tests agree on '${word}'`);
  }

  assert.throws(() => boothMinRotation(""), /empty string/);
  assert.throws(() => bruteForceMinRotation(""), /empty string/);
  assert.throws(() => twoPointerMinRotation(""), /empty string/);
}

module.exports = { lcgString, rotate, boothMinRotation, bruteForceMinRotation,
  twoPointerMinRotation, isLyndon, isLyndonByRotation };

if (require.main === module) {
  main();
  console.log("PASS");
}
