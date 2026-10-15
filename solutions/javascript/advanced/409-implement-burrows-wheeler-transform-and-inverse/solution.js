/** Task 409 - Implement the Burrows-Wheeler transform and its inverse. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Cyclic BWT: sort all n cyclic rotations of s, take the last column, and record the *primary
 * index* - the row holding the rotation that starts at position 0. Hand anchors: banana ->
 * ("nnbaaa", 3), abab -> ("bbaa", 0), aaaa -> ("aaaa", 0). Reversal is only possible with the
 * primary index: "ab" and "ba" both transform to "ba" but use primary 0 and 1.
 *
 * Inverse: with F = sorted(L) (the first column) the LF-mapping row_of(L[i]) -> position in F
 * (counting equal characters in order) maps a rotation to the rotation starting one position
 * earlier; walking n steps from the primary row and reversing the collected characters
 * rebuilds the string. Ties between equal rotations are resolved by row index (stable sort).
 *
 * Sentinel variant: BWT(s + "$") with a unique smallest character needs no primary index - the
 * row whose last character is "$" is the rotation starting at 0, so the same LF walk recovers
 * s exactly ("banana" -> "annb$aa").
 *
 * Checks: hand values, LF-mapping is a permutation, multiset preserved, both inverses
 * round-trip on LCG strings including the empty string, and the medium case (300 characters).
 */
"use strict";

const SENTINEL = "$";

function bwtCyclic(s) {
  const n = s.length;
  if (n === 0) return { last: "", primary: 0 };
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => {
    for (let k = 0; k < n; k++) {
      const ca = s[(a + k) % n];
      const cb = s[(b + k) % n];
      if (ca !== cb) return ca < cb ? -1 : 1;
    }
    return 0;                                  // equal rotations keep index order (stable)
  });
  let last = "";
  let primary = -1;
  for (let i = 0; i < n; i++) {
    last += s[(order[i] - 1 + n) % n];
    if (order[i] === 0) primary = i;
  }
  return { last, primary };
}

function lfMapping(last) {
  const starts = new Map();
  let total = 0;
  for (const ch of [...new Set(last)].sort()) {
    starts.set(ch, total);
    total += [...last].filter((x) => x === ch).length;
  }
  const seen = new Map();
  const lf = new Array(last.length).fill(0);
  for (let i = 0; i < last.length; i++) {
    const ch = last[i];
    const rank = seen.get(ch) || 0;
    lf[i] = starts.get(ch) + rank;
    seen.set(ch, rank + 1);
  }
  return lf;
}

function inverseBwtCyclic(last, primary) {
  const n = last.length;
  if (n === 0) return "";
  const lf = lfMapping(last);
  let row = primary;
  const collected = [];
  for (let step = 0; step < n; step++) {
    collected.push(last[row]);
    row = lf[row];
  }
  if (row !== primary) throw new Error("LF walk must return to start");
  return collected.reverse().join("");
}

function bwtSentinel(s) {
  if (s.includes(SENTINEL)) throw new Error("text must not contain $");
  return bwtCyclic(s + SENTINEL).last;
}

function inverseBwtSentinel(last) {
  const n = last.length - 1;
  let row = last.indexOf(SENTINEL);
  const lf = lfMapping(last);
  const collected = [];
  for (let step = 0; step < n + 1; step++) {
    collected.push(last[row]);
    row = lf[row];
  }
  return collected.reverse().join("").slice(0, n);   // reversed join is s + "$"
}

function lcgString(length, alphabet, seed) {
  const chars = [];
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    chars.push(alphabet[seed % alphabet.length]);
  }
  return [chars.join(""), seed];
}

function sortedChars(s) {
  return [...s].sort().join("");
}

module.exports = { bwtCyclic, lfMapping, inverseBwtCyclic, bwtSentinel, inverseBwtSentinel, lcgString };

if (require.main === module) {
  const assert = require("node:assert");
  const checkBwt = (s, expectedLast, expectedPrimary) => {
    const result = bwtCyclic(s);
    assert.strictEqual(result.last, expectedLast, `last ${s}`);
    assert.strictEqual(result.primary, expectedPrimary, `primary ${s}`);
  };
  checkBwt("banana", "nnbaaa", 3);
  checkBwt("abab", "bbaa", 0);
  checkBwt("aaaa", "aaaa", 0);
  checkBwt("", "", 0);
  checkBwt("a", "a", 0);
  checkBwt("ba", "ba", 1);
  checkBwt("ab", "ba", 0);

  assert.strictEqual(inverseBwtCyclic("ba", 0), "ab");
  assert.strictEqual(inverseBwtCyclic("ba", 1), "ba");
  const banana = bwtCyclic("banana");
  assert.strictEqual(inverseBwtCyclic(banana.last, banana.primary), "banana");
  const miss = bwtCyclic("mississippi");
  assert.strictEqual(inverseBwtCyclic(miss.last, miss.primary), "mississippi");

  assert.strictEqual(bwtSentinel("banana"), "annb$aa");
  assert.strictEqual(bwtSentinel(""), "$");
  assert.strictEqual(bwtSentinel("a"), "a$");
  assert.strictEqual(inverseBwtSentinel("$"), "");
  assert.strictEqual(inverseBwtSentinel("annb$aa"), "banana");
  assert.strictEqual(inverseBwtSentinel(bwtSentinel("abracadabra")), "abracadabra");

  assert.strictEqual(sortedChars(miss.last), sortedChars("mississippi"));
  const lf = lfMapping(miss.last);
  assert.deepStrictEqual([...lf].sort((a, b) => a - b), Array.from({ length: lf.length }, (_, i) => i));

  let seed = 2718;
  for (const alphabet of ["ab", "abc"]) {
    for (let trial = 0; trial < 40; trial++) {
      let s;
      [s, seed] = lcgString(trial % 41, alphabet, seed);
      const result = bwtCyclic(s);
      assert.strictEqual(sortedChars(result.last), sortedChars(s), s);
      assert.strictEqual(inverseBwtCyclic(result.last, result.primary), s, s);
      assert.strictEqual(inverseBwtSentinel(bwtSentinel(s)), s, s);
    }
  }

  let medium;
  [medium, seed] = lcgString(300, "abc", seed);
  const mediumResult = bwtCyclic(medium);
  assert.strictEqual(mediumResult.last.length, 300);
  assert.strictEqual(inverseBwtCyclic(mediumResult.last, mediumResult.primary), medium);
  assert.strictEqual(inverseBwtSentinel(bwtSentinel(medium)), medium);

  assert.throws(() => bwtSentinel("bad$text"), /must not contain/);
  console.log("PASS");
}
