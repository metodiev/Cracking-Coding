/** Task 410 - FM-index: basic concept and small demo. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Structure for text s: build t = s + "$" with a unique smallest sentinel, store
 *   * the Burrows-Wheeler last column L of t,
 *   * the C table: C[c] = number of characters of t strictly smaller than c,
 *   * rank samples: for every character, the cumulative count in L at each block boundary
 *     (block size configurable); occ(c, i) = sample + direct scan of the residual (< block).
 *
 * Backward search walks the pattern from the right: starting from the full row range [0, n+1),
 * for each character c set lo = C[c] + occ(c, lo) and hi = C[c] + occ(c, hi). The interval is
 * the contiguous range of rows whose rotations start with the pattern suffix processed so far,
 * so the final width counts the matches; an empty interval means no match.
 *
 * LF-mapping (the inverse of the sorted first column F): LF[i] = C[L[i]] + rank of L[i] among
 * equal characters before i. Following LF moves from a row to the rotation starting one
 * position earlier, so walking LF from a match row until row 0 (the "$..." rotation) gives the
 * position: after `steps` LF applications the match starts at (n + steps) mod (n + 1) in t.
 *
 * Checks: hand values for banana ("ana" -> 2, "na" -> 2, "a" -> 3, locate positions [1,3]
 * etc.), occ() against direct counts, LF is one cycle, LCG texts/patterns agree with a naive
 * scan for both count and locate, and several block sizes give identical answers.
 */
"use strict";

const FM_SENTINEL = "$";

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
    return 0;
  });
  let last = "";
  for (let i = 0; i < n; i++) last += s[(order[i] - 1 + n) % n];
  return { last, primary: 0 };
}

function lfMapping(last) {
  const stamps = new Map();
  let total = 0;
  for (const ch of [...new Set(last)].sort()) {
    stamps.set(ch, total);
    total += [...last].filter((x) => x === ch).length;
  }
  const seen = new Map();
  const lf = new Array(last.length).fill(0);
  for (let i = 0; i < last.length; i++) {
    const ch = last[i];
    const rank = seen.get(ch) || 0;
    lf[i] = stamps.get(ch) + rank;
    seen.set(ch, rank + 1);
  }
  return lf;
}

class FmIndex {
  constructor(text, block = 4) {
    this.text = text;
    this.block = block;
    this.t = text + FM_SENTINEL;
    this.last = bwtCyclic(this.t).last;
    this.cTable = new Map();
    for (const ch of [...new Set([...this.t])].sort()) {
      let smaller = 0;
      for (const x of this.t) if (x < ch) smaller++;
      this.cTable.set(ch, smaller);
    }
    this.samples = new Map();
    for (const ch of [...new Set([...this.t])].sort()) {
      const row = new Array(Math.floor(this.t.length / block) + 1).fill(0);
      let count = 0;
      for (let pos = 0; pos <= this.t.length; pos++) {
        if (pos % block === 0) row[pos / block] = count;
        if (pos < this.t.length && this.last[pos] === ch) count++;
      }
      this.samples.set(ch, row);
    }
    this.lf = lfMapping(this.last);
  }

  occ(ch, i) {
    const row = this.samples.get(ch);
    const start = Math.floor(i / this.block) * this.block;
    let count = row[Math.floor(i / this.block)];
    for (let pos = start; pos < i; pos++) if (this.last[pos] === ch) count++;
    return count;
  }

  backwardSearch(pattern) {
    let lo = 0;
    let hi = this.t.length;
    for (let i = pattern.length - 1; i >= 0; i--) {
      const ch = pattern[i];
      if (!this.cTable.has(ch)) return [lo, lo];
      lo = this.cTable.get(ch) + this.occ(ch, lo);
      hi = this.cTable.get(ch) + this.occ(ch, hi);
      if (lo >= hi) return [lo, hi];
    }
    return [lo, hi];
  }

  count(pattern) {
    if (pattern.length === 0) return 0;             // documented: empty pattern is not searched
    const [lo, hi] = this.backwardSearch(pattern);
    return hi - lo;
  }

  locate(pattern) {
    if (pattern.length === 0) return [];
    const [lo, hi] = this.backwardSearch(pattern);
    const positions = [];
    for (let row = lo; row < hi; row++) {
      let steps = 0;
      let current = row;
      while (current !== 0) {
        current = this.lf[current];
        steps++;
      }
      positions.push((this.text.length + steps) % this.t.length);
    }
    return positions.sort((a, b) => a - b);
  }
}

function naiveCount(text, pattern) {
  let count = 0;
  for (let i = 0; i + pattern.length <= text.length; i++) {
    if (text.startsWith(pattern, i)) count++;
  }
  return count;
}

function naiveLocate(text, pattern) {
  const positions = [];
  for (let i = 0; i + pattern.length <= text.length; i++) {
    if (text.startsWith(pattern, i)) positions.push(i);
  }
  return positions;
}

function lcgString(length, alphabet, seed) {
  const chars = [];
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    chars.push(alphabet[seed % alphabet.length]);
  }
  return [chars.join(""), seed];
}

function checkAgainstNaive(text, patterns) {
  const assert = require("node:assert");
  const index = new FmIndex(text);
  for (const pattern of patterns) {
    assert.strictEqual(index.count(pattern), naiveCount(text, pattern), `${text}/${pattern}`);
    assert.deepStrictEqual(index.locate(pattern), naiveLocate(text, pattern), `${text}/${pattern}`);
  }
}

module.exports = { FmIndex, bwtCyclic, lfMapping, naiveCount, naiveLocate, lcgString, checkAgainstNaive };

if (require.main === module) {
  const assert = require("node:assert");
  const index = new FmIndex("banana");
  assert.strictEqual(index.t, "banana$");
  assert.strictEqual(index.last, "annb$aa");
  assert.deepStrictEqual([...index.cTable].sort(), [["$", 0], ["a", 1], ["b", 4], ["n", 5]]);
  for (const ch of ["$", "a", "b", "n"]) {
    for (let i = 0; i <= index.t.length; i++) {
      assert.strictEqual(index.occ(ch, i), [...index.last.slice(0, i)].filter((x) => x === ch).length,
        `occ ${ch}/${i}`);
    }
  }

  assert.strictEqual(index.count("ana"), 2);
  assert.strictEqual(index.count("na"), 2);
  assert.strictEqual(index.count("a"), 3);
  assert.strictEqual(index.count("banana"), 1);
  assert.strictEqual(index.count("nana"), 1);
  assert.strictEqual(index.count("anana"), 1);
  assert.strictEqual(index.count("bananas"), 0);
  assert.strictEqual(index.count("x"), 0);
  assert.strictEqual(index.count(""), 0);
  assert.deepStrictEqual(index.locate("ana"), [1, 3]);
  assert.deepStrictEqual(index.locate("na"), [2, 4]);
  assert.deepStrictEqual(index.locate("a"), [1, 3, 5]);
  assert.deepStrictEqual(index.locate("banana"), [0]);
  assert.deepStrictEqual(index.locate("nana"), [2]);
  assert.deepStrictEqual(index.locate("anana"), [1]);
  assert.deepStrictEqual(index.locate("bananas"), []);
  assert.deepStrictEqual(index.locate("x"), []);
  assert.deepStrictEqual(index.locate(""), []);
  assert.deepStrictEqual([...index.lf].sort((a, b) => a - b), Array.from({ length: index.lf.length }, (_, i) => i));
  let row = 0;
  for (let step = 0; step < index.t.length; step++) row = index.lf[row];
  assert.strictEqual(row, 0);

  const reference = new FmIndex("banana", 1);
  for (const block of [1, 2, 3, 5, 8, 16]) {
    const variant = new FmIndex("banana", block);
    for (const pattern of ["a", "an", "ana", "anan", "n", "na", "banana", "zz"]) {
      assert.strictEqual(variant.count(pattern), reference.count(pattern), `block ${block}`);
      assert.deepStrictEqual(variant.locate(pattern), reference.locate(pattern), `block ${block}`);
    }
    for (const ch of ["$", "a", "b", "n"]) {
      for (let i = 0; i <= variant.t.length; i++) {
        assert.strictEqual(variant.occ(ch, i), reference.occ(ch, i), `block ${block}/${ch}/${i}`);
      }
    }
  }

  checkAgainstNaive("mississippi", ["i", "is", "issi", "ssi", "ppi", "sipp", "mississippix", "q"]);
  checkAgainstNaive("", ["a", "x"]);
  checkAgainstNaive("a", ["a", "aa", "b"]);
  checkAgainstNaive("aaaa", ["a", "aa", "aaa", "aaaa", "aaaaa", "b"]);

  let seed = 2718;
  for (let trial = 0; trial < 30; trial++) {
    let text;
    [text, seed] = lcgString(30, "abc", seed);
    const patterns = [];
    for (let i = 0; i < text.length; i++) {
      for (let j = i + 1; j <= Math.min(text.length, i + 4); j++) patterns.push(text.slice(i, j));
    }
    patterns.push(text, text.slice(0, -1), "d" + text.slice(0, 5), "zz", "q");
    checkAgainstNaive(text, patterns);
  }

  let medium;
  [medium, seed] = lcgString(300, "abc", seed);
  const mediumIndex = new FmIndex(medium, 7);
  for (const pattern of ["abc", "cba", "cab", "abcabc", "zzz", medium.slice(150, 160), medium]) {
    assert.strictEqual(mediumIndex.count(pattern), naiveCount(medium, pattern), pattern);
    assert.deepStrictEqual(mediumIndex.locate(pattern), naiveLocate(medium, pattern), pattern);
  }
  console.log("PASS");
}
