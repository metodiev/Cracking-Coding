/** Task 316 - Repeated DNA sequences of length 10 (hashing). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

const DNA_LENGTH = 10;
const BASES = "ACGT";

/**
 * All k-letter substrings appearing more than once, sorted; [] if none. Each base is packed
 * into 2 bits, so the rolling key in [0, 4^k) is collision-free and O(1) per step.
 */
function repeatedDnaSequences(s, k = DNA_LENGTH) {
  const result = [];
  if (k <= 0 || s.length < k) return result;
  const mask = 2 ** (2 * k) - 1;         // keeps only the last k bases
  const seen = new Set();
  const repeated = new Set();
  let key = 0;
  for (let i = 0; i < s.length; i++) {
    key = ((key << 2) | BASES.indexOf(s[i])) & mask;
    if (i >= k - 1) {
      const word = s.slice(i - k + 1, i + 1);
      if (seen.has(key)) repeated.add(word);
      else seen.add(key);
    }
  }
  return [...repeated].sort();
}

/** Counting map over every substring; the independent oracle. */
function repeatedDnaNaive(s, k = DNA_LENGTH) {
  if (k <= 0 || s.length < k) return [];
  const counts = new Map();
  for (let i = 0; i + k <= s.length; i++) {
    const word = s.slice(i, i + k);
    counts.set(word, (counts.get(word) || 0) + 1);
  }
  const result = [];
  for (const [word, count] of counts) if (count > 1) result.push(word);
  return result.sort();
}

/** Occurrences counted with overlaps (non-overlapping counting would undercount repeats). */
function overlappingCount(s, word) {
  let count = 0;
  for (let i = 0; i + word.length <= s.length; i++) {
    if (s.slice(i, i + word.length) === word) count++;
  }
  return count;
}

function lcgDna(length, count, seed = 3160) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += BASES[seed % 4];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { DNA_LENGTH, BASES, repeatedDnaSequences, repeatedDnaNaive, overlappingCount, lcgDna };

if (require.main === module) {
  const assert = require("node:assert");

  const classic = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT";
  assert.deepStrictEqual(repeatedDnaSequences(classic), ["AAAAACCCCC", "CCCCCAAAAA"]);
  assert.deepStrictEqual(repeatedDnaNaive(classic), ["AAAAACCCCC", "CCCCCAAAAA"]);

  assert.deepStrictEqual(repeatedDnaSequences("ACGTACGTAC"), []);
  assert.deepStrictEqual(repeatedDnaSequences("AAAAAAAAAAA"), ["AAAAAAAAAA"]);
  assert.deepStrictEqual(repeatedDnaSequences("AAAAAAAAAA"), []);
  assert.deepStrictEqual(repeatedDnaSequences("ACGT"), []);
  assert.deepStrictEqual(repeatedDnaSequences(""), []);
  assert.deepStrictEqual(repeatedDnaSequences("ACGTACGTAC", 4), ["ACGT", "CGTA", "GTAC"]);
  assert.deepStrictEqual(repeatedDnaSequences("ACGTACGTAC", 0), []);
  assert.deepStrictEqual(repeatedDnaSequences("ACGTACGTAC", 20), []);

  const samples = [classic, "AAAAAAAAAAAA", "ACGTACGTACGTACGTACGT", ...lcgDna(40, 40), ...lcgDna(20, 20)];
  for (const s of samples) {
    const found = repeatedDnaSequences(s);
    assert.deepStrictEqual(found, repeatedDnaNaive(s), s);
    assert.deepStrictEqual(found, [...found].sort(), s);
    assert.strictEqual(new Set(found).size, found.length, s);
    for (const word of found) {                            // every report really repeats
      assert.strictEqual(word.length, DNA_LENGTH, s);
      assert.ok(overlappingCount(s, word) > 1, `${s} ${word}`);
    }
  }

  // The result is exactly the set of 10-mers occurring at least twice
  for (const s of lcgDna(30, 20)) {
    const expect = [];
    for (let start = 0; start + DNA_LENGTH <= s.length; start++) {
      const word = s.slice(start, start + DNA_LENGTH);
      if (overlappingCount(s, word) > 1 && !expect.includes(word)) expect.push(word);
    }
    assert.deepStrictEqual(repeatedDnaSequences(s), expect.sort(), s);
  }

  console.log("PASS");
}
