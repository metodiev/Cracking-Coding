/** Task 234 - Group anagrams (count signature as key). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

/**
 * Key each word by its 26-entry count signature (how many 'a's, 'b's, ... it
 * contains) instead of the sorted word: O(n * L) time, no sorting. Inputs are
 * assumed lowercase. Ordering is identical to task 233 (first-seen key order,
 * input order within a group), so both approaches return equal output.
 */
function groupAnagramsSignature(words) {
  const groups = new Map();
  for (const word of words) {
    const counts = new Array(26).fill(0);
    for (const ch of word) counts[ch.charCodeAt(0) - 97]++;
    const key = counts.join(",");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(word);
  }
  return [...groups.values()];
}

/** Task 233's sorted-key variant, kept here only for the agreement checks. */
function groupAnagramsSorted(words) {
  const groups = new Map();
  for (const word of words) {
    const key = [...word].sort().join("");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(word);
  }
  return [...groups.values()];
}

module.exports = { groupAnagramsSignature, groupAnagramsSorted };

if (require.main === module) {
  const assert = require("node:assert");
  const cases = [
    ["eat", "tea", "tan", "ate", "nat", "bat"],
    ["listen", "silent", "enlist", "google", "banana", "banana"],
    ["", "a", "", "a", "ab", "ba"],
    ["abc", "abc"],
    ["z"],
    [],
  ];
  for (const words of cases) {                  // both approaches must agree
    assert.deepStrictEqual(groupAnagramsSignature(words), groupAnagramsSorted(words));
  }

  assert.deepStrictEqual(groupAnagramsSignature(["eat", "tea", "tan", "ate", "nat", "bat"]),
    [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]);
  assert.deepStrictEqual(groupAnagramsSignature([]), []);
  assert.deepStrictEqual(groupAnagramsSignature([""]), [[""]]);
  assert.deepStrictEqual(groupAnagramsSignature(["a", "b", "a"]), [["a", "a"], ["b"]]);
  assert.deepStrictEqual(groupAnagramsSignature(["ab", "ba", "abc", "cab", "bca"]),
    [["ab", "ba"], ["abc", "cab", "bca"]]);

  // deterministic LCG so the random agreement check is reproducible
  let seed = 234;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  for (let test = 0; test < 200; test++) {
    const count = Math.floor(rand() * 11);
    const words = [];
    for (let i = 0; i < count; i++) {
      let word = "";
      const len = Math.floor(rand() * 7);
      for (let j = 0; j < len; j++) word += "abc"[Math.floor(rand() * 3)];
      words.push(word);
    }
    assert.deepStrictEqual(groupAnagramsSignature(words), groupAnagramsSorted(words));
  }
  console.log("PASS");
}
