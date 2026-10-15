/** Task 309 - Longest common prefix among many strings. (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function longestCommonPrefixSorted(strs) {
  if (strs.length === 0) return "";
  let smallest = strs[0];
  let largest = strs[0];
  for (const s of strs) {
    if (s < smallest) smallest = s;
    if (s > largest) largest = s;
  }
  let i = 0;
  while (i < smallest.length && i < largest.length && smallest[i] === largest[i]) i++;
  return smallest.slice(0, i);
}

function longestCommonPrefixVertical(strs) {
  if (strs.length === 0) return "";
  for (let column = 0; column < strs[0].length; column++) {
    const ch = strs[0][column];
    for (let k = 1; k < strs.length; k++) {
      if (column >= strs[k].length || strs[k][column] !== ch) return strs[0].slice(0, column);
    }
  }
  return strs[0];
}

function longestCommonPrefixTrie(strs) {
  const root = new Map();
  for (const word of strs) {
    let node = root;
    for (const ch of word) {
      if (!node.has(ch)) node.set(ch, new Map());
      node = node.get(ch);
    }
    node.set("$", true);
  }
  let prefix = "";
  let node = root;
  while (node.size === 1 && !node.has("$")) {
    const [ch, next] = [...node.entries()][0];
    prefix += ch;
    node = next;
  }
  return prefix;
}

function lcgLists(count, seed = 1592) {
  const lists = [];
  for (let k = 0; k < count; k++) {
    const size = 1 + (seed % 3);
    const words = [];
    for (let w = 0; w < size; w++) {
      const length = 1 + (seed % 4);
      let word = "";
      for (let i = 0; i < length; i++) {
        seed = (seed * 75 + 74) % 65537;
        word += "abc"[seed % 3];
      }
      words.push(word);
    }
    seed = (seed * 75 + 74) % 65537;
    lists.push(words);
  }
  return lists;
}

module.exports = { longestCommonPrefixSorted, longestCommonPrefixVertical, longestCommonPrefixTrie, lcgLists };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(longestCommonPrefixSorted(["flower", "flow", "flight"]), "fl");
  assert.strictEqual(longestCommonPrefixSorted(["dog", "racecar", "car"]), "");
  assert.strictEqual(longestCommonPrefixSorted(["abc", "abc"]), "abc");
  assert.strictEqual(longestCommonPrefixSorted(["a"]), "a");
  assert.strictEqual(longestCommonPrefixSorted([]), "");
  assert.strictEqual(longestCommonPrefixSorted(["", "b"]), "");
  assert.strictEqual(longestCommonPrefixSorted(["abab", "aba", "ababbb"]), "aba");

  for (const strs of [["flower", "flow", "flight"], ["dog", "racecar", "car"], ["abc", "abc"], ["", "b"]]) {
    const expected = longestCommonPrefixSorted(strs);
    assert.strictEqual(longestCommonPrefixVertical(strs), expected, strs.join(","));
    assert.strictEqual(longestCommonPrefixTrie(strs), expected, strs.join(","));
  }
  for (const strs of lcgLists(80)) {
    const expected = longestCommonPrefixSorted(strs);
    assert.strictEqual(longestCommonPrefixVertical(strs), expected);
    assert.strictEqual(longestCommonPrefixTrie(strs), expected);
  }
  console.log("PASS");
}
