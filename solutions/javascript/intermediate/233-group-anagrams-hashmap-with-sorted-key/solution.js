/** Task 233 - Group anagrams (hashmap with sorted key). (Tier: Intermediate, CTCI Ch 10) */
"use strict";

/**
 * Key each word by its characters sorted ("eat"/"tea"/"ate" -> "aet") and collect
 * the words in a Map: O(n * L log L) for n words of length L.
 * Deterministic order: groups appear in first-seen key order, words keep input order.
 */
function groupAnagrams(words) {
  const groups = new Map();
  for (const word of words) {
    const key = [...word].sort().join("");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(word);
  }
  return [...groups.values()];
}

module.exports = { groupAnagrams };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]),
    [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]);
  assert.deepStrictEqual(groupAnagrams([]), []);
  assert.deepStrictEqual(groupAnagrams([""]), [[""]]);
  assert.deepStrictEqual(groupAnagrams(["", ""]), [["", ""]]);
  assert.deepStrictEqual(groupAnagrams(["a"]), [["a"]]);
  assert.deepStrictEqual(groupAnagrams(["ab", "ba", "abc", "cab", "bca"]),
    [["ab", "ba"], ["abc", "cab", "bca"]]);
  assert.deepStrictEqual(groupAnagrams(["a", "b", "a"]), [["a", "a"], ["b"]]);
  assert.deepStrictEqual(groupAnagrams(["abc", "abc"]), [["abc", "abc"]]);
  assert.deepStrictEqual(groupAnagrams(["listen", "silent", "enlist", "google"]),
    [["listen", "silent", "enlist"], ["google"]]);
  console.log("PASS");
}
