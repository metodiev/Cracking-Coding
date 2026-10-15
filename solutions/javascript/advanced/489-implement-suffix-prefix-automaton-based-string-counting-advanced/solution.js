/** Task 489 - Implement suffix-prefix automaton based string counting (advanced). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Documented clean task: given a finite set of strings S, count the distinct strings x that are both
 * a non-empty prefix of some s in S and a non-empty suffix of some s in S, i.e. |P ∩ Q| where
 * P = {s[:i] : s in S, 1 <= i <= |s|} and Q = {s[j:] : s in S, 0 <= j < |s|}.
 *
 * Automaton approach: build a trie over all prefixes of S (every trie node corresponds to a member
 * of P, and the trie accepts exactly P - it is the "prefix automaton" of S). For each suffix
 * x = s[j:] of every s in S, feed x to the automaton from the root in one walk: x is accepted iff
 * x in P, so the walk decides membership x in P ∩ Q. A set of the accepted x counts each string
 * once. The walk costs O(|x|) per suffix, O(sum |s|^2) overall - documented as fine for the small
 * sets used here.
 *
 * The result is asserted against a brute-force set intersection on small LCG-generated sets
 * (30 sets of 5 strings over {a, b}, lengths 1..12) and on hand cases; the random-set count total is
 * pinned at 330 and the larger {a,b,c} case at 11. Hand case: {ababa, baba} -> 5
 * (a, ba, aba, baba, ababa).
 */
"use strict";

class TrieNode {
  constructor() {
    this.children = new Map();
  }
}

class Trie {
  constructor() {
    this.root = new TrieNode();
    this.size = 0;
  }

  insert(word) {
    let node = this.root;
    for (const ch of word) {
      let next = node.children.get(ch);
      if (next === undefined) {
        next = new TrieNode();
        node.children.set(ch, next);
        this.size += 1;
      }
      node = next;
    }
  }

  accepts(word) {
    let node = this.root;
    for (const ch of word) {
      node = node.children.get(ch);
      if (node === undefined) return false;
    }
    return true;
  }
}

function buildPrefixAutomaton(strings) {
  const trie = new Trie();
  for (const s of strings) {
    for (let i = 1; i <= s.length; i++) trie.insert(s.slice(0, i));
  }
  return trie;
}

function countPrefixSuffixIntersection(strings) {
  const trie = buildPrefixAutomaton(strings);
  const matched = new Set();
  let walked = 0;
  for (const s of strings) {
    for (let j = 0; j < s.length; j++) {
      const x = s.slice(j);
      walked += 1;
      if (trie.accepts(x)) matched.add(x);
    }
  }
  return { count: matched.size, walked, nodes: trie.size };
}

function brutePrefixSuffixIntersection(strings) {
  const prefixes = new Set();
  const suffixes = new Set();
  for (const s of strings) {
    for (let i = 1; i <= s.length; i++) prefixes.add(s.slice(0, i));
    for (let j = 0; j < s.length; j++) suffixes.add(s.slice(j));
  }
  const common = new Set();
  for (const x of prefixes) if (suffixes.has(x)) common.add(x);
  return common;
}

function lcgStrings(count, seed, maxLength, alphabet) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    seed = (seed * 75 + 74) % 65537;
    const length = 1 + (seed % maxLength);
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += alphabet[seed % alphabet.length];
    }
    strings.push(s);
  }
  return strings;
}

function prefixCount(strings) {
  const prefixes = new Set();
  for (const s of strings) {
    for (let i = 1; i <= s.length; i++) prefixes.add(s.slice(0, i));
  }
  return prefixes.size;
}

function totalSuffixes(strings) {
  return strings.reduce((acc, s) => acc + s.length, 0);
}

module.exports = { Trie, buildPrefixAutomaton, countPrefixSuffixIntersection, brutePrefixSuffixIntersection };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(countPrefixSuffixIntersection(["ababa", "baba"]).count, 5);
  assert.deepStrictEqual(
    Array.from(brutePrefixSuffixIntersection(["ababa", "baba"])).sort(),
    ["ababa", "aba", "ba", "a", "baba"].sort()
  );
  assert.strictEqual(countPrefixSuffixIntersection(["abc"]).count, 1);
  assert.deepStrictEqual(Array.from(brutePrefixSuffixIntersection(["abc"])), ["abc"]);
  assert.strictEqual(countPrefixSuffixIntersection(["a"]).count, 1);
  assert.strictEqual(countPrefixSuffixIntersection(["ab", "ba"]).count, 4);
  assert.deepStrictEqual(
    Array.from(brutePrefixSuffixIntersection(["ab", "ba"])).sort(),
    ["a", "b", "ab", "ba"].sort()
  );
  assert.strictEqual(countPrefixSuffixIntersection(["aaaa"]).count, 4);

  for (const strings of [["ababa", "baba"], ["abc", "cba", "bac"]]) {
    const trie = buildPrefixAutomaton(strings);
    assert.strictEqual(trie.size, prefixCount(strings));
    const prefixes = new Set();
    for (const s of strings) {
      for (let i = 1; i <= s.length; i++) prefixes.add(s.slice(0, i));
    }
    for (const x of prefixes) assert.ok(trie.accepts(x), `accepts ${x}`);
    assert.ok(!trie.accepts("zzz"));
  }

  let total = 0;
  for (let k = 0; k < 30; k++) {
    const strings = lcgStrings(5, 489 + k, 12, "ab");
    const result = countPrefixSuffixIntersection(strings);
    const expected = brutePrefixSuffixIntersection(strings);
    assert.strictEqual(result.count, expected.size, `random set ${k}`);
    assert.strictEqual(result.walked, totalSuffixes(strings), `walked ${k}`);
    assert.strictEqual(result.nodes, prefixCount(strings), `nodes ${k}`);
    total += result.count;
  }
  assert.strictEqual(total, 330);

  const strings = lcgStrings(6, 777, 14, "abc");
  const result = countPrefixSuffixIntersection(strings);
  const expected = brutePrefixSuffixIntersection(strings);
  assert.strictEqual(result.count, expected.size);
  assert.strictEqual(result.count, 11);
  assert.strictEqual(result.walked, totalSuffixes(strings));
  assert.strictEqual(result.nodes, prefixCount(strings));

  assert.strictEqual(
    countPrefixSuffixIntersection(["abc", "abc", "abc"]).count,
    countPrefixSuffixIntersection(["abc"]).count
  );
  console.log("PASS");
}
