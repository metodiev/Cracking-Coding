/** Task 406 - Implement an online suffix tree (Ukkonen outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Outline: a real Ukkonen tree builds the *compressed* suffix tree online in O(n) by extending
 * the last suffix, following a suffix link when a split happens, and keeping the suffix link
 * of every new node correct. This outline keeps the same skeleton for tiny strings: an
 * explicit suffix trie of the text (every node is a distinct substring, edges are single
 * characters) plus the suffix link of each node, which points at the node of its path label
 * minus the first character. That link is exactly the invariant Ukkonen restores when it
 * splits an edge, and it is what makes "ends with suffix link" navigation possible here:
 * following links from a node walks through all suffixes of its path in O(depth) steps.
 *
 * Links are wired breadth-first: a depth-1 node links to the root, and a node whose parent p
 * is reached over character c links to children[link[p]][c], which always exists in a suffix
 * trie.
 *
 * Checks on tiny texts only (banana, mississippi, abab, aaaa): every substring is findable
 * and every generated non-substring fails, node count matches the brute-force distinct
 * substring set, each link target really is the node of path[1:], link chains enumerate all
 * suffixes, and subtree terminal counts reproduce naive occurrence counts.
 */
"use strict";

class SuffixTreeOutline {
  constructor(text) {
    this.text = text;
    this.children = [new Map()];
    this.parent = [-1];
    this.parentChar = ["\0"];
    this.depth = [0];
    this.terminal = [false];
    this.link = [-1];
    for (let start = 0; start < text.length; start++) {
      let node = 0;
      for (let pos = start; pos < text.length; pos++) {
        const ch = text[pos];
        if (!this.children[node].has(ch)) {
          const child = this.depth.length;
          this.children[node].set(ch, child);
          this.children.push(new Map());
          this.parent.push(node);
          this.parentChar.push(ch);
          this.depth.push(this.depth[node] + 1);
          this.terminal.push(false);
          this.link.push(-1);
        }
        node = this.children[node].get(ch);
      }
      this.terminal[node] = true;
    }
    this.wireSuffixLinks();
  }

  wireSuffixLinks() {
    const order = [0];
    for (let i = 0; i < order.length; i++) {
      for (const child of this.children[order[i]].values()) order.push(child);
    }
    for (const node of order) {
      if (node === 0) continue;
      if (this.depth[node] === 1) {
        this.link[node] = 0;
      } else {
        const p = this.parent[node];
        this.link[node] = this.children[this.link[p]].get(this.parentChar[node]);
      }
    }
  }

  pathLabel(node) {
    const chars = [];
    while (node !== 0) {
      chars.push(this.parentChar[node]);
      node = this.parent[node];
    }
    return chars.reverse().join("");
  }

  find(pattern) {
    let node = 0;
    for (const ch of pattern) {
      if (!this.children[node].has(ch)) return -1;
      node = this.children[node].get(ch);
    }
    return node;
  }

  contains(pattern) {
    return this.find(pattern) !== -1;
  }

  suffixChain(node) {
    const chain = [];
    while (node !== -1) {
      chain.push(node);
      node = this.link[node];
    }
    return chain;
  }

  occurrences(pattern) {
    const node = this.find(pattern);
    if (node === -1) return 0;
    let count = 0;
    const stack = [node];
    while (stack.length) {
      const current = stack.pop();
      if (this.terminal[current]) count++;
      for (const child of this.children[current].values()) stack.push(child);
    }
    return count;
  }

  distinctSubstrings() {
    return this.depth.length - 1;
  }
}

function allSubstrings(s) {
  const seen = new Set();
  for (let i = 0; i < s.length; i++) {
    for (let j = i + 1; j <= s.length; j++) seen.add(s.slice(i, j));
  }
  return seen;
}

function countOccurrences(s, pattern) {
  if (!pattern) return 0;
  let count = 0;
  for (let i = 0; i + pattern.length <= s.length; i++) {
    if (s.startsWith(pattern, i)) count++;
  }
  return count;
}

function enumerateStrings(alphabet, maxLen) {
  const out = [];
  let frontier = [""];
  for (let depth = 0; depth <= maxLen; depth++) {
    out.push(...frontier);
    const nextFrontier = [];
    for (const word of frontier) {
      for (const ch of alphabet) nextFrontier.push(word + ch);
    }
    frontier = nextFrontier;
  }
  return out;
}

function lcgString(length, alphabet, seed) {
  const chars = [];
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    chars.push(alphabet[seed % alphabet.length]);
  }
  return [chars.join(""), seed];
}

function checkOutline(text, alphabet, probeLen) {
  const tree = new SuffixTreeOutline(text);
  const substrings = allSubstrings(text);
  const assert = require("node:assert");

  assert.strictEqual(tree.distinctSubstrings(), substrings.size, text);
  assert.ok(tree.contains("") && tree.find("") === 0, text);
  for (const t of substrings) {
    const node = tree.find(t);
    assert.notStrictEqual(node, -1, t);
    assert.strictEqual(tree.pathLabel(node), t, t);
    assert.strictEqual(tree.occurrences(t), countOccurrences(text, t), t);
  }
  for (const t of enumerateStrings(alphabet, probeLen)) {
    if (t && !substrings.has(t)) assert.ok(!tree.contains(t), t);
  }
  for (let node = 1; node < tree.depth.length; node++) {
    const label = tree.pathLabel(node);
    const link = tree.link[node];
    assert.notStrictEqual(link, -1, label);
    assert.strictEqual(tree.pathLabel(link), label.slice(1), label);
    const chain = tree.suffixChain(node);
    assert.strictEqual(chain.length, label.length + 1, label);
    const labels = chain.map((n) => tree.pathLabel(n));
    const expected = Array.from({ length: label.length }, (_, k) => label.slice(k));
    expected.push("");
    assert.deepStrictEqual(labels, expected, label);
    assert.strictEqual(tree.depth[link], tree.depth[node] - 1, label);
  }
}

module.exports = { SuffixTreeOutline, allSubstrings, countOccurrences, enumerateStrings, lcgString, checkOutline };

if (require.main === module) {
  const assert = require("node:assert");
  checkOutline("banana", "abn", 4);
  const banana = new SuffixTreeOutline("banana");
  assert.strictEqual(banana.occurrences("ana"), 2);
  assert.strictEqual(banana.occurrences("a"), 3);
  assert.strictEqual(banana.occurrences("na"), 2);
  assert.strictEqual(banana.occurrences("banana"), 1);
  assert.ok(!banana.contains("bananas") && !banana.contains("nana-"));

  checkOutline("mississippi", "imps", 3);
  const mississippi = new SuffixTreeOutline("mississippi");
  assert.strictEqual(mississippi.occurrences("issi"), 2);
  assert.strictEqual(mississippi.occurrences("ssi"), 2);
  assert.strictEqual(mississippi.occurrences("i"), 4);

  checkOutline("abab", "ab", 6);
  checkOutline("aaaa", "a", 6);
  checkOutline("abcab", "abc", 5);
  checkOutline("", "ab", 2);
  checkOutline("z", "ab", 2);

  let seed = 2718;
  for (let trial = 0; trial < 20; trial++) {
    let text;
    [text, seed] = lcgString(8, "ab", seed);
    checkOutline(text, "ab", 3);
  }
  for (let trial = 0; trial < 10; trial++) {
    let text;
    [text, seed] = lcgString(7, "abc", seed);
    checkOutline(text, "abc", 3);
  }

  const base = new SuffixTreeOutline("aban");
  assert.strictEqual(base.occurrences("a"), 2);
  const extended = new SuffixTreeOutline("abana");
  assert.strictEqual(extended.distinctSubstrings(), allSubstrings("abana").size);
  assert.strictEqual(extended.occurrences("a"), 3);
  assert.strictEqual(extended.pathLabel(extended.link[extended.find("bana")]), "ana");
  console.log("PASS");
}
