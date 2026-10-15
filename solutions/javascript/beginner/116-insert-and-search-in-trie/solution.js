/** Task 116 - Insert and search in trie. (Tier: Beginner, CTCI Ch 4) */
"use strict";

class ArrayNode {
  constructor() {
    this.kids = new Array(26).fill(null);
    this.end = false;
  }
}

class WordTrie {
  constructor() {
    this.root = new ArrayNode();
  }

  insert(word) {
    let node = this.root;
    for (const ch of word) {
      const i = ch.charCodeAt(0) - 97;
      if (!node.kids[i]) node.kids[i] = new ArrayNode();
      node = node.kids[i];
    }
    node.end = true;
  }

  search(word) {
    const node = this.nodeFor(word);
    return node !== null && node.end;
  }

  countWithPrefix(prefix) {
    const node = this.nodeFor(prefix);
    return node === null ? 0 : this.count(node);
  }

  wordsWithPrefix(prefix) {
    const node = this.nodeFor(prefix);
    const out = [];
    if (node !== null) this.collect(node, prefix.split(""), out);
    return out;
  }

  nodeFor(s) {
    let node = this.root;
    for (const ch of s) {
      node = node.kids[ch.charCodeAt(0) - 97];
      if (!node) return null;
    }
    return node;
  }

  count(node) {
    let total = node.end ? 1 : 0;
    for (const kid of node.kids) if (kid) total += this.count(kid);
    return total;
  }

  collect(node, path, out) {
    if (node.end) out.push(path.join(""));
    for (let i = 0; i < 26; i++) {
      if (node.kids[i]) {
        path.push(String.fromCharCode(97 + i));
        this.collect(node.kids[i], path, out);
        path.pop();
      }
    }
  }
}

module.exports = { ArrayNode, WordTrie };

if (require.main === module) {
  const assert = require("node:assert");
  const t = new WordTrie();
  for (const w of ["app", "apple", "apply", "apt", "banana"]) t.insert(w);
  assert.strictEqual(t.search("app"), true);
  assert.strictEqual(t.search("apples"), false);
  assert.strictEqual(t.countWithPrefix("app"), 3);
  assert.strictEqual(t.countWithPrefix("z"), 0);
  assert.deepStrictEqual(t.wordsWithPrefix("app"), ["app", "apple", "apply"]);
  console.log("PASS");
}
