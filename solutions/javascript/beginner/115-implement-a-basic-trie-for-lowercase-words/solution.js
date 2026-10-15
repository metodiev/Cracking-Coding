/** Task 115 - Implement a basic trie for lowercase words. (Tier: Beginner, CTCI Ch 4) */
"use strict";

class TrieNode {
  constructor() {
    this.children = new Map();
    this.isWord = false;
  }
}

class Trie {
  constructor() {
    this.root = new TrieNode();
    this.wordCount = 0;
  }

  insert(word) {
    let node = this.root;
    for (const ch of word) {
      if (!node.children.has(ch)) node.children.set(ch, new TrieNode());
      node = node.children.get(ch);
    }
    if (!node.isWord) {
      node.isWord = true;
      this.wordCount++;
    }
  }

  contains(word) {
    const node = this.walk(word);
    return node !== null && node.isWord;
  }

  startsWith(prefix) {
    return this.walk(prefix) !== null;
  }

  walk(s) {
    let node = this.root;
    for (const ch of s) {
      node = node.children.get(ch);
      if (!node) return null;
    }
    return node;
  }
}

module.exports = { TrieNode, Trie };

if (require.main === module) {
  const assert = require("node:assert");
  const t = new Trie();
  for (const w of ["cat", "car", "card", "dog"]) t.insert(w);
  assert.strictEqual(t.wordCount, 4);
  t.insert("cat");
  assert.strictEqual(t.wordCount, 4);
  assert.strictEqual(t.contains("cat"), true);
  assert.strictEqual(t.contains("ca"), false);
  assert.strictEqual(t.startsWith("ca"), true);
  assert.strictEqual(t.startsWith("z"), false);
  console.log("PASS");
}
