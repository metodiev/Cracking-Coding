/** Task 252 - Design a phonebook using trie with counts and delete. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class TrieNode {
  constructor() {
    this.children = new Map();
    this.count = 0;                     // duplicates of the name ending here (0 = prefix only)
  }
}

class Phonebook {
  constructor() {
    this.root = new TrieNode();
  }

  walk(name) {
    let node = this.root;
    for (const ch of name) {
      node = node.children.get(ch);
      if (!node) return null;
    }
    return node;
  }

  insert(name) {
    let node = this.root;
    for (const ch of name) {
      let child = node.children.get(ch);
      if (!child) {
        child = new TrieNode();
        node.children.set(ch, child);
      }
      node = child;
    }
    node.count += 1;
    return node.count;
  }

  count(name) {
    const node = this.walk(name);
    return node ? node.count : 0;
  }

  /** Decrements the count and prunes dead branches; true iff the name existed. */
  delete(name) {
    if (!name) return false;
    const rec = (node, i) => {
      if (i === name.length) {
        if (node.count === 0) return false;
        node.count -= 1;
        return true;
      }
      const ch = name[i];
      const child = node.children.get(ch);
      if (!child) return false;
      if (!rec(child, i + 1)) return false;
      if (child.count === 0 && child.children.size === 0) node.children.delete(ch);
      return true;
    };
    return rec(this.root, 0);
  }

  /** All stored (name, count) pairs starting with prefix, lexicographically sorted. */
  listPrefix(prefix) {
    const node = this.walk(prefix);
    const out = [];
    if (!node) return out;
    const dfs = (current, path) => {
      if (current.count > 0) out.push([path, current.count]);
      for (const ch of [...current.children.keys()].sort()) {
        dfs(current.children.get(ch), path + ch);
      }
    };
    dfs(node, prefix);
    return out;
  }

  words() {
    return this.listPrefix("");
  }

  nodeCount() {
    let total = 0;
    const stack = [this.root];
    while (stack.length) {
      const node = stack.pop();
      total += 1;
      for (const child of node.children.values()) stack.push(child);
    }
    return total;
  }
}

module.exports = { TrieNode, Phonebook };

if (require.main === module) {
  const assert = require("node:assert");
  const all = [["alan", 1], ["albert", 1], ["alice", 2], ["b", 1], ["barb", 1], ["bob", 1]];

  const empty = new Phonebook();
  assert.strictEqual(empty.nodeCount(), 1);

  const book = new Phonebook();
  assert.strictEqual(book.insert("alice"), 1);
  assert.strictEqual(book.insert("alice"), 2);          // duplicate inserts increment
  for (const name of ["alan", "albert", "bob", "barb", "b"]) {
    assert.strictEqual(book.insert(name), 1);
  }
  // 1 root + 17 shared-prefix nodes: a,al,ala,alan,alb,albe,alber,albert,ali,alic,alice,
  // b,ba,bar,barb,bo,bob
  assert.strictEqual(book.nodeCount(), 18);

  assert.strictEqual(book.count("alice"), 2);
  assert.strictEqual(book.count("alan"), 1);
  assert.strictEqual(book.count("alicia"), 0);
  assert.strictEqual(book.count("al"), 0);              // a prefix is not a name
  assert.strictEqual(book.count("alic"), 0);
  assert.strictEqual(book.count(""), 0);

  assert.deepStrictEqual(book.listPrefix("al"), [["alan", 1], ["albert", 1], ["alice", 2]]);
  assert.deepStrictEqual(book.listPrefix("ali"), [["alice", 2]]);
  assert.deepStrictEqual(book.listPrefix("b"), [["b", 1], ["barb", 1], ["bob", 1]]);
  assert.deepStrictEqual(book.listPrefix("z"), []);
  assert.deepStrictEqual(book.listPrefix(""), all);
  assert.deepStrictEqual(book.words(), all);

  assert.strictEqual(book.delete("alice"), true);       // one reference left
  assert.strictEqual(book.count("alice"), 1);
  assert.strictEqual(book.nodeCount(), 18);
  assert.deepStrictEqual(book.listPrefix("al"), [["alan", 1], ["albert", 1], ["alice", 1]]);
  assert.strictEqual(book.delete("alice"), true);       // gone: alice/alic/ali pruned
  assert.strictEqual(book.count("alice"), 0);
  assert.strictEqual(book.nodeCount(), 15);
  assert.deepStrictEqual(book.listPrefix("al"), [["alan", 1], ["albert", 1]]);
  assert.deepStrictEqual(book.listPrefix("ali"), []);
  assert.strictEqual(book.delete("alice"), false);
  assert.strictEqual(book.delete("zzz"), false);
  assert.strictEqual(book.delete(""), false);

  assert.strictEqual(book.delete("b"), true);           // count 0 but bob/barb survive
  assert.strictEqual(book.count("b"), 0);
  assert.strictEqual(book.nodeCount(), 15);
  assert.deepStrictEqual(book.listPrefix("b"), [["barb", 1], ["bob", 1]]);

  assert.strictEqual(book.delete("barb"), true);        // barb/bar/ba pruned
  assert.strictEqual(book.nodeCount(), 12);
  assert.deepStrictEqual(book.words(), [["alan", 1], ["albert", 1], ["bob", 1]]);

  assert.strictEqual(book.delete("alan"), true);        // alan/ala pruned
  assert.strictEqual(book.nodeCount(), 10);
  assert.strictEqual(book.delete("albert"), true);      // albert..a pruned entirely
  assert.strictEqual(book.nodeCount(), 4);              // root + b + bo + bob
  assert.deepStrictEqual(book.words(), [["bob", 1]]);

  assert.strictEqual(book.delete("bob"), true);
  assert.strictEqual(book.nodeCount(), 1);              // back to the empty state
  assert.deepStrictEqual(book.words(), []);
  assert.deepStrictEqual(book.listPrefix(""), []);
  assert.strictEqual(book.delete("bob"), false);
  console.log("PASS");
}
