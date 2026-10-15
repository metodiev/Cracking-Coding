/** Task 253 - Implement compressed trie (radix tree) basics. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class RadixNode {
  constructor(label) {
    this.label = label;
    this.children = new Map();          // keyed by the first character of each child label
    this.isWord = false;
  }
}

/**
 * Radix tree: runs of single-child characters become one edge label; insert() splits an
 * edge when the new word shares only part of it. "cart" then "car" splits into car* + t*,
 * then "care" adds e* under car. shape() renders that as "[car*[e*,t*]]".
 */
function insert(root, word) {
  let node = root;
  let rest = word;
  for (;;) {
    if (rest === "") {
      const wasNew = !node.isWord;
      node.isWord = true;
      return wasNew;
    }
    const first = rest[0];
    const child = node.children.get(first);
    if (!child) {
      const fresh = new RadixNode(rest);
      fresh.isWord = true;
      node.children.set(first, fresh);
      return true;
    }
    let common = 0;
    while (common < rest.length && common < child.label.length
           && rest[common] === child.label[common]) common += 1;
    if (common === child.label.length) {          // consumed the whole edge
      node = child;
      rest = rest.slice(common);
      continue;
    }
    const split = new RadixNode(child.label.slice(0, common));   // split the edge
    child.label = child.label.slice(common);
    split.children.set(child.label[0], child);
    node.children.set(first, split);
    rest = rest.slice(common);
    if (rest === "") {
      split.isWord = true;
    } else {
      const fresh = new RadixNode(rest);
      fresh.isWord = true;
      split.children.set(rest[0], fresh);
    }
    return true;
  }
}

function contains(root, word) {
  let node = root;
  let i = 0;
  while (i < word.length) {
    const child = node.children.get(word[i]);
    if (!child || word.slice(i, i + child.label.length) !== child.label) return false;
    i += child.label.length;
    node = child;
  }
  return node.isWord;
}

function wordsWithPrefix(root, prefix) {
  let node = root;
  let soFar = "";
  let i = 0;
  while (i < prefix.length) {
    const child = node.children.get(prefix[i]);
    if (!child) return [];
    const want = prefix.length - i;
    const take = Math.min(want, child.label.length);
    if (child.label.slice(0, take) !== prefix.slice(i, i + take)) return [];
    node = child;
    soFar += child.label;
    if (want <= child.label.length) break;        // prefix ends inside this edge
    i += child.label.length;
  }
  const out = [];
  collect(node, soFar, out);
  return out;
}

function collect(node, soFar, out) {
  if (node.isWord) out.push(soFar);
  for (const ch of [...node.children.keys()].sort()) {
    const child = node.children.get(ch);
    collect(child, soFar + child.label, out);
  }
}

function startsWith(root, prefix) {
  return wordsWithPrefix(root, prefix).length > 0;
}

/** Deterministic compressed rendering: labels glued by commas, '*' = stored word. */
function shape(node) {
  const text = node.label + (node.isWord ? "*" : "");
  if (node.children.size === 0) return text;
  const parts = [...node.children.keys()].sort().map((ch) => shape(node.children.get(ch)));
  return text + "[" + parts.join(",") + "]";
}

/** Number of nodes excluding the root, i.e. the number of edges. */
function nodeCount(root) {
  let total = root.children.size;
  for (const child of root.children.values()) total += nodeCount(child);
  return total;
}

function plainTrieEdges(words) {
  const prefixes = new Set();
  for (const word of words) {
    for (let i = 1; i <= word.length; i += 1) prefixes.add(word.slice(0, i));
  }
  return prefixes.size;
}

module.exports = { RadixNode, insert, contains, wordsWithPrefix, startsWith, shape, nodeCount, plainTrieEdges };

if (require.main === module) {
  const assert = require("node:assert");
  const root = new RadixNode("");

  assert.strictEqual(insert(root, "cart"), true);
  assert.strictEqual(shape(root), "[cart*]");
  assert.strictEqual(insert(root, "car"), true);             // split: car* + t*
  assert.strictEqual(shape(root), "[car*[t*]]");
  assert.strictEqual(insert(root, "care"), true);            // care joins under car
  assert.strictEqual(shape(root), "[car*[e*,t*]]");
  assert.strictEqual(insert(root, "car"), false);            // duplicate word

  for (const word of ["car", "care", "cart"]) assert.strictEqual(contains(root, word), true);
  for (const word of ["ca", "c", "carts", "caret", "card", ""]) {
    assert.strictEqual(contains(root, word), false);
  }

  assert.strictEqual(nodeCount(root), 3);
  assert.strictEqual(plainTrieEdges(["cart", "car", "care"]), 5);
  assert.ok(nodeCount(root) < plainTrieEdges(["cart", "car", "care"]));

  assert.strictEqual(insert(root, "do"), true);
  assert.strictEqual(insert(root, "dot"), true);
  assert.strictEqual(insert(root, "dog"), true);
  assert.strictEqual(shape(root), "[car*[e*,t*],do*[g*,t*]]");

  assert.deepStrictEqual(wordsWithPrefix(root, "car"), ["car", "care", "cart"]);
  assert.deepStrictEqual(wordsWithPrefix(root, "ca"), ["car", "care", "cart"]);
  assert.deepStrictEqual(wordsWithPrefix(root, "care"), ["care"]);
  assert.deepStrictEqual(wordsWithPrefix(root, "do"), ["do", "dog", "dot"]);
  assert.deepStrictEqual(wordsWithPrefix(root, "d"), ["do", "dog", "dot"]);
  assert.deepStrictEqual(wordsWithPrefix(root, ""), ["car", "care", "cart", "do", "dog", "dot"]);
  assert.deepStrictEqual(wordsWithPrefix(root, "z"), []);
  assert.deepStrictEqual(wordsWithPrefix(root, "carts"), []);
  assert.ok(startsWith(root, "car"));
  assert.ok(!startsWith(root, "cat"));

  assert.strictEqual(nodeCount(root), 6);
  assert.strictEqual(plainTrieEdges(["car", "care", "cart", "do", "dog", "dot"]), 9);
  for (const word of ["car", "care", "cart", "do", "dog", "dot"]) {
    assert.strictEqual(contains(root, word), true);
  }
  for (const word of ["co", "dat", "dodge", "done"]) assert.strictEqual(contains(root, word), false);
  console.log("PASS");
}
