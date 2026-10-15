/** Task 249 - Implement treap (BST + heap) insert/delete. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class TreapNode {
  constructor(key, pri) {
    this.key = key;
    this.pri = pri;
    this.left = null;
    this.right = null;
  }
}

function rotateRight(y) {
  const x = y.left;
  y.left = x.right;
  x.right = y;
  return x;
}

function rotateLeft(x) {
  const y = x.right;
  x.right = y.left;
  y.left = x;
  return y;
}

function insert(node, key, pri) {
  if (!node) return { node: new TreapNode(key, pri), inserted: true };
  if (key === node.key) return { node, inserted: false };
  let inserted;
  if (key < node.key) {
    ({ node: node.left, inserted } = insert(node.left, key, pri));
    if (node.left.pri > node.pri) node = rotateRight(node);
  } else {
    ({ node: node.right, inserted } = insert(node.right, key, pri));
    if (node.right.pri > node.pri) node = rotateLeft(node);
  }
  return { node, inserted };
}

function erase(node, key) {
  if (!node) return { node: null, removed: false };
  let removed;
  if (key < node.key) {
    ({ node: node.left, removed } = erase(node.left, key));
  } else if (key > node.key) {
    ({ node: node.right, removed } = erase(node.right, key));
  } else {
    removed = true;
    if (!node.left) return { node: node.right, removed };
    if (!node.right) return { node: node.left, removed };
    if (node.left.pri > node.right.pri) {
      node = rotateRight(node);
      ({ node: node.right } = erase(node.right, key));
    } else {
      node = rotateLeft(node);
      ({ node: node.left } = erase(node.left, key));
    }
  }
  return { node, removed };
}

function build(entries) {
  let root = null;
  for (const [key, pri] of entries) root = insert(root, key, pri).node;
  return root;
}

function inorder(node, out = []) {
  if (node) { inorder(node.left, out); out.push(node.key); inorder(node.right, out); }
  return out;
}

function preorder(node, out = []) {
  if (node) { out.push(node.key); preorder(node.left, out); preorder(node.right, out); }
  return out;
}

function isHeap(node) {
  if (!node) return true;
  if (node.left && node.left.pri > node.pri) return false;
  if (node.right && node.right.pri > node.pri) return false;
  return isHeap(node.left) && isHeap(node.right);
}

function contains(node, key) {
  while (node) {
    if (key === node.key) return true;
    node = key < node.key ? node.left : node.right;
  }
  return false;
}

module.exports = { TreapNode, insert, erase, build, inorder, preorder, isHeap, contains };

if (require.main === module) {
  const assert = require("node:assert");
  const entries = [[50, 40], [30, 25], [70, 60], [20, 15], [40, 35], [60, 50], [80, 70]];
  let root = build(entries);
  assert.strictEqual(root.key, 80);
  assert.deepStrictEqual(preorder(root), [80, 70, 60, 50, 40, 30, 20]);
  assert.deepStrictEqual(inorder(root), [20, 30, 40, 50, 60, 70, 80]);
  assert.ok(isHeap(root));

  assert.strictEqual(insert(build([[1, 5]]), 1, 9).inserted, false);

  ({ node: root } = erase(root, 30));
  assert.deepStrictEqual(inorder(root), [20, 40, 50, 60, 70, 80]);
  assert.ok(isHeap(root));
  ({ node: root } = erase(root, 80));
  assert.strictEqual(root.key, 70);
  assert.ok(isHeap(root));
  assert.strictEqual(erase(root, 999).removed, false);
  assert.deepStrictEqual(inorder(root), [20, 40, 50, 60, 70]);
  assert.ok([20, 40, 50, 60, 70].every((k) => contains(root, k)));
  assert.ok(!contains(root, 30));
  console.log("PASS");
}
