/** Task 247 - Implement AVL tree insert (rotations + balance). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class AVLNode {
  constructor(key) {
    this.key = key;
    this.left = null;
    this.right = null;
    this.height = 1;
  }
}

const height = (node) => (node ? node.height : 0);

function update(node) {
  node.height = 1 + Math.max(height(node.left), height(node.right));
}

const balanceFactor = (node) => height(node.left) - height(node.right);

function rotateRight(y) {
  const x = y.left;
  y.left = x.right;
  x.right = y;
  update(y);
  update(x);
  return x;
}

function rotateLeft(x) {
  const y = x.right;
  x.right = y.left;
  y.left = x;
  update(x);
  update(y);
  return y;
}

function insert(node, key) {
  if (!node) return new AVLNode(key);
  if (key < node.key) node.left = insert(node.left, key);
  else if (key > node.key) node.right = insert(node.right, key);
  else return node;

  update(node);
  const bf = balanceFactor(node);
  if (bf > 1 && key < node.left.key) return rotateRight(node);
  if (bf < -1 && key > node.right.key) return rotateLeft(node);
  if (bf > 1 && key > node.left.key) { node.left = rotateLeft(node.left); return rotateRight(node); }
  if (bf < -1 && key < node.right.key) { node.right = rotateRight(node.right); return rotateLeft(node); }
  return node;
}

function build(keys) {
  let root = null;
  for (const key of keys) root = insert(root, key);
  return root;
}

function inorder(node, out = []) {
  if (node) { inorder(node.left, out); out.push(node.key); inorder(node.right, out); }
  return out;
}

function isBalanced(node) {
  if (!node) return true;
  if (Math.abs(balanceFactor(node)) > 1) return false;
  return isBalanced(node.left) && isBalanced(node.right);
}

const count = (node) => (node ? 1 + count(node.left) + count(node.right) : 0);

module.exports = { AVLNode, insert, build, inorder, isBalanced, height, count };

if (require.main === module) {
  const assert = require("node:assert");
  let root = build([10, 20, 30]);
  assert.deepStrictEqual([root.key, root.left.key, root.right.key], [20, 10, 30]);
  root = build([30, 20, 10]);
  assert.deepStrictEqual([root.key, root.left.key, root.right.key], [20, 10, 30]);
  root = build([30, 10, 20]);
  assert.deepStrictEqual([root.key, root.left.key, root.right.key], [20, 10, 30]);
  root = build([10, 30, 20]);
  assert.deepStrictEqual([root.key, root.left.key, root.right.key], [20, 10, 30]);

  const asc = build([1, 2, 3, 4, 5, 6, 7]);
  assert.deepStrictEqual(inorder(asc), [1, 2, 3, 4, 5, 6, 7]);
  assert.ok(isBalanced(asc) && asc.height === 3);

  const ten = build([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.ok(count(ten) === 10 && isBalanced(ten) && ten.height === 4);
  assert.strictEqual(count(build([5, 5, 5])), 1);
  console.log("PASS");
}
