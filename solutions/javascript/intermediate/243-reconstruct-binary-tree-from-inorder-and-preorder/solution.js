/** Task 243 - Reconstruct binary tree from inorder and preorder. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class TreeNode {
  constructor(val, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}

function buildTree(values) {
  if (values.length === 0 || values[0] === null) return null;
  const root = new TreeNode(values[0]);
  const q = [root];
  let i = 1;
  while (q.length && i < values.length) {
    const node = q.shift();
    if (i < values.length && values[i] !== null) { node.left = new TreeNode(values[i]); q.push(node.left); }
    i++;
    if (i < values.length && values[i] !== null) { node.right = new TreeNode(values[i]); q.push(node.right); }
    i++;
  }
  return root;
}

function preorder(node, out = []) {
  if (node) { out.push(node.val); preorder(node.left, out); preorder(node.right, out); }
  return out;
}

function inorder(node, out = []) {
  if (node) { inorder(node.left, out); out.push(node.val); inorder(node.right, out); }
  return out;
}

function buildFromPreIn(preorder, inorder) {
  if (preorder.length === 0 || preorder.length !== inorder.length) {
    throw new Error("preorder and inorder must be non-empty and equal-length");
  }
  const index = new Map(inorder.map((v, i) => [v, i]));
  let pos = 0;

  function build(lo, hi) {
    if (lo > hi) return null;
    const rootVal = preorder[pos++];
    const node = new TreeNode(rootVal);
    const mid = index.get(rootVal);
    node.left = build(lo, mid - 1);
    node.right = build(mid + 1, hi);
    return node;
  }

  return build(0, inorder.length - 1);
}

module.exports = { TreeNode, buildTree, preorder, inorder, buildFromPreIn };

if (require.main === module) {
  const assert = require("node:assert");
  const pre = [3, 9, 20, 15, 7];
  const ino = [9, 3, 15, 20, 7];
  const root = buildFromPreIn(pre, ino);
  assert.deepStrictEqual(preorder(root), pre);
  assert.deepStrictEqual(inorder(root), ino);
  const spine = buildFromPreIn([1, 2, 3], [3, 2, 1]);
  assert.deepStrictEqual(preorder(spine), [1, 2, 3]);
  assert.deepStrictEqual(inorder(spine), [3, 2, 1]);
  console.log("PASS");
}
