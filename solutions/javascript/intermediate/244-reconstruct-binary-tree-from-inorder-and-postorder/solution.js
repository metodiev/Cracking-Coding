/** Task 244 - Reconstruct binary tree from inorder and postorder. (Tier: Intermediate, CTCI Ch 4) */
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

function buildFromInPost(inorder, postorder) {
  if (postorder.length === 0 || postorder.length !== inorder.length) {
    throw new Error("inorder and postorder must be non-empty and equal-length");
  }
  const index = new Map(inorder.map((v, i) => [v, i]));
  let pos = postorder.length - 1;

  function build(lo, hi) {
    if (lo > hi) return null;
    const rootVal = postorder[pos--];
    const node = new TreeNode(rootVal);
    const mid = index.get(rootVal);
    node.right = build(mid + 1, hi);
    node.left = build(lo, mid - 1);
    return node;
  }

  return build(0, inorder.length - 1);
}

function postorderTraverse(node, out = []) {
  if (node) {
    postorderTraverse(node.left, out);
    postorderTraverse(node.right, out);
    out.push(node.val);
  }
  return out;
}

module.exports = { TreeNode, buildTree, inorder, buildFromInPost, postorderTraverse };

if (require.main === module) {
  const assert = require("node:assert");
  const ino = [9, 3, 15, 20, 7];
  const post = [9, 15, 7, 20, 3];
  const root = buildFromInPost(ino, post);
  assert.deepStrictEqual(inorder(root), ino);
  assert.deepStrictEqual(postorderTraverse(root), post);
  console.log("PASS");
}
