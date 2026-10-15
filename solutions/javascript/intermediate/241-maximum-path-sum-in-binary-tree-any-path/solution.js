/** Task 241 - Maximum path sum in binary tree (any path). (Tier: Intermediate, CTCI Ch 4) */
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

function maxPathSum(root) {
  if (!root) throw new Error("empty tree has no path");
  let best = -Infinity;

  function gain(node) {
    if (!node) return 0;
    const left = Math.max(0, gain(node.left));
    const right = Math.max(0, gain(node.right));
    best = Math.max(best, node.val + left + right);
    return node.val + Math.max(left, right);
  }

  gain(root);
  return best;
}

module.exports = { TreeNode, buildTree, maxPathSum };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(maxPathSum(buildTree([1, 2, 3])), 6);
  assert.strictEqual(maxPathSum(buildTree([-10, 9, 20, null, null, 15, 7])), 42);
  assert.strictEqual(maxPathSum(buildTree([-3])), -3);
  assert.strictEqual(maxPathSum(buildTree([2, -1])), 2);
  console.log("PASS");
}
