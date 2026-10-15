/** Task 140 - Max depth of binary tree. (Tier: Beginner, CTCI Ch 4) */
"use strict";

class TreeNode {
  constructor(val, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}

/** Build a binary tree from level-order values (null marks a missing child). */
function buildTree(values) {
  if (values.length === 0 || values[0] === null) return null;
  const root = new TreeNode(values[0]);
  const q = [root];
  let i = 1;
  while (q.length && i < values.length) {
    const node = q.shift();
    if (i < values.length && values[i] !== null) {
      node.left = new TreeNode(values[i]);
      q.push(node.left);
    }
    i++;
    if (i < values.length && values[i] !== null) {
      node.right = new TreeNode(values[i]);
      q.push(node.right);
    }
    i++;
  }
  return root;
}

function maxDepth(node) {
  if (!node) return 0;
  return 1 + Math.max(maxDepth(node.left), maxDepth(node.right));
}

function maxDepthIterative(root) {
  if (!root) return 0;
  let depth = 0;
  let queue = [root];
  while (queue.length) {
    depth++;
    const next = [];
    for (const node of queue) {
      if (node.left) next.push(node.left);
      if (node.right) next.push(node.right);
    }
    queue = next;
  }
  return depth;
}

module.exports = { TreeNode, buildTree, maxDepth, maxDepthIterative };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(maxDepth(buildTree([1, 2, 3, 4, 5, null, 6])), 3);
  assert.strictEqual(maxDepth(null), 0);
  assert.strictEqual(maxDepth(buildTree([7])), 1);
  assert.strictEqual(maxDepth(buildTree([1, 2, null, 3, null, 4])), 4);
  const data = [1, 2, 3, 4, 5, null, 6];
  assert.strictEqual(maxDepth(buildTree(data)), maxDepthIterative(buildTree(data)));
  console.log("PASS");
}
