/** Task 141 - Min depth of binary tree. (Tier: Beginner, CTCI Ch 4)
 *
 * Minimum depth = nodes on the shortest root-to-leaf path; a node with only
 * one child is not a leaf, so [1, 2, null, 3] has min depth 3 (not 2).
 */
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


/** BFS level count: the first level holding a leaf is the shallowest one. */
function minDepth(root) {
  if (!root) return 0;
  let depth = 0;
  let queue = [root];
  while (queue.length) {
    depth++;
    const next = [];
    for (const node of queue) {
      if (!node.left && !node.right) return depth;
      if (node.left) next.push(node.left);
      if (node.right) next.push(node.right);
    }
    queue = next;
  }
  return depth;
}

/** Follow the only child instead of treating a missing child as an empty leaf. */
function minDepthRecursive(node) {
  if (!node) return 0;
  if (!node.left && !node.right) return 1;
  if (!node.left) return 1 + minDepthRecursive(node.right);
  if (!node.right) return 1 + minDepthRecursive(node.left);
  return 1 + Math.min(minDepthRecursive(node.left), minDepthRecursive(node.right));
}

module.exports = { TreeNode, buildTree, minDepth, minDepthRecursive };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(minDepth(null), 0);
  assert.strictEqual(minDepthRecursive(null), 0);
  assert.strictEqual(minDepth(buildTree([7])), 1);
  assert.strictEqual(minDepthRecursive(buildTree([7])), 1);
  // trap: node 2 has no left child, so 2 is not a leaf -> the 3 sits at depth 3
  assert.strictEqual(minDepth(buildTree([1, 2, null, 3])), 3);
  assert.strictEqual(minDepthRecursive(buildTree([1, 2, null, 3])), 3);
  assert.strictEqual(minDepth(buildTree([3, 9, 20, null, null, 15, 7])), 2);
  assert.strictEqual(minDepth(buildTree([1, 2, null, 3, null, 4])), 4);
  for (const data of [[1, 2, 3, 4, 5, null, 6], [1, 2, null, 3], [1, 2, 3], [1, null, 2]]) {
    assert.strictEqual(minDepth(buildTree(data)), minDepthRecursive(buildTree(data)), String(data));
  }
  console.log("PASS");
}
