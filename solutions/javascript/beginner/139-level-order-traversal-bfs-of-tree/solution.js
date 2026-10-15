/** Task 139 - Level order traversal (BFS) of tree. (Tier: Beginner, CTCI Ch 4) */
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

function levelOrder(root) {
  if (!root) return [];
  const levels = [];
  let queue = [root];
  while (queue.length) {
    const level = [];
    const next = [];
    for (const node of queue) {
      level.push(node.val);
      if (node.left) next.push(node.left);
      if (node.right) next.push(node.right);
    }
    levels.push(level);
    queue = next;
  }
  return levels;
}

module.exports = { TreeNode, buildTree, levelOrder };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(levelOrder(buildTree([1, 2, 3, 4, 5, null, 6])), [[1], [2, 3], [4, 5, 6]]);
  assert.deepStrictEqual(levelOrder(null), []);
  assert.deepStrictEqual(levelOrder(buildTree([7])), [[7]]);
  console.log("PASS");
}
