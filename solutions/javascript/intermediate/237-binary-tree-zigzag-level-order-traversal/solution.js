/** Task 237 - Binary tree zigzag level order traversal. (Tier: Intermediate, CTCI Ch 4) */
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

/**
 * Level-order BFS with a per-level snapshot; odd levels (0-based) are reversed so
 * the output alternates left->right / right->left. An empty tree yields []. O(n).
 */
function zigzagLevelOrder(root) {
  if (root === null) return [];
  const levels = [];
  let queue = [root];
  let leftToRight = true;
  while (queue.length) {
    const level = [];
    const next = [];
    for (const node of queue) {
      level.push(node.val);
      if (node.left) next.push(node.left);
      if (node.right) next.push(node.right);
    }
    levels.push(leftToRight ? level : level.reverse());
    leftToRight = !leftToRight;
    queue = next;
  }
  return levels;
}

module.exports = { TreeNode, buildTree, zigzagLevelOrder };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(zigzagLevelOrder(buildTree([3, 9, 20, null, null, 15, 7])),
    [[3], [20, 9], [15, 7]]);
  assert.deepStrictEqual(zigzagLevelOrder(buildTree([1, 2, 3, 4, 5, 6, 7])),
    [[1], [3, 2], [4, 5, 6, 7]]);
  assert.deepStrictEqual(zigzagLevelOrder(buildTree([1, 2, 3, 4, null, null, 5])),
    [[1], [3, 2], [4, 5]]);
  assert.deepStrictEqual(zigzagLevelOrder(buildTree([1])), [[1]]);
  assert.deepStrictEqual(zigzagLevelOrder(buildTree([1, 2])), [[1], [2]]);
  assert.deepStrictEqual(zigzagLevelOrder(null), []);
  console.log("PASS");
}
