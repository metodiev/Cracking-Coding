/** Task 151 - Path sum existence in tree. (Tier: Beginner, CTCI Ch 4) */
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

/** True iff a root-to-leaf path sums to target (leaf = both children null). */
function hasPathSum(root, target) {
  if (!root) return false;
  if (!root.left && !root.right) return root.val === target;
  return hasPathSum(root.left, target - root.val) ||
    hasPathSum(root.right, target - root.val);
}

module.exports = { TreeNode, buildTree, hasPathSum };

if (require.main === module) {
  const assert = require("node:assert");
  const root = buildTree([5, 4, 8, 11, null, 13, 4, 7, 2, null, null, null, 1]);
  assert.strictEqual(hasPathSum(root, 22), true);       // 5 + 4 + 11 + 2
  assert.strictEqual(hasPathSum(root, 18), true);       // 5 + 8 + 4 + 1
  assert.strictEqual(hasPathSum(root, 5), false);       // 5 is not a leaf
  assert.strictEqual(hasPathSum(null, 0), false);       // empty tree has no paths
  assert.strictEqual(hasPathSum(buildTree([1]), 1), true);
  assert.strictEqual(hasPathSum(buildTree([1]), 2), false);
  assert.strictEqual(hasPathSum(buildTree([1, 2]), 1), false);  // one missing child
  assert.strictEqual(hasPathSum(buildTree([1, 2]), 3), true);   // 1 -> 2
  assert.strictEqual(hasPathSum(buildTree([-2, null, -3]), -5), true);
  assert.strictEqual(hasPathSum(buildTree([0, 1, 1]), 1), true);
  assert.strictEqual(hasPathSum(buildTree([0, 1, 1]), 0), false);
  console.log("PASS");
}
