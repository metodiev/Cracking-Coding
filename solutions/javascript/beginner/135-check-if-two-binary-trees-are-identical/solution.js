/** Task 135 - Check if two binary trees are identical. (Tier: Beginner, CTCI Ch 4) */
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

function areIdentical(a, b) {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  return a.val === b.val && areIdentical(a.left, b.left) && areIdentical(a.right, b.right);
}

module.exports = { TreeNode, buildTree, areIdentical };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(areIdentical(buildTree([1, 2, 3, null, 4]), buildTree([1, 2, 3, null, 4])), true);
  assert.strictEqual(areIdentical(buildTree([1, 2, 3, null, 4]), buildTree([1, 2, 3, 4])), false);
  assert.strictEqual(areIdentical(buildTree([1, 2]), buildTree([1, 3])), false);
  assert.strictEqual(areIdentical(null, null), true);
  assert.strictEqual(areIdentical(buildTree([1]), null), false);
  console.log("PASS");
}
