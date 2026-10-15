/** Task 142 - Check if tree is balanced (height difference <= 1). (Tier: Beginner, CTCI Ch 4)
 *
 * Balanced = at EVERY node the two subtree heights differ by at most 1.
 * The bottom-up pass returns a height, or -1 as soon as a node fails: O(n).
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


/** Number of nodes on the longest downward path (empty tree = 0). */
function height(node) {
  if (!node) return 0;
  return 1 + Math.max(height(node.left), height(node.right));
}

/** Subtree height, or -1 when some node inside the subtree is unbalanced. */
function heightOrFailure(node) {
  if (!node) return 0;
  const left = heightOrFailure(node.left);
  if (left === -1) return -1;
  const right = heightOrFailure(node.right);
  if (right === -1) return -1;
  if (Math.abs(left - right) > 1) return -1;
  return 1 + Math.max(left, right);
}

/** O(n) bottom-up: the -1 sentinel short-circuits everything above it. */
function isBalanced(root) {
  return heightOrFailure(root) !== -1;
}

/** O(n^2) contrast: recompute both subtree heights at every node. */
function isBalancedNaive(root) {
  if (!root) return true;
  if (Math.abs(height(root.left) - height(root.right)) > 1) return false;
  return isBalancedNaive(root.left) && isBalancedNaive(root.right);
}

module.exports = { TreeNode, buildTree, height, isBalanced, isBalancedNaive };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isBalanced(null), true);
  assert.strictEqual(isBalanced(buildTree([1])), true);
  assert.strictEqual(isBalanced(buildTree([1, 2])), true);
  assert.strictEqual(isBalanced(buildTree([3, 9, 20, null, null, 15, 7])), true);
  assert.strictEqual(isBalanced(buildTree([1, 2, 3, 4, 5])), true);
  assert.strictEqual(isBalanced(buildTree([1, 2, 2, 3, 3, null, null, 4, 4])), false);
  assert.strictEqual(isBalanced(buildTree([1, 2, null, 3, null, 4])), false);
  assert.strictEqual(isBalanced(buildTree([1, null, 2, null, 3])), false);
  assert.strictEqual(height(buildTree([1, 2, 3, 4, 5, null, 6])), 3);
  for (const data of [[3, 9, 20, null, null, 15, 7], [1, 2, 2, 3, 3, null, null, 4, 4],
                      [1, 2, null, 3, null, 4], [1, 2, 3, 4, 5], [1, 2, 3]]) {
    assert.strictEqual(isBalanced(buildTree(data)), isBalancedNaive(buildTree(data)), String(data));
  }
  console.log("PASS");
}
