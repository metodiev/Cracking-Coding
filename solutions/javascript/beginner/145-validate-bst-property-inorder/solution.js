/** Task 145 - Validate BST property (inorder). (Tier: Beginner, CTCI Ch 4)
 *
 * Two equivalent O(n) routes:
 *   1. min/max range recursion - every node must stay inside the open interval
 *      inherited from its ancestors (Number.MIN_SAFE_INTEGER/MAX_SAFE_INTEGER
 *      bounds, so extreme values are still legal);
 *   2. inorder must be strictly increasing (see isValidBstInorder below).
 * Comparing only a node with its own children is NOT enough:
 * [5, 1, 4, null, null, 3, 6] passes that weaker test although 3 < 5.
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


/** Range recursion: each node carries the (lo, hi) window it must fit in. */
function isValidBst(root) {
  return validRange(root, Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER);
}

function validRange(node, lo, hi) {
  if (!node) return true;
  if (!(node.val > lo && node.val < hi)) return false;
  return validRange(node.left, lo, node.val) && validRange(node.right, node.val, hi);
}

/** Alternative route: walk the tree in order with an explicit stack. */
function isValidBstInorder(root) {
  let previous = Number.MIN_SAFE_INTEGER;
  const stack = [];
  let node = root;
  while (stack.length || node) {
    while (node) {
      stack.push(node);
      node = node.left;
    }
    node = stack.pop();
    if (node.val <= previous) return false;
    previous = node.val;
    node = node.right;
  }
  return true;
}

module.exports = { TreeNode, buildTree, isValidBst, isValidBstInorder };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isValidBst(null), true);
  assert.strictEqual(isValidBstInorder(null), true);
  assert.strictEqual(isValidBst(buildTree([2, 1, 3])), true);
  assert.strictEqual(isValidBst(buildTree([7])), true);
  assert.strictEqual(isValidBst(buildTree([3, 2, null, 1])), true);
  // classic pitfall: 4 sits in the right subtree of 5 but is smaller than 5
  assert.strictEqual(isValidBst(buildTree([5, 1, 4, null, null, 3, 6])), false);
  assert.strictEqual(isValidBst(buildTree([10, 5, 15, null, null, 6, 20])), false);
  assert.strictEqual(isValidBst(buildTree([5, 4, 6, null, null, 3, 7])), false);
  assert.strictEqual(isValidBst(buildTree([1, 1])), false);
  // wide bounds: large magnitudes are legal node values
  assert.strictEqual(isValidBst(buildTree([2147483647, -2147483648, null])), true);
  for (const data of [[2, 1, 3], [5, 1, 4, null, null, 3, 6], [10, 5, 15, null, null, 6, 20],
                      [1, 1], [5, 4, 6, null, null, 3, 7], [3, 2, null, 1],
                      [2147483647, -2147483648, null]]) {
    const tree = buildTree(data);
    assert.strictEqual(isValidBst(tree), isValidBstInorder(tree), String(data));
  }
  console.log("PASS");
}
