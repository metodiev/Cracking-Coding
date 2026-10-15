/** Task 144 - Convert sorted array to balanced BST. (Tier: Beginner, CTCI Ch 4)
 *
 * The middle element becomes the root, so both halves stay equally deep and the
 * resulting tree has the minimal possible height for those values.
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


/** Height-balanced BST from ascending nums; null for an empty array. */
function sortedArrayToBst(nums) {
  function build(lo, hi) {
    if (lo > hi) return null;
    const mid = Math.floor((lo + hi) / 2);   // left-biased middle keeps it stable
    const node = new TreeNode(nums[mid]);
    node.left = build(lo, mid - 1);
    node.right = build(mid + 1, hi);
    return node;
  }
  return build(0, nums.length - 1);
}

/** Left - node - right values; sorted output proves the BST ordering. */
function inorder(root) {
  if (!root) return [];
  return inorder(root.left).concat([root.val], inorder(root.right));
}

function height(node) {
  if (!node) return 0;
  return 1 + Math.max(height(node.left), height(node.right));
}

/** Check the requirement directly: every node's subtree heights differ by <= 1. */
function isHeightBalanced(root) {
  if (!root) return true;
  return (
    Math.abs(height(root.left) - height(root.right)) <= 1 &&
    isHeightBalanced(root.left) &&
    isHeightBalanced(root.right)
  );
}

module.exports = { TreeNode, buildTree, sortedArrayToBst, inorder, height, isHeightBalanced };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(sortedArrayToBst([]), null);
  assert.strictEqual(sortedArrayToBst([5]).val, 5);
  const small = sortedArrayToBst([1, 2, 3]);
  assert.deepStrictEqual([small.val, small.left.val, small.right.val], [2, 1, 3]);
  const cases = [[], [1], [1, 2, 3], [1, 2, 3, 4, 5, 6], [-10, -3, 0, 5, 9],
                 [1, 2, 3, 4, 5, 6, 7], [1, 1, 1], [0, 1000000]];
  for (const nums of cases) {
    const tree = sortedArrayToBst(nums);
    assert.deepStrictEqual(inorder(tree), nums, String(nums));    // BST property
    assert.strictEqual(isHeightBalanced(tree), true, String(nums));
    if (nums.length) {
      assert.strictEqual(tree.val, nums[Math.floor((nums.length - 1) / 2)], "middle is the root");
    }
  }
  assert.strictEqual(height(sortedArrayToBst([1, 2, 3, 4, 5, 6, 7])), 3);
  assert.strictEqual(height(sortedArrayToBst([-10, -3, 0, 5, 9])), 3);
  assert.strictEqual(height(sortedArrayToBst([1, 2, 3, 4, 5, 6])), 3);
  console.log("PASS");
}
