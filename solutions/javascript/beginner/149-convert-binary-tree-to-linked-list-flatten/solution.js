/** Task 149 - Convert binary tree to linked list (flatten). (Tier: Beginner, CTCI Ch 4)
 *
 * LeetCode 114: rewrite the tree IN PLACE into its preorder order, using the right
 * pointers as the "next" links and leaving every left pointer empty.
 * [1, 2, 5, 3, 4, null, 6] becomes the chain 1 -> 2 -> 3 -> 4 -> 5 -> 6.
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


/** Recursive: flatten both subtrees, then splice the left chain in front of the right one. */
function flatten(root) {
  if (!root) return null;
  const left = root.left;
  const right = root.right;
  flatten(left);
  flatten(right);
  root.left = null;
  root.right = left;
  let node = root;
  while (node.right) node = node.right;     // tail of the flattened left part
  node.right = right;
  return root;
}

/** Preorder with an explicit stack: link each node to the previously visited one. */
function flattenIterative(root) {
  if (!root) return null;
  const stack = [root];
  let prev = null;
  while (stack.length) {
    const node = stack.pop();
    if (prev) {
      prev.left = null;
      prev.right = node;
    }
    if (node.right) stack.push(node.right);   // right first: left is visited first
    if (node.left) stack.push(node.left);
    prev = node;
  }
  return root;
}

function preorderValues(root) {
  if (!root) return [];
  return [root.val].concat(preorderValues(root.left), preorderValues(root.right));
}

/** Values following the right pointers (the flattened preorder order). */
function rightChain(root) {
  const values = [];
  while (root) {
    values.push(root.val);
    root = root.right;
  }
  return values;
}

/** After flattening nothing may remain on a left pointer. */
function allLeftNull(node) {
  while (node) {
    if (node.left) return false;
    node = node.right;
  }
  return true;
}

module.exports = { TreeNode, buildTree, flatten, flattenIterative };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(flatten(null), null);
  assert.strictEqual(flattenIterative(null), null);
  assert.deepStrictEqual(rightChain(flatten(buildTree([7]))), [7]);
  assert.deepStrictEqual(rightChain(flatten(buildTree([1, 2, 5, 3, 4, null, 6]))), [1, 2, 3, 4, 5, 6]);
  const cases = [[], [7], [1, 2, 5, 3, 4, null, 6], [1, 2, null, 3], [1, null, 2],
                 [1, 2, null, 3, null, 4], [1, 2, 3, 4, 5, null, 6]];
  for (const data of cases) {
    const expected = preorderValues(buildTree(data));   // preorder is the target order
    for (const flat of [flatten, flattenIterative]) {
      const result = flat(buildTree(data));
      assert.deepStrictEqual(rightChain(result), expected, String(data));
      assert.strictEqual(allLeftNull(result), true, String(data));
    }
  }
  console.log("PASS");
}
