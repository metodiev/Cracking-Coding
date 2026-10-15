/** Task 143 - Symmetric tree check (mirror). (Tier: Beginner, CTCI Ch 4)
 *
 * A tree is symmetric when its left and right subtrees are mirror images.
 * Traversal order alone is not enough: compare a.left with b.right and
 * a.right with b.left.
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


/** Two subtrees mirror each other when their mirrored children match. */
function isMirror(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return a.val === b.val && isMirror(a.left, b.right) && isMirror(a.right, b.left);
}

/** Recursive: root is symmetric when its two subtrees mirror each other. */
function isSymmetric(root) {
  if (!root) return true;
  return isMirror(root.left, root.right);
}

/** Iterative variant: stack of mirrored (a, b) pairs. */
function isSymmetricIterative(root) {
  if (!root) return true;
  const stack = [[root.left, root.right]];
  while (stack.length) {
    const [a, b] = stack.pop();
    if (!a && !b) continue;
    if (!a || !b || a.val !== b.val) return false;
    stack.push([a.left, b.right]);
    stack.push([a.right, b.left]);
  }
  return true;
}

/** Independent oracle: build the mirror image of a tree. */
function mirror(root) {
  if (!root) return null;
  return new TreeNode(root.val, mirror(root.right), mirror(root.left));
}

/** Structural equality: same shape and same values everywhere. */
function identical(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return a.val === b.val && identical(a.left, b.left) && identical(a.right, b.right);
}

module.exports = { TreeNode, buildTree, isMirror, isSymmetric, isSymmetricIterative };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isSymmetric(null), true);
  assert.strictEqual(isSymmetric(buildTree([1])), true);
  assert.strictEqual(isSymmetric(buildTree([1, 2, 2])), true);
  assert.strictEqual(isSymmetric(buildTree([1, 2, 2, 3, 4, 4, 3])), true);
  // both 3s hang as RIGHT children, so the mirrored pairing fails
  assert.strictEqual(isSymmetric(buildTree([1, 2, 2, null, 3, null, 3])), false);
  assert.strictEqual(isSymmetric(buildTree([1, 2, 2, null, 3, 3, null])), true);
  assert.strictEqual(isSymmetric(buildTree([1, 2, 3])), false);
  for (const data of [[], [1], [1, 2, 2], [1, 2, 2, 3, 4, 4, 3], [1, 2, 2, null, 3, null, 3],
                      [1, 2, 2, null, 3, 3, null], [1, 2, 3], [1, 2, 2, 3, null, null, 3]]) {
    const tree = buildTree(data);
    const expected = identical(tree, mirror(tree));   // tree == its own mirror image
    assert.strictEqual(isSymmetric(tree), expected, String(data));
    assert.strictEqual(isSymmetricIterative(tree), expected, String(data));
  }
  console.log("PASS");
}
