/** Task 147 - Lowest common ancestor in binary tree (general). (Tier: Beginner, CTCI Ch 4)
 *
 * No BST ordering is available, so a postorder search is used: a subtree reports
 * the first key it finds from below; the node where both subtrees report a key is
 * the split point and therefore the LCA. When either key is not part of the tree
 * at all, the answer is null.
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


/** Plain depth-first search: a general tree has no ordering to exploit. */
function find(root, val) {
  if (!root || root.val === val) return root;
  return find(root.left, val) || find(root.right, val);
}

/** Identity search: is that very node object part of this tree? */
function contains(root, target) {
  if (!root) return false;
  return root === target || contains(root.left, target) || contains(root.right, target);
}

/** Postorder LCA; null when either node is missing from the tree. */
function lowestCommonAncestor(root, p, q) {
  if (!root || !p || !q) return null;
  if (!contains(root, p) || !contains(root, q)) return null;
  return search(root, p, q);
}

/** Classic postorder: report a target from below, else report the split node. */
function search(node, p, q) {
  if (!node || node === p || node === q) return node;
  const left = search(node.left, p, q);
  const right = search(node.right, p, q);
  if (left && right) return node;        // p and q were found in different subtrees
  return left || right;
}

module.exports = { TreeNode, buildTree, find, contains, lowestCommonAncestor };

if (require.main === module) {
  const assert = require("node:assert");
  const tree = buildTree([3, 5, 1, 6, 2, 0, 8, null, null, 7, 4]);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 5), find(tree, 1)).val, 3);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 5), find(tree, 4)).val, 5);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 6), find(tree, 7)).val, 5);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 6), find(tree, 4)).val, 5);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 7), find(tree, 4)).val, 2);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 0), find(tree, 4)).val, 3);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 3), find(tree, 5)).val, 3);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 7), find(tree, 7)).val, 7);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 0), find(tree, 99)), null);
  assert.strictEqual(lowestCommonAncestor(tree, null, find(tree, 5)), null);
  assert.strictEqual(lowestCommonAncestor(null, find(tree, 5), find(tree, 1)), null);
  // same value, but a different node object is not part of this tree
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 3), new TreeNode(5)), null);
  const chain = buildTree([1, null, 2, null, 3]);
  assert.strictEqual(lowestCommonAncestor(chain, find(chain, 1), find(chain, 3)).val, 1);
  console.log("PASS");
}
