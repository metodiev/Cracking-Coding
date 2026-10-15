/** Task 146 - Lowest common ancestor in BST (iterative). (Tier: Beginner, CTCI Ch 4)
 *
 * The BST ordering turns the search into a walk: while both keys sit on the same
 * side of the current node, step that way. The first node where the keys split
 * (or where one key IS the node) is the lowest common ancestor.
 * O(h) steps, O(1) extra space.
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


/** Locate a node by value using the BST ordering; null when it is absent. */
function find(root, val) {
  while (root && root.val !== val) {
    root = val < root.val ? root.left : root.right;
  }
  return root;
}

/** Iterative split-point walk; null when a key is missing from this BST. */
function lowestCommonAncestor(root, p, q) {
  if (!root || !p || !q) return null;
  let node = root;
  while (node) {
    if (p.val < node.val && q.val < node.val) {
      node = node.left;                   // both keys live in the left subtree
    } else if (p.val > node.val && q.val > node.val) {
      node = node.right;                  // both keys live in the right subtree
    } else {
      return node;                        // split point, or one key is node itself
    }
  }
  return null;
}

module.exports = { TreeNode, buildTree, find, lowestCommonAncestor };

if (require.main === module) {
  const assert = require("node:assert");
  const tree = buildTree([6, 2, 8, 0, 4, 7, 9, null, null, 3, 5]);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 2), find(tree, 8)).val, 6);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 2), find(tree, 4)).val, 2);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 0), find(tree, 5)).val, 2);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 4), find(tree, 5)).val, 4);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 7), find(tree, 9)).val, 8);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 2), find(tree, 2)).val, 2);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 0), find(tree, 0)).val, 0);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 0), find(tree, 3)).val, 2);
  assert.strictEqual(lowestCommonAncestor(tree, find(tree, 0), find(tree, 99)), null);
  assert.strictEqual(lowestCommonAncestor(tree, null, find(tree, 8)), null);
  assert.strictEqual(lowestCommonAncestor(null, find(tree, 2), find(tree, 8)), null);
  // 2 and 4 sit in opposite subtrees of the root 3; 5 is 4's ancestor
  const chain = buildTree([3, 1, 5, null, 2, 4]);
  assert.strictEqual(lowestCommonAncestor(chain, find(chain, 2), find(chain, 4)).val, 3);
  assert.strictEqual(lowestCommonAncestor(chain, find(chain, 1), find(chain, 2)).val, 1);
  assert.strictEqual(lowestCommonAncestor(chain, find(chain, 4), find(chain, 5)).val, 5);
  console.log("PASS");
}
