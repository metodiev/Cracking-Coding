/** Task 153 - Count nodes in complete binary tree faster than O(n). (Tier: Beginner, CTCI Ch 4) */
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

function leftHeight(node) {
  let h = 0;
  while (node) { h++; node = node.left; }
  return h;
}

function rightHeight(node) {
  let h = 0;
  while (node) { h++; node = node.right; }
  return h;
}

/**
 * O(log^2 n) node count for a complete binary tree: when the left and right
 * spine heights match, the subtree is perfect and holds 2^h - 1 nodes;
 * otherwise recurse into both children.
 */
function countNodes(root) {
  if (!root) return 0;
  const lh = leftHeight(root);
  const rh = rightHeight(root);
  if (lh === rh) return 2 ** lh - 1;
  return 1 + countNodes(root.left) + countNodes(root.right);
}

/** Plain O(n) count used only to cross-check countNodes. */
function naiveCount(node) {
  if (!node) return 0;
  return 1 + naiveCount(node.left) + naiveCount(node.right);
}

module.exports = { TreeNode, buildTree, countNodes, naiveCount };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countNodes(null), 0);
  assert.strictEqual(countNodes(buildTree([1])), 1);

  // Level-order arrays without gaps build complete (sometimes perfect) trees.
  const sizes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 31, 63, 64, 65, 100, 1000];
  for (const n of sizes) {
    const root = buildTree(Array.from({ length: n }, (_, i) => i));
    assert.strictEqual(countNodes(root), n, `count == ${n}`);
    assert.strictEqual(countNodes(root), naiveCount(root), `fast == naive for n=${n}`);
  }

  assert.strictEqual(countNodes(buildTree([1, 2, 3, 4, 5, 6, 7])), 7);   // perfect
  const partial = buildTree([1, 2, 3, 4, 5, 6, null]);                   // partial last level
  assert.strictEqual(countNodes(partial), 6);
  assert.strictEqual(naiveCount(partial), 6);
  console.log("PASS");
}
