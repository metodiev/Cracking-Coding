/** Task 138 - Tree postorder traversal recursive. (Tier: Beginner, CTCI Ch 4) */
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

function postorder(node, out = []) {
  if (!node) return out;
  postorder(node.left, out);
  postorder(node.right, out);
  out.push(node.val);
  return out;
}

if (require.main === module) {
  const assert = require("node:assert");
  const t = buildTree([1, 2, 3, 4, 5, null, 6]);
  assert.deepStrictEqual(postorder(t), [4, 5, 2, 6, 3, 1]);
  assert.deepStrictEqual(postorder(null), []);
  assert.deepStrictEqual(postorder(buildTree([7])), [7]);
  console.log("PASS");
}
