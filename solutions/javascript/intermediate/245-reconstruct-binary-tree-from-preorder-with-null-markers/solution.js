/** Task 245 - Reconstruct binary tree from preorder with null markers. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class TreeNode {
  constructor(val, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}

function buildTree(values) {
  if (values.length === 0 || values[0] === null) return null;
  const root = new TreeNode(values[0]);
  const q = [root];
  let i = 1;
  while (q.length && i < values.length) {
    const node = q.shift();
    if (i < values.length && values[i] !== null) { node.left = new TreeNode(values[i]); q.push(node.left); }
    i++;
    if (i < values.length && values[i] !== null) { node.right = new TreeNode(values[i]); q.push(node.right); }
    i++;
  }
  return root;
}

function preorder(node, out = []) {
  if (node) { out.push(node.val); preorder(node.left, out); preorder(node.right, out); }
  return out;
}

function inorder(node, out = []) {
  if (node) { inorder(node.left, out); out.push(node.val); inorder(node.right, out); }
  return out;
}

function buildFromPreorderTokens(csv) {
  const tokens = Array.isArray(csv) ? csv.map(String) : String(csv).split(",");
  let pos = 0;

  function build() {
    if (pos >= tokens.length || tokens[pos] === "#") {
      pos++;
      return null;
    }
    const node = new TreeNode(Number(tokens[pos++]));
    node.left = build();
    node.right = build();
    return node;
  }

  return tokens.length ? build() : null;
}

module.exports = { TreeNode, buildTree, preorder, inorder, buildFromPreorderTokens };

if (require.main === module) {
  const assert = require("node:assert");
  const root = buildFromPreorderTokens("1,2,#,#,3,#,#");
  assert.deepStrictEqual(preorder(root), [1, 2, 3]);
  assert.deepStrictEqual(inorder(root), [2, 1, 3]);

  const root2 = buildFromPreorderTokens("1,2,4,#,#,5,#,#,3,#,6,#,#");
  assert.deepStrictEqual(preorder(root2), [1, 2, 4, 5, 3, 6]);
  assert.deepStrictEqual(inorder(root2), [4, 2, 5, 1, 3, 6]);

  assert.strictEqual(buildFromPreorderTokens("#"), null);
  const single = buildFromPreorderTokens("9,#,#");
  assert.strictEqual(single.val, 9);
  console.log("PASS");
}
