/** Task 246 - Serialize/deserialize binary search tree compactly. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class TreeNode {
  constructor(val, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}

function preorder(node, out = []) {
  if (node) { out.push(node.val); preorder(node.left, out); preorder(node.right, out); }
  return out;
}

function inorder(node, out = []) {
  if (node) { inorder(node.left, out); out.push(node.val); inorder(node.right, out); }
  return out;
}

function serialize(root) {
  return preorder(root).join(",");
}

function serializeWithNulls(root) {
  const tokens = [];
  (function walk(node) {
    if (!node) { tokens.push("#"); return; }
    tokens.push(String(node.val));
    walk(node.left);
    walk(node.right);
  })(root);
  return tokens.join(",");
}

function deserialize(data) {
  const values = data === "" ? [] : data.split(",").map(Number);
  let pos = 0;

  function build(lo, hi) {
    if (pos >= values.length) return null;
    const value = values[pos];
    if (value < lo || value > hi) return null;
    pos++;
    const node = new TreeNode(value);
    node.left = build(lo, value - 1);
    node.right = build(value + 1, hi);
    return node;
  }

  return build(-Infinity, Infinity);
}

function buildBst(values) {
  let root = null;
  for (const value of values) {
    if (!root) { root = new TreeNode(value); continue; }
    let node = root;
    for (;;) {
      if (value < node.val) {
        if (!node.left) { node.left = new TreeNode(value); break; }
        node = node.left;
      } else {
        if (!node.right) { node.right = new TreeNode(value); break; }
        node = node.right;
      }
    }
  }
  return root;
}

module.exports = { TreeNode, preorder, inorder, serialize, serializeWithNulls, deserialize, buildBst };

if (require.main === module) {
  const assert = require("node:assert");
  const values = [5, 3, 8, 2, 4, 7, 9];
  const root = buildBst(values);
  const data = serialize(root);
  assert.strictEqual(data, "5,3,2,4,8,7,9");
  assert.strictEqual(serializeWithNulls(root), "5,3,2,#,#,4,#,#,8,7,#,#,9,#,#");
  assert.ok(data.split(",").length < serializeWithNulls(root).split(",").length);
  const clone = deserialize(data);
  assert.deepStrictEqual(preorder(clone), [5, 3, 2, 4, 8, 7, 9]);
  assert.deepStrictEqual(inorder(clone), [2, 3, 4, 5, 7, 8, 9]);
  assert.strictEqual(deserialize(""), null);
  console.log("PASS");
}
