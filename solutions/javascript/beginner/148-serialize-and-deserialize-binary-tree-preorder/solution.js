/** Task 148 - Serialize and deserialize binary tree (preorder). (Tier: Beginner, CTCI Ch 4)
 *
 * Format (identical in all four languages): preorder values separated by commas,
 * with "#" marking every missing child, e.g. "1,2,#,#,3,#,#". An empty tree is "#"
 * and a single node is "7,#,#", so every tree with n nodes uses exactly 2n + 1
 * tokens and the stream can be rebuilt by a single left-to-right pass.
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


/** Preorder walk that also emits a sentinel for each missing child. */
function serialize(root) {
  const parts = [];
  (function walk(node) {
    if (!node) {
      parts.push("#");
      return;
    }
    parts.push(String(node.val));
    walk(node.left);
    walk(node.right);
  })(root);
  return parts.join(",");
}

/** Rebuild the tree from the preorder stream produced by serialize. */
function deserialize(data) {
  if (data === "") return null;
  const tokens = data.split(",");
  let pos = 0;                            // stream cursor shared with build
  function build() {
    const token = tokens[pos++];
    if (token === "#") return null;
    const node = new TreeNode(Number.parseInt(token, 10));
    node.left = build();                  // the left subtree comes next
    node.right = build();
    return node;
  }
  return build();
}

/** Preorder values only (the sentinels are dropped). */
function preorderValues(root) {
  if (!root) return [];
  return [root.val].concat(preorderValues(root.left), preorderValues(root.right));
}

/** Round-trip check: same shape and same values everywhere. */
function areIdentical(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return a.val === b.val && areIdentical(a.left, b.left) && areIdentical(a.right, b.right);
}

module.exports = { TreeNode, buildTree, serialize, deserialize, preorderValues, areIdentical };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(serialize(null), "#");
  assert.strictEqual(deserialize("#"), null);
  assert.strictEqual(deserialize(""), null);
  assert.strictEqual(serialize(buildTree([7])), "7,#,#");
  assert.strictEqual(serialize(buildTree([1, 2, null, 3])), "1,2,3,#,#,#,#");
  assert.strictEqual(serialize(buildTree([1, 2, 5, 3, 4, null, 6])), "1,2,3,#,#,4,#,#,5,#,6,#,#");
  assert.strictEqual(serialize(buildTree([1, 2, 3])).split(",").length, 7);
  for (const data of [[], [7], [1, 2, null, 3], [1, 2, 5, 3, 4, null, 6], [1, 2, 3, 4, 5, null, 6]]) {
    const tree = buildTree(data);
    const text = serialize(tree);
    const tokens = text.split(",").filter((t) => t !== "#");
    assert.deepStrictEqual(preorderValues(tree), tokens.map(Number), String(data));
    const rebuilt = deserialize(text);
    assert.strictEqual(areIdentical(tree, rebuilt), true, String(data));
    assert.strictEqual(serialize(rebuilt), text, String(data));
    if (tree) {
      assert.notStrictEqual(rebuilt, tree, "deserialize builds a fresh structure");
    }
  }
  console.log("PASS");
}
