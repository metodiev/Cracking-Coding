/** Task 150 - Sum of root-to-leaf numbers in tree. (Tier: Beginner, CTCI Ch 4)
 *
 * Every root-to-leaf path spells a decimal number (1 -> 2 -> 3 means 123). The DFS
 * carries the prefix built so far: acc * 10 + node.val, and adds the accumulated
 * value whenever a leaf closes a path.
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


/** Recursive DFS with the running prefix; an empty tree contributes 0. */
function sumNumbers(root) {
  function walk(node, acc) {
    if (!node) return 0;
    acc = acc * 10 + node.val;
    if (!node.left && !node.right) return acc;      // a leaf closes one number
    return walk(node.left, acc) + walk(node.right, acc);
  }
  return walk(root, 0);
}

/** Stack of [node, prefix] pairs; a leaf adds its prefix to the total. */
function sumNumbersIterative(root) {
  if (!root) return 0;
  let total = 0;
  const stack = [[root, root.val]];
  while (stack.length) {
    const [node, acc] = stack.pop();
    if (!node.left && !node.right) {
      total += acc;
      continue;
    }
    if (node.left) stack.push([node.left, acc * 10 + node.left.val]);
    if (node.right) stack.push([node.right, acc * 10 + node.right.val]);
  }
  return total;
}

module.exports = { TreeNode, buildTree, sumNumbers, sumNumbersIterative };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(sumNumbers(null), 0);
  assert.strictEqual(sumNumbersIterative(null), 0);
  assert.strictEqual(sumNumbers(buildTree([7])), 7);
  assert.strictEqual(sumNumbers(buildTree([1, 2, 3])), 25);
  assert.strictEqual(sumNumbers(buildTree([4, 9, 0, 5, 1])), 1026);
  assert.strictEqual(sumNumbers(buildTree([1, 0, 0])), 20);
  assert.strictEqual(sumNumbers(buildTree([1, 2, null, 3])), 123);
  assert.strictEqual(sumNumbers(buildTree([1, 2, 3, 4, 5])), 262);
  for (const data of [[], [7], [1, 2, 3], [4, 9, 0, 5, 1], [1, 0, 0], [1, 2, null, 3],
                      [1, 2, 3, 4, 5], [1, 9, 9, null, 1]]) {
    const tree = buildTree(data);
    assert.strictEqual(sumNumbers(tree), sumNumbersIterative(tree), String(data));
  }
  console.log("PASS");
}
