/** Task 152 - All root-to-leaf paths with given sum. (Tier: Beginner, CTCI Ch 8) */
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

/** All root-to-leaf paths summing to target, left subtree before right. */
function pathSumPaths(root, target) {
  const result = [];
  const path = [];

  function dfs(node, remaining) {
    if (!node) return;
    path.push(node.val);
    if (!node.left && !node.right) {
      if (remaining === node.val) result.push(path.slice());
    } else {
      dfs(node.left, remaining - node.val);
      dfs(node.right, remaining - node.val);
    }
    path.pop();
  }

  dfs(root, target);
  return result;
}

module.exports = { TreeNode, buildTree, pathSumPaths };

if (require.main === module) {
  const assert = require("node:assert");
  const root = buildTree([5, 4, 8, 11, null, 13, 4, 7, 2, null, null, 5, 1]);
  assert.deepStrictEqual(pathSumPaths(root, 22), [[5, 4, 11, 2], [5, 8, 4, 5]]);
  assert.deepStrictEqual(pathSumPaths(root, 26), [[5, 8, 13]]);
  assert.deepStrictEqual(pathSumPaths(root, 100), []);          // no matching path
  assert.deepStrictEqual(pathSumPaths(null, 0), []);            // empty tree
  assert.deepStrictEqual(pathSumPaths(buildTree([7]), 7), [[7]]);
  assert.deepStrictEqual(pathSumPaths(buildTree([7]), 1), []);
  assert.deepStrictEqual(pathSumPaths(buildTree([1, 2]), 1), []);  // one missing child
  assert.deepStrictEqual(pathSumPaths(buildTree([1, 2]), 3), [[1, 2]]);
  assert.deepStrictEqual(pathSumPaths(buildTree([-1, -2, -3]), -3), [[-1, -2]]);
  console.log("PASS");
}
