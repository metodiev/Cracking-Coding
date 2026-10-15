/** Task 240 - Sum of nodes at each level (level order). (Tier: Intermediate, CTCI Ch 4) */
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

/**
 * Level-order BFS with a per-level snapshot sums each level independently:
 * one pass over the tree, O(n) time. The empty tree yields [].
 */
function levelSums(root) {
  if (root === null) return [];
  const sums = [];
  let queue = [root];
  while (queue.length) {
    let total = 0;
    const next = [];
    for (const node of queue) {
      total += node.val;
      if (node.left) next.push(node.left);
      if (node.right) next.push(node.right);
    }
    sums.push(total);
    queue = next;
  }
  return sums;
}

/** Independent helper: accumulate sums by depth with a DFS. */
function levelSumsDfs(root) {
  const sums = [];
  const visit = (node, depth) => {
    if (node === null) return;
    if (depth === sums.length) sums.push(0);
    sums[depth] += node.val;
    visit(node.left, depth + 1);
    visit(node.right, depth + 1);
  };
  visit(root, 0);
  return sums;
}

module.exports = { TreeNode, buildTree, levelSums, levelSumsDfs };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(levelSums(buildTree([1, 2, 3, 4, 5, null, 6])), [1, 5, 15]); // {1}, {2,3}, {4,5,6}
  assert.deepStrictEqual(levelSums(buildTree([1, 2, 3])), [1, 5]);
  assert.deepStrictEqual(levelSums(buildTree([1])), [1]);
  assert.deepStrictEqual(levelSums(null), []);
  assert.deepStrictEqual(levelSums(buildTree([1, 2, null, 3, null, 4])), [1, 2, 3, 4]);
  assert.deepStrictEqual(levelSums(buildTree([-1, -2, -3])), [-1, -5]);
  assert.deepStrictEqual(levelSums(buildTree([0, 0, 0, 0])), [0, 0, 0]);

  // deterministic LCG so the cross-check is reproducible
  let seed = 240;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const prune = (node) => {
    if (node === null || rand() < 0.25) return null;
    node.left = prune(node.left);
    node.right = prune(node.right);
    return node;
  };
  for (let test = 0; test < 200; test++) {      // BFS result == DFS depth accumulation
    const n = Math.floor(rand() * 15);
    const nodes = [];
    for (let i = 0; i < n; i++) nodes.push(new TreeNode(Math.floor(rand() * 41) - 20));
    for (let i = 0; i < n; i++) {
      if (2 * i + 1 < n) nodes[i].left = nodes[2 * i + 1];
      if (2 * i + 2 < n) nodes[i].right = nodes[2 * i + 2];
    }
    const root = n === 0 ? null : prune(nodes[0]);
    assert.deepStrictEqual(levelSums(root), levelSumsDfs(root));
  }
  console.log("PASS");
}
