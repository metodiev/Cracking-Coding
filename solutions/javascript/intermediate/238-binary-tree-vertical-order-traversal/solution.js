/** Task 238 - Binary tree vertical order traversal. (Tier: Intermediate, CTCI Ch 4) */
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
 * Columns from the root (root = 0, left = -1, right = +1). BFS with columns reads
 * each column top-to-bottom and visits same-row nodes left-to-right (tie rule);
 * the column keys are emitted in increasing order. O(n log n) time, O(n) space.
 */
function verticalOrder(root) {
  if (root === null) return [];
  const columns = new Map();
  const q = [[root, 0]];
  while (q.length) {
    const [node, col] = q.shift();
    if (!columns.has(col)) columns.set(col, []);
    columns.get(col).push(node.val);
    if (node.left) q.push([node.left, col - 1]);
    if (node.right) q.push([node.right, col + 1]);
  }
  return [...columns.keys()].sort((a, b) => a - b).map((col) => columns.get(col));
}

/** Independent helper: sort (column, row, preorder index) then group by column. */
function verticalOrderSorted(root) {
  const items = [];
  const dfs = (node, col, row) => {
    if (node === null) return;
    items.push([col, row, items.length, node.val]);
    dfs(node.left, col - 1, row + 1);
    dfs(node.right, col + 1, row + 1);
  };
  dfs(root, 0, 0);
  items.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const result = [];
  let lastCol = null;
  for (const [col, , , val] of items) {
    if (result.length === 0 || col !== lastCol) {
      result.push([]);
      lastCol = col;
    }
    result[result.length - 1].push(val);
  }
  return result;
}

module.exports = { TreeNode, buildTree, verticalOrder, verticalOrderSorted };

if (require.main === module) {
  const assert = require("node:assert");
  const classic = buildTree([3, 9, 20, null, null, 15, 7]);
  assert.deepStrictEqual(verticalOrder(classic), [[9], [3, 15], [20], [7]]);
  assert.deepStrictEqual(verticalOrderSorted(classic), [[9], [3, 15], [20], [7]]);

  const perfect = buildTree([1, 2, 3, 4, 5, 6, 7]);
  assert.deepStrictEqual(verticalOrder(perfect), [[4], [2], [1, 5, 6], [3], [7]]);
  assert.deepStrictEqual(verticalOrderSorted(perfect), [[4], [2], [1, 5, 6], [3], [7]]);

  const deeper = buildTree([3, 9, 20, null, null, 15, 7, 8, 11]);
  assert.deepStrictEqual(verticalOrder(deeper), [[9, 8], [3, 15], [20, 11], [7]]);
  assert.deepStrictEqual(verticalOrderSorted(deeper), [[9, 8], [3, 15], [20, 11], [7]]);

  const chain = buildTree([1, 2, null, 3, null, 4]);
  assert.deepStrictEqual(verticalOrder(chain), [[4], [3], [2], [1]]);

  assert.deepStrictEqual(verticalOrder(buildTree([5])), [[5]]);
  assert.deepStrictEqual(verticalOrder(null), []);

  // deterministic LCG so the cross-check is reproducible
  let seed = 238;
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
  for (let test = 0; test < 200; test++) {      // BFS order == sorted-DFS order
    const n = Math.floor(rand() * 15);
    const nodes = [];
    for (let i = 0; i < n; i++) nodes.push(new TreeNode(i + 1));
    for (let i = 0; i < n; i++) {
      if (2 * i + 1 < n) nodes[i].left = nodes[2 * i + 1];
      if (2 * i + 2 < n) nodes[i].right = nodes[2 * i + 2];
    }
    const root = n === 0 ? null : prune(nodes[0]);
    assert.deepStrictEqual(verticalOrder(root), verticalOrderSorted(root));
  }
  console.log("PASS");
}
